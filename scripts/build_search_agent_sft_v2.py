#!/usr/bin/env python3
"""Search-agent SFT v2 — iterative batch-listening wavetable search.

Each search agent conversation:
  1. Hears the target audio once
  2. Iterates through batches of candidate wavetable probes (audio + name)
  3. Maintains a running shortlist across batches
  4. Returns a final set of wavetable names that "sound like they belong"

The agent's skill: comparative perceptual evaluation of raw wavetable probes
against a fully-processed target. At inference, this generalizes to any
wavetable library — no memorization of names needed, just ears.

Usage:
    python scripts/build_search_agent_sft_v2.py \
        --manifest outputs/smoke_test_v10/manifest.jsonl \
        --index-npy outputs/wt_retrieval_baseline/wt_index.npz \
        --index-meta outputs/wt_retrieval_baseline/wt_index_meta.json \
        --wavetable-lib data/wavetable_lib.json \
        --out-jsonl outputs/search_agent_v2/search.jsonl \
        --omni-server http://localhost:8000 \
        --max-samples 4
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.agent_sft_common import (
    DawFarmRolloutCtx,
    assert_valid_ms_swift_multiturn_record,
    build_clap_shortlist_data,
    build_gt_similarity_pool,
    build_list_wavetables_slice_snippet,
    build_name_embedding_map,
    build_render_probes_snippet,
    emit_code_mistake_sequence,
    ensure_candidate_probes_for_names,
    execute_for_traceback,
    extract_gt_wavetable_names,
    is_clap_selected,
    load_index_rows,
    load_manifest_entries,
    load_wavetable_lib,
    make_agent_id,
    select_and_apply_mutation,
    select_probe_rows_by_name,
    _bash_tool_response,
    _read_tool_response_audio,
    _tool_call as _tool_call_common,
    _wrap_as_bash,
)
from scripts.build_main_agent_sft_v2 import (
    _check_server_reachable,
    _llm_post,
)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _b64(path: str | Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


from maestro.synth.path_gen import _normalize


def describe_key_transforms(target_preset: dict) -> str:
    """Summarize the most impactful processing transforms in the target preset.

    Reads actual filter, envelope, modulation, and FX settings from the preset
    to produce a dynamic, per-sample description of what transforms the raw
    wavetable into the target sound.
    """
    s = target_preset.get("settings", {})
    parts: list[str] = []

    # Filter character
    f1_on = float(s.get("filter_fx_on", 0)) > 0.5 or float(s.get("filter_1_on", 1)) > 0.5
    cutoff_norm = _normalize("filter_1_cutoff", float(s.get("filter_1_cutoff", 80)))
    res_norm = _normalize("filter_1_resonance", float(s.get("filter_1_resonance", 0)))
    if cutoff_norm is not None and cutoff_norm < 0.65 and f1_on:
        res_desc = f" with {'high' if res_norm and res_norm > 0.4 else 'moderate'} resonance" if res_norm and res_norm > 0.15 else ""
        parts.append(f"filter darkens the tone (cutoff at {cutoff_norm:.0%}){res_desc}")
    elif cutoff_norm is not None and cutoff_norm > 0.75 and f1_on:
        parts.append(f"filter opens bright (cutoff at {cutoff_norm:.0%})")

    # Envelope 1 (amplitude) shape
    env1_attack = float(s.get("env_1_attack", 0))
    env1_decay = float(s.get("env_1_decay", 0))
    env1_sustain_norm = _normalize("env_1_sustain", float(s.get("env_1_sustain", 1.0)))
    if env1_attack > 0.5:
        parts.append(f"slow swelling attack ({env1_attack:.1f}s)")
    elif env1_attack < 0.02:
        parts.append("sharp, immediate attack")
    if env1_sustain_norm is not None and env1_sustain_norm < 0.3 and env1_decay < 0.5:
        parts.append("short plucky decay")
    elif env1_sustain_norm is not None and env1_sustain_norm > 0.7:
        parts.append("sustained, held tone")

    # Envelope 2 (modulation) — often shapes filter
    env2_attack = float(s.get("env_2_attack", 0))
    if env2_attack > 0.3:
        parts.append("modulation envelope with slow onset")

    # Unison
    voices = int(float(s.get("osc_1_unison_voices", 1)))
    detune_norm = _normalize("osc_1_unison_detune", float(s.get("osc_1_unison_detune", 0)))
    if voices > 1 and detune_norm and detune_norm > 0.05:
        parts.append(f"{voices}-voice unison with detune")

    # Active modulation routes (top 3)
    mod_parts: list[str] = []
    for mod in s.get("modulations", []):
        src = mod.get("source", "")
        dst = mod.get("destination", "")
        if src and dst:
            # Clean up source/destination names for readability
            mod_parts.append(f"{src} → {dst}")
            if len(mod_parts) >= 3:
                break
    if mod_parts:
        parts.append("modulation: " + ", ".join(mod_parts))

    # Active effects
    fx_active: list[str] = []
    for fx in ("chorus", "reverb", "delay", "distortion", "phaser", "flanger", "compressor"):
        key = f"{fx}_on" if fx != "compressor" else "compressor_enabled"
        if float(s.get(key, 0)) > 0.5:
            fx_active.append(fx)
    if fx_active:
        parts.append(f"effects: {', '.join(fx_active)}")

    if not parts:
        return "minimal processing — the raw wavetable carries most of the target character"
    return "; ".join(parts[:6])


# ---------------------------------------------------------------------------
# Omni calls for search agent
# ---------------------------------------------------------------------------


def omni_describe_single_audio(
    audio_path: Path,
    prompt_text: str,
    omni_server: str,
    omni_model: str,
    max_tokens: int = 160,
    temperature: float = 0.4,
    timeout: float = 60.0,
    fallback: str = "",
) -> str:
    """Single-audio Omni call. Sends ONE audio clip + ONE text prompt, returns
    the model's completion stripped. Used in place of multi-audio batch calls
    to avoid position-confusion hallucinations where the model scrambles
    which description belongs to which clip.
    """
    content = [
        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{_b64(audio_path)}"}},
        {"type": "text", "text": prompt_text},
    ]
    try:
        r = _llm_post(
            f"{omni_server}/v1/chat/completions",
            {"model": omni_model, "messages": [{"role": "user", "content": content}],
             "max_tokens": max_tokens, "temperature": temperature},
            timeout=timeout,
        )
        return r["choices"][0]["message"]["content"].strip()
    except Exception:
        return fallback


def describe_target(
    target_wav: Path,
    archetype: str,
    omni_server: str,
    omni_model: str,
) -> str:
    """One call: describe the target sound's timbre. Cached per-record, reused across batches."""
    return omni_describe_single_audio(
        audio_path=target_wav,
        prompt_text=(
            f"This is a target {archetype} sound we're trying to recreate with a Vital synth.\n"
            f"Describe its TIMBRE in one or two sentences: harmonic content (pure/rich/bright/dull), "
            f"texture (smooth/buzzy/metallic/vocal), attack shape (sharp/slow), and any distinctive "
            f"effects you hear (reverb, chorus, distortion). Focus on timbral qualities only — "
            f"do not mention note pattern or melody."
        ),
        omni_server=omni_server,
        omni_model=omni_model,
        max_tokens=180,
        fallback=f"A {archetype} sound.",
    )


_CAND_DESC_CACHE: dict[str, str] = {}
_CAND_DESC_LOCK = threading.Lock()


def describe_candidate(
    candidate_name: str,
    candidate_wav: Path,
    omni_server: str,
    omni_model: str,
) -> str:
    """One call: describe a candidate wavetable's timbre in isolation.

    Memoized process-globally by wav path: candidate descriptions are
    target-independent and the shared probe wavs are identical across
    samples, agents, and rounds — without the memo a full run re-describes
    the same audio thousands of times and saturates the omni server.
    """
    _key = f"{candidate_wav}::{omni_model}"
    with _CAND_DESC_LOCK:
        if _key in _CAND_DESC_CACHE:
            return _CAND_DESC_CACHE[_key]
    _desc = _describe_candidate_uncached(candidate_name, candidate_wav, omni_server, omni_model)
    with _CAND_DESC_LOCK:
        _CAND_DESC_CACHE[_key] = _desc
    return _desc


def _describe_candidate_uncached(
    candidate_name: str,
    candidate_wav: Path,
    omni_server: str,
    omni_model: str,
) -> str:
    return omni_describe_single_audio(
        audio_path=candidate_wav,
        prompt_text=(
            f"This is a wavetable rendered through a default Vital preset (probe: 4 triads "
            f"over ~10s). The note pattern is the same across all probes; what varies is the "
            f"wavetable's TIMBRE.\n\n"
            f"In ONE sentence, describe only this wavetable's timbral character: harmonic content "
            f"(pure/rich/inharmonic), brightness, texture (smooth/buzzy/metallic), and attack shape. "
            f"Ignore note pattern. Do not guess the wavetable's name."
        ),
        omni_server=omni_server,
        omni_model=omni_model,
        max_tokens=120,
        fallback="Candidate wavetable.",
    )


def synthesize_batch_observations(
    target_description: str,
    candidate_descriptions: dict[str, str],
    archetype: str,
    stage2_server: str,
    stage2_model: str,
) -> str:
    """Text-only call: given the target's timbre description and per-candidate
    descriptions (both obtained via single-audio calls), write per-candidate
    building-block assessments. Drops back to the existing `'<name>': <text>`
    line format so downstream Stage 2 can merge with selection labels."""
    cand_block = "\n".join(
        f"  '{name}': {desc}" for name, desc in candidate_descriptions.items()
    )
    prompt = (
        f"You are a sound design assistant evaluating wavetable candidates for a {archetype} target.\n\n"
        f"Target timbre (described from a separate isolated listen):\n"
        f"  {target_description}\n\n"
        f"Candidate wavetables (each described in isolation):\n"
        f"{cand_block}\n\n"
        f"For EACH candidate, write ONE sentence about how its raw timbre relates to the target — "
        f"could this wavetable be a building block for recreating the target once filters, "
        f"envelopes, and effects are applied?\n\n"
        f"Format one line per candidate, in the order above:\n"
        f"  '<name>': <one sentence tying its timbre to the target>\n\n"
        f"Use the EXACT names. Do not mention the note pattern or number of notes."
    )
    try:
        r = _llm_post(
            f"{stage2_server}/v1/chat/completions",
            {"model": stage2_model,
             "messages": [{"role": "user", "content": prompt}],
             "max_tokens": 500, "temperature": 0.4},
            timeout=120.0,
        )
        return r["choices"][0]["message"]["content"].strip()
    except Exception:
        return "\n".join(
            f"'{n}': {d}" for n, d in candidate_descriptions.items()
        )


def stage2_batch_notes_merged(
    target_description: str,
    candidate_descriptions: dict[str, str],
    candidate_names: list[str],
    selection_labels: dict[str, bool],
    batch_number: int,
    archetype: str,
    stage2_server: str,
    stage2_model: str,
    gt_transform_description: str = "",
) -> str:
    """Single-call variant folding synthesize_batch_observations into the
    notes prompt: raw describe texts go in directly and the model relates
    them to the target while writing the label-aligned reasoning. Halves the
    per-batch stage-2 call count (--merged-stage2)."""
    cand_block = "\n".join(
        f"  '{n}': {candidate_descriptions.get(n, 'Candidate wavetable.')} "
        f"[{'Selected' if selection_labels.get(n, False) else 'Not selected'}]"
        for n in candidate_names
    )
    gt_context_block = ""
    if gt_transform_description:
        gt_context_block = (
            f"\nTarget processing chain (for grounding your reasoning about Selected "
            f"candidates — DO NOT quote it verbatim, DO NOT mention 'GT' or identify "
            f"any candidate as the correct answer):\n  {gt_transform_description}\n"
        )
    prompt = (
        f"You are a sound design assistant evaluating wavetable candidates for a {archetype} target.\n\n"
        f"Target timbre (from an isolated listen):\n  {target_description}\n\n"
        f"Candidates, each with an isolated-listen description and a pre-determined "
        f"selection status:\n{cand_block}\n{gt_context_block}\n"
        f"Write EXACTLY {len(candidate_names)} lines, one per candidate, in the order above.\n"
        f"Format: '<name>': <one sentence relating its raw timbre to the target and "
        f"justifying the status>. Selected / Not selected.\n\n"
        f"Rules:\n"
        f"- Use the EXACT names above.\n"
        f"- Selected: what raw quality makes it a plausible building block once filters, "
        f"envelopes and effects are applied (reference one specific transform).\n"
        f"- Not selected: what makes it incompatible.\n"
        f"- DO NOT use 'GT', 'ground truth', 'oracle', or identify a known right answer.\n"
        f"- Copy the Selected/Not selected label exactly as given.\n"
        f"- Do not mention the note pattern or number of notes."
    )
    try:
        r = _llm_post(
            f"{stage2_server}/v1/chat/completions",
            {"model": stage2_model, "messages": [{"role": "user", "content": prompt}],
             "max_tokens": 500, "temperature": 0.4},
            timeout=120.0,
        )
        return r["choices"][0]["message"]["content"].strip()
    except Exception:
        return "\n".join(
            f"'{n}': Candidate evaluated. "
            f"{'Selected' if selection_labels.get(n, False) else 'Not selected'}."
            for n in candidate_names
        )


def stage2_batch_notes(
    omni_observations: str,
    candidate_names: list[str],
    selection_labels: dict[str, bool],
    gt_names_in_batch: list[str],
    batch_number: int,
    archetype: str,
    stage2_server: str,
    stage2_model: str,
    gt_transform_description: str = "",
) -> str:
    """Stage 2: write per-candidate reasoning. Selection labels are pre-determined by CLAP.

    Stage 2 sees the labels (Selected / Not selected) and writes reasoning
    that ALIGNS with the correct decision. It never decides the selection itself.

    GT-grounding is provided as out-of-line context, NOT as an inline marker on
    the candidate line — Stage-2 must NOT emit the literal "GT" or copy the
    processing-chain description verbatim. It uses the chain to inform its
    reasoning about the Selected candidates without naming which are GT.
    """
    # Build candidate display with pre-determined labels — NO GT marker.
    candidate_lines = [
        f"  '{name}': {('Selected' if selection_labels.get(name, False) else 'Not selected')}"
        for name in candidate_names
    ]
    candidates_block = "\n".join(candidate_lines)

    # Out-of-line GT-grounding context — used to inform reasoning but never
    # to identify which candidate is "right."
    gt_context_block = ""
    if gt_transform_description:
        gt_context_block = (
            f"\nTarget processing chain (for grounding your reasoning about Selected "
            f"candidates — DO NOT quote this description verbatim, DO NOT mention "
            f"'GT' or 'ground truth', and DO NOT identify any candidate as the "
            f"correct answer):\n  {gt_transform_description}\n"
        )

    prompt = (
        f"You are a sound design assistant evaluating wavetable candidates for "
        f"the target sound.\n\n"
        f"Audio observations from listening:\n{omni_observations}\n"
        f"{gt_context_block}\n"
        f"Candidates and their selection status (pre-determined):\n{candidates_block}\n\n"
        f"Write EXACTLY {len(candidate_names)} lines, one per candidate.\n"
        f"Format: '<name>': <one sentence assessment>. Selected / Not selected.\n\n"
        f"Rules:\n"
        f"- Use the EXACT names above (no variants, no abbreviations).\n"
        f"- For Selected candidates: explain what raw quality makes it compatible "
        f"with the target after synthesis processing. Reference at least one specific "
        f"transform from the processing chain (e.g. 'a low-pass filter at moderate "
        f"cutoff would tame this brightness') without quoting the chain verbatim.\n"
        f"- For Not selected candidates: explain what makes it incompatible.\n"
        f"- DO NOT use the words 'GT', 'ground truth', 'oracle', 'correct answer', or "
        f"any phrasing that identifies a candidate as the known right one.\n"
        f"- Copy the Selected/Not selected label exactly as given above."
    )
    try:
        r = _llm_post(
            f"{stage2_server}/v1/chat/completions",
            {"model": stage2_model, "messages": [{"role": "user", "content": prompt}],
             "max_tokens": 400, "temperature": 0.4},
            timeout=120.0,
        )
        return r["choices"][0]["message"]["content"].strip()
    except Exception:
        # Fallback: simple labels with no reasoning
        lines = []
        for name in candidate_names:
            label = "Selected" if selection_labels.get(name, False) else "Not selected"
            lines.append(f"'{name}': Candidate evaluated. {label}.")
        return "\n".join(lines)


def stage2_final_summary(
    shortlist: list[str],
    archetype: str,
    prior_notes: list[str],
    stage2_server: str,
    stage2_model: str,
) -> str:
    """Stage 2: write a short closing narration summarising what each shortlisted
    candidate contributes. No list-echo line — the shortlist surfaces via the
    subsequent file write (claw-code-style handoff)."""
    names_str = ", ".join(f"'{n}'" for n in shortlist)
    recent_notes = "\n".join(prior_notes[-2:]) if prior_notes else ""
    prompt = (
        f"You are a sound design assistant wrapping up a slice evaluation. "
        f"Your final shortlist for this slice is: [{names_str}].\n\n"
        f"Recent assessment notes:\n{recent_notes}\n\n"
        f"Write EXACTLY ONE sentence (under 30 words) explaining what each shortlisted "
        f"candidate contributes to the target, referencing their individual sonic "
        f"character (e.g. 'X provides the harmonic body, Y adds the brightness'). "
        f"Be specific to these particular wavetables — do NOT use generic phrases like "
        f"'rich harmonic content' or 'dynamic evolution'. Do NOT re-list the names as "
        f"a bracketed JSON array — the list will appear in the output file next."
    )
    try:
        r = _llm_post(
            f"{stage2_server}/v1/chat/completions",
            {"model": stage2_model, "messages": [{"role": "user", "content": prompt}],
             "max_tokens": 120, "temperature": 0.4},
            timeout=120.0,
        )
        return r["choices"][0]["message"]["content"].strip()
    except Exception:
        return (
            f"Shortlisted candidates each offer distinct building-block qualities "
            f"for the target sound."
        )


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """Return type for build_search_record(): the SFT record + the shortlist
    that the main agent would see when it reads this agent's output file."""
    record: dict | None
    shortlist: list[str]
    shard_start: int
    shard_end: int


_TOOL_SPECS = json.dumps([
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Execute shell/Python commands to render and evaluate wavetable candidates.",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read a file. For audio files, returns the audio content for listening.",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string", "description": "Absolute path to the file."}},
                "required": ["file_path"],
            },
        },
    },
], ensure_ascii=False)


def _tool_call(name: str, arguments: dict) -> dict:
    return _tool_call_common(name, arguments)


def build_search_record(
    *,
    sample_id: str,
    agent_idx: int,
    archetype: str,
    target_audio_path: Path,
    target_preset: dict,
    gt_wavetable_names: list[str],
    shard_start: int,
    shard_end: int,
    name_to_idx: dict[str, int],
    idx_to_name: dict[int, str],
    candidate_audio: dict[str, Path],
    name_to_emb: dict[str, "np.ndarray"],
    omni_server: str,
    omni_model: str,
    stage2_server: str,
    stage2_model: str,
    candidates_per_batch: int = 8,
    shortlist_dir: Path | None = None,
    clap_threshold: float = 0.97,
    midi_path: str | None = None,
    code_mistake_rate: float = 0.0,
    seed: int = 1337,
    probe_audio_dir: Path | None = None,
    dawfarm: "DawFarmRolloutCtx | None" = None,
    force_miss_names: list[str] | None = None,
    merged_stage2: bool = False,
) -> SearchResult:
    """Build one search agent SFT record: render all probes upfront, then
    listen and assess in batches.

    Renders all candidates in [shard_start, shard_end) in a single DawDreamer
    bash call, then iterates through batches of Read (listen) + analysis.
    """
    _empty = SearchResult(record=None, shortlist=[], shard_start=shard_start, shard_end=shard_end)
    if shard_end <= shard_start:
        return _empty

    # Names in shard (by dense index)
    shard = [idx_to_name[i] for i in range(shard_start, shard_end) if i in idx_to_name]
    if not shard:
        return _empty

    gt_set = set(gt_wavetable_names)
    gt_in_shard = [n for n in shard if n in gt_set]

    # Compute processing chain description for GT grounding
    gt_transform_desc = describe_key_transforms(target_preset) if gt_in_shard else ""

    # Sequential batching: walk the shard in index order, chunking into
    # fixed-size batches. GT (if present in shard) falls wherever its index
    # places it — no special-casing. `is_clap_selected` returns True for
    # exact GT matches, and `selected_so_far` is append-only, so once GT is
    # found it stays in the shortlist for the rest of the slice.
    all_ordered = [
        shard[i:i + candidates_per_batch]
        for i in range(0, len(shard), candidates_per_batch)
    ]
    if not all_ordered:
        return None

    # If GT wasn't in shard at all, that's fine — the agent legitimately finds nothing
    # from this shard that matches. This is realistic for some shards.

    # Build messages
    messages: list[dict] = []
    audio_assets: list[str] = [str(target_audio_path)]
    all_batch_notes: list[str] = []

    # Output directory for rendered probes. In daw-farm mode the snippet
    # writes to the container rollout dir (what the model sees at inference)
    # and rendered wavs are fetched back to *probe_out_dir* for `audios`.
    if probe_audio_dir is not None:
        probe_out_dir = str(probe_audio_dir / f"{sample_id}_agent{agent_idx}")
    else:
        probe_out_dir = f"/tmp/search_probes/{sample_id}_agent{agent_idx}"
    display_probe_dir = (
        dawfarm.cw("", f"search_probes_a{agent_idx}") if dawfarm is not None
        else probe_out_dir
    )

    _user_parts = [f"<audio>"]
    if midi_path:
        _user_parts.append(f"Transcription MIDI: {midi_path}.")
    _user_parts.append(
        f"Evaluate wavetables at indices {shard_start}-{shard_end - 1}. "
        f"Scan Vital's data directories for .vitaltable and .vital files to get names in your range, "
        f"render all candidates at once, then listen to each batch and assess. "
        f"Return a JSON shortlist of 2-4 wavetable names."
    )
    messages.append({
        "role": "user",
        "content": "\n".join(_user_parts),
    })

    # Step 1: list_wavetables — agent fetches names in its range
    list_cmd = _wrap_as_bash(build_list_wavetables_slice_snippet(shard_start, shard_end))
    shard_entries = [
        {"idx": i, "name": idx_to_name[i]}
        for i in range(shard_start, shard_end) if i in idx_to_name
    ]
    messages.append({
        "role": "assistant",
        "content": f"Fetching candidate names at indices {shard_start}-{shard_end - 1}.",
    })
    messages.append(_tool_call("Bash", {"command": list_cmd}))
    if dawfarm is not None:
        _lres = dawfarm.real_exec(list_cmd, f"wavetable slice {shard_start}-{shard_end}")
        try:
            _container_names = [w["name"] for w in json.loads(_lres.stdout)["wavetables"]]
        except Exception as exc:
            raise RuntimeError(f"unparseable slice listing: {exc}: {_lres.stdout[:300]}")
        _host_names = [e["name"] for e in shard_entries]
        if _container_names != _host_names:
            # Library-order or sync divergence corrupts every idx-keyed step.
            raise RuntimeError(
                f"container wavetable slice {shard_start}-{shard_end} disagrees with host "
                f"library (first diff at "
                f"{next((i for i, (a, b) in enumerate(zip(_container_names, _host_names)) if a != b), 'len')})"
            )
        messages.append(_bash_tool_response(_lres.stdout))
    else:
        _list_stdout = json.dumps({
            "wavetables": shard_entries,
            "start": shard_start,
            "end": shard_end,
            "count": len(shard_entries),
        }) + "\n"
        messages.append(_bash_tool_response(_list_stdout))

    selected_so_far: list[str] = []  # CLAP-selected candidates, accumulated across batches
    pending_notes: str | None = None

    # Single-audio policy (anti-hallucination): describe the target ONCE via a
    # single-audio Omni call. This description is reused as context across all
    # batches. Per-candidate descriptions happen inside each batch below, also
    # via single-audio calls. The per-candidate and target descriptions are then
    # synthesized into the assistant-visible observation text by a text-only call.
    target_description = (
        describe_target(
            target_wav=target_audio_path,
            archetype=archetype,
            omni_server=omni_server,
            omni_model=omni_model,
        ) if omni_server else f"A {archetype} sound."
    )

    # --- Render ALL probes upfront in one bash call ---
    import re as _re
    def _slugify(s: str) -> str:
        return (_re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_") or "unnamed")[:80]

    all_shard_idxs = [name_to_idx[n] for n in shard if n in name_to_idx]
    render_snippet = build_render_probes_snippet(
        idxs=all_shard_idxs, out_dir=display_probe_dir,
        midi_path=midi_path,
    )
    render_cmd = _wrap_as_bash(render_snippet)

    # Conversation shows display paths (container in daw-farm mode); `audios`
    # carries the host copies (fetched from the container after the render).
    all_rendered_entries = []
    name_to_audio_read_path: dict[str, str] = {}
    name_to_audio_host_path: dict[str, str] = {}
    for name in shard:
        idx = name_to_idx.get(name)
        if idx is None:
            continue
        fname = f"wt_{idx:04d}_{_slugify(name)}.wav"
        display_path = f"{display_probe_dir}/{fname}"
        host_path = f"{probe_out_dir}/{fname}"
        all_rendered_entries.append({"idx": idx, "name": name, "out": display_path})
        audio_assets.append(host_path)
        name_to_audio_read_path[name] = display_path
        name_to_audio_host_path[name] = host_path

    messages.append({
        "role": "assistant",
        "content": f"Rendering all {len(shard)} candidates in my range.",
    })

    # Code mistake injection: mutate probes render and capture real traceback.
    import random as _random
    _code_mistake_info: dict | None = None
    _sid_seed = int(hashlib.sha1(f"{sample_id}:{agent_idx}".encode()).hexdigest()[:8], 16)
    _code_rng = _random.Random(seed + _sid_seed + 9999)
    if _code_rng.random() < code_mistake_rate:
        _mut_result = select_and_apply_mutation(
            "probes", render_snippet, _code_rng,
            reaper_available=dawfarm is not None,
        )
        if _mut_result is not None:
            _mutation, _broken_code = _mut_result
            _traceback = execute_for_traceback(
                _broken_code, cwd=str(ROOT),
                session=dawfarm.session if dawfarm is not None else None,
            )
            if _traceback:
                emit_code_mistake_sequence(
                    messages, _broken_code, render_snippet, _traceback, _mutation,
                )
                _code_mistake_info = {"mutation": _mutation.name, "category": _mutation.category, "target": "probes_render"}

    messages.append(_tool_call("Bash", {"command": render_cmd}))
    if dawfarm is not None:
        # One exec renders the whole shard in a single DawDreamer engine spawn.
        _rres = dawfarm.real_exec(render_cmd, f"probe render agent{agent_idx}",
                                  timeout=max(dawfarm.exec_timeout, 600.0))
        try:
            _real_rendered = {e["name"] for e in json.loads(_rres.stdout)["rendered"]}
        except Exception as exc:
            raise RuntimeError(f"unparseable probe render stdout: {exc}: {_rres.stdout[:300]}")
        _expected = {e["name"] for e in all_rendered_entries}
        if _real_rendered != _expected:
            raise RuntimeError(
                f"probe render mismatch agent{agent_idx}: "
                f"missing={sorted(_expected - _real_rendered)[:5]} "
                f"extra={sorted(_real_rendered - _expected)[:5]}")
        dawfarm.fetch_dir(display_probe_dir, probe_out_dir)
        _missing = [p for p in name_to_audio_host_path.values() if not Path(p).exists()]
        if _missing:
            raise RuntimeError(f"probe fetch incomplete agent{agent_idx}: missing {len(_missing)} "
                               f"(first: {_missing[0]})")
        messages.append(_bash_tool_response(_rres.stdout))
    else:
        _render_stdout = json.dumps({"status": "ok", "rendered": all_rendered_entries}) + "\n"
        messages.append(_bash_tool_response(_render_stdout))

    # --- Per-batch: listen (Read) + analyze ---
    for bi, batch in enumerate(all_ordered):
        batch_num = bi + 1
        gt_in_batch = [n for n in batch if n in gt_set]

        batch_audio_paths = [name_to_audio_read_path[n] for n in batch if n in name_to_audio_read_path]

        intro = f"Listening to batch {batch_num}."
        if pending_notes:
            intro = f"{pending_notes}\n\n{intro}"
            pending_notes = None

        if batch_audio_paths:
            messages.append({"role": "assistant", "content": intro})
            for rp in batch_audio_paths:
                messages.append(_tool_call("Read", {"file_path": rp}))
                messages.append(_read_tool_response_audio())

        # CLAP-based selection: determine Selected/Not selected for each candidate
        selection_labels: dict[str, bool] = {}
        for name in batch:
            selection_labels[name] = is_clap_selected(
                name, gt_wavetable_names, name_to_emb, threshold=clap_threshold
            )
            if force_miss_names and name in force_miss_names:
                # Forced perceptual miss (round-1 re-search training signal):
                # the agent auditions this GT but does not shortlist it.
                selection_labels[name] = False

        # Accumulate selected candidates (builder-controlled, no model parsing needed)
        batch_selected = [n for n in batch if selection_labels[n]]
        for n in batch_selected:
            if n not in selected_so_far:
                selected_so_far.append(n)

        # Omni Stage 1 (single-audio policy): describe each candidate in
        # isolation, then synthesize against the cached target description.
        if omni_server:
            valid_batch = [n for n in batch if n in candidate_audio]
            cand_descs = {}
            if valid_batch:
                with ThreadPoolExecutor(max_workers=min(8, len(valid_batch))) as ex:
                    futures = {
                        ex.submit(
                            describe_candidate,
                            name,
                            candidate_audio[name],
                            omni_server,
                            omni_model,
                        ): name
                        for name in valid_batch
                    }
                    for fut in as_completed(futures):
                        cand_descs[futures[fut]] = fut.result()
            # Preserve batch order in the dict (Python 3.7+ dicts keep insertion order)
            ordered_descs = {n: cand_descs.get(n, "Candidate wavetable.") for n in valid_batch}
            if not merged_stage2:
                omni_obs = synthesize_batch_observations(
                    target_description=target_description,
                    candidate_descriptions=ordered_descs,
                    archetype=archetype,
                    stage2_server=stage2_server,
                    stage2_model=stage2_model,
                )
        else:
            ordered_descs = {}
            omni_obs = "\n".join(f'"{n}": Candidate evaluated.' for n in batch)

        # Stage 2: write reasoning ONLY (labels are pre-determined by CLAP).
        if merged_stage2 and omni_server:
            batch_notes = stage2_batch_notes_merged(
                target_description=target_description,
                candidate_descriptions=ordered_descs,
                candidate_names=batch,
                selection_labels=selection_labels,
                batch_number=batch_num,
                archetype=archetype,
                stage2_server=stage2_server,
                stage2_model=stage2_model,
                gt_transform_description=gt_transform_desc,
            )
        else:
            batch_notes = stage2_batch_notes(
                omni_observations=omni_obs,
                candidate_names=batch,
                selection_labels=selection_labels,
                gt_names_in_batch=gt_in_batch,
                batch_number=batch_num,
                archetype=archetype,
                stage2_server=stage2_server,
                stage2_model=stage2_model,
                gt_transform_description=gt_transform_desc,
            )

        all_batch_notes.append(batch_notes)
        pending_notes = batch_notes

    # Final narration (merge with pending batch notes to avoid adjacent assistants)
    if omni_server:
        final_text = stage2_final_summary(selected_so_far, archetype, all_batch_notes, stage2_server, stage2_model)
    else:
        final_text = (
            f"Shortlisted candidates each offer distinct building-block qualities "
            f"for the target sound."
        )

    if pending_notes:
        final_text = f"{pending_notes}\n\n{final_text}"
        pending_notes = None

    # Final assistant turn — summarize result (the framework captures this as output)
    shortlist_str = ", ".join(f'"{n}"' for n in selected_so_far)
    final_assistant_content = (
        f"{final_text}\n\n"
        f"Shortlist: [{shortlist_str}]. {len(selected_so_far)} candidate(s) "
        f"flagged for the judge agent."
    )

    # Persist the final assistant message to disk (build-time) so the main
    # agent's `cat` works.  In claw-code the framework writes the sub-agent's
    # last response to the outputFile — plain text, not JSON.
    shortlist_path: str | None = None
    if shortlist_dir is not None:
        from scripts.agent_sft_common import make_agent_id  # type: ignore
        shortlist_dir.mkdir(parents=True, exist_ok=True)
        agent_id = make_agent_id(sample_id, "wavetable_search", agent_idx)
        shortlist_path = str(shortlist_dir / f"{agent_id}.md")
        with open(shortlist_path, "w") as f:
            f.write(final_assistant_content)
            f.write("\n")

    messages.append({
        "role": "assistant",
        "content": final_assistant_content,
    })

    record = {
        "id": f"{sample_id}_search",
        "task_type": "search_v2",
        "tools": _TOOL_SPECS,
        "messages": messages,
        "audios": audio_assets,
        "meta": {
            "pipeline_version": "v2_search",
            "sample_id": sample_id,
            "archetype": archetype,
            "shard_size": len(shard),
            "shard_start": shard_start,
            "shard_end": shard_end,
            "n_batches": len(all_ordered),
            "candidates_per_batch": candidates_per_batch,
            "gt_in_shard": gt_in_shard,
            "forced_miss_names": force_miss_names or [],
            "final_shortlist": selected_so_far,
            "gt_on_shortlist": [n for n in selected_so_far if n in gt_set],
            "shortlist_output_file": shortlist_path,
            "injected_code_mistake": _code_mistake_info,
        },
    }

    assert_valid_ms_swift_multiturn_record(record)
    return SearchResult(
        record=record,
        shortlist=list(selected_so_far),
        shard_start=shard_start,
        shard_end=shard_end,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Build search-agent SFT v2 (iterative batch listening).")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--index-npy", type=Path, default=Path("outputs/wt_retrieval_baseline/wt_index.npz"))
    ap.add_argument("--index-meta", type=Path, default=Path("outputs/wt_retrieval_baseline/wt_index_meta.json"))
    ap.add_argument("--wavetable-lib", type=Path, default=Path("data/wavetable_lib.json"))
    ap.add_argument("--out-jsonl", required=True, type=Path)
    ap.add_argument("--max-samples", type=int, default=256)
    ap.add_argument("--pool-top-k", type=int, default=48,
        help="[LEGACY] ignored in index-based mode.")
    ap.add_argument("--num-agents", type=int, default=4)
    ap.add_argument("--candidates-per-slice", type=int, default=48,
        help="Number of wavetables per search agent slice (default 48).")
    ap.add_argument("--candidates-per-batch", type=int, default=8)
    ap.add_argument("--probe-dir", type=Path, default=Path("outputs/agent_sft/candidate_probes"))
    ap.add_argument("--shortlist-dir", type=Path,
        default=Path("/tmp/agent_sft/search_shortlists"),
        help="Directory to write per-agent shortlist JSON files. Each agent's "
             "rollout ends with a bash tool_call that writes a shortlist file "
             "to this dir (claw-code-style file-handoff to the main agent at "
             "inference). Pass an empty string to disable.")
    ap.add_argument("--omni-server", default="")
    ap.add_argument("--omni-model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--stage2-server", default="")
    ap.add_argument("--stage2-model", default="")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--code-mistake-rate", type=float, default=0.08,
        help="Probability of injecting a code mistake (real traceback) per search record (default 0.08).")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    stage2_server = args.stage2_server or args.omni_server
    stage2_model = args.stage2_model or args.omni_model

    if args.omni_server:
        _check_server_reachable(args.omni_server, "Omni")

    entries = load_manifest_entries(args.manifest, max_samples=args.max_samples)
    index_rows = load_index_rows(args.index_meta)
    selected_by_name = select_probe_rows_by_name(index_rows)
    wavetable_lib = load_wavetable_lib(args.wavetable_lib)
    shortlist_data = build_clap_shortlist_data(args.index_npy, index_rows)

    candidate_audio: dict[str, Path] = {}
    serial_lock = threading.Lock()

    def _process(entry: dict) -> list[dict]:
        sample_id = str(entry["sample_id"])
        archetype = str(entry.get("archetype", "synth"))
        target_audio_path = Path(entry.get("gt_wav") or entry.get("gt_probe_wav"))

        # Resolve GT wavetable names
        target_preset_path = entry.get("target_preset_path")
        if not target_preset_path:
            path_file = entry.get("path_file")
            if path_file:
                with open(path_file) as f:
                    pd = json.load(f)
                target_preset_path = pd.get("target_preset_path")
        if not target_preset_path:
            return []
        gt_names = list(extract_gt_wavetable_names(Path(target_preset_path)))
        if not gt_names:
            return []
        with open(target_preset_path) as f:
            target_preset = json.load(f)

        # Load target MIDI notes for rendering probes with the actual melody
        _midi_path: str | None = None
        _midi_notes_dicts: list[dict] | None = None
        _midi_notes_pm: list | None = None
        source_midi_path = entry.get("source_midi_path")
        if source_midi_path and Path(source_midi_path).exists():
            try:
                from scripts.build_transcription_agent_sft_v3 import load_notes_from_midi
                _midi_notes_dicts = load_notes_from_midi(source_midi_path)
            except Exception:
                _midi_notes_dicts = None
        if _midi_notes_dicts:
            from maestro.render.dawdreamer import notes_from_dicts
            _midi_notes_pm = notes_from_dicts(_midi_notes_dicts)
            _trans_agent_id = make_agent_id(sample_id, "melody_transcription")
            _trans_dir = Path(f"/tmp/agents/{sample_id}")
            _trans_dir.mkdir(parents=True, exist_ok=True)
            _midi_path = str(_trans_dir / f"{sample_id}_notes.json")
            if not Path(_midi_path).exists():
                with open(_midi_path, "w") as _trf:
                    json.dump({"notes": _midi_notes_dicts, "n_notes": len(_midi_notes_dicts)}, _trf)
                    _trf.write("\n")

        # Build dense name↔idx mapping — matches list_wavetables.py (dedup by name)
        _seen: set[str] = set()
        _unique_names: list[str] = []
        for wt in wavetable_lib:
            if not isinstance(wt, dict) or "name" not in wt:
                continue
            if wt["name"] in _seen:
                continue
            _seen.add(wt["name"])
            _unique_names.append(wt["name"])
        idx_to_name_full: dict[int, str] = {i: n for i, n in enumerate(_unique_names)}
        name_to_idx_full: dict[str, int] = {n: i for i, n in idx_to_name_full.items()}
        total_named = len(_unique_names)

        # CLAP name→embedding for selection decisions
        _name_to_emb = build_name_embedding_map(shortlist_data["embeddings"], index_rows)

        sid_seed = int(hashlib.sha1(sample_id.encode()).hexdigest()[:8], 16)

        # Compute slice offsets: N evenly-spaced slices of size candidates_per_slice.
        slice_size = args.candidates_per_slice
        n_agents = args.num_agents
        stride = max(1, total_named // n_agents)

        def _compute_slices(base: int) -> list[int]:
            starts = []
            for i in range(n_agents):
                start = (base + i * stride) % total_named
                # Clamp so slice ends at total_named (no wraparound within a slice)
                if start + slice_size > total_named:
                    start = max(0, total_named - slice_size)
                starts.append(start)
            return starts

        base_offset = sid_seed % stride
        slice_starts = _compute_slices(base_offset)

        # GT oracle: rotate base_offset until at least one GT is covered
        gt_idxs = [name_to_idx_full[n] for n in gt_names if n in name_to_idx_full]
        if gt_idxs:
            def _gt_covered(starts: list[int]) -> bool:
                return any(
                    s <= gi < (s + slice_size)
                    for s in starts for gi in gt_idxs
                )

            tried = 0
            step = max(1, stride // 4)
            while not _gt_covered(slice_starts) and tried < total_named:
                base_offset = (base_offset + step) % stride
                slice_starts = _compute_slices(base_offset)
                tried += step

        # Pre-render probes for all wavetables across all slices (build-time)
        all_slice_names: list[str] = []
        for s in slice_starts:
            for i in range(s, min(s + slice_size, total_named)):
                all_slice_names.append(idx_to_name_full[i])
        all_slice_names = list(dict.fromkeys(all_slice_names))  # dedupe, preserve order

        ensure_candidate_probes_for_names(
            names=all_slice_names,
            wavetable_lib=wavetable_lib,
            selected_rows=selected_by_name,
            out_dir=args.probe_dir,
            cache=candidate_audio,
                    )

        # Build one record per search agent, one per slice.
        # Agents within a sample are independent — parallelize their Omni calls.
        agent_jobs = []
        for ai, start in enumerate(slice_starts):
            end = min(start + slice_size, total_named)
            if end <= start:
                continue
            agent_jobs.append((ai, start, end))

        def _run_agent(job):
            ai, start, end = job
            result = build_search_record(
                sample_id=sample_id,
                agent_idx=ai + 1,
                archetype=archetype,
                target_audio_path=target_audio_path,
                target_preset=target_preset,
                gt_wavetable_names=gt_names,
                shard_start=start,
                shard_end=end,
                name_to_idx=name_to_idx_full,
                idx_to_name=idx_to_name_full,
                candidate_audio=candidate_audio,
                name_to_emb=_name_to_emb,
                omni_server=args.omni_server,
                omni_model=args.omni_model,
                stage2_server=stage2_server,
                stage2_model=stage2_model,
                candidates_per_batch=args.candidates_per_batch,
                shortlist_dir=args.shortlist_dir,
                midi_path=_midi_path,
                code_mistake_rate=args.code_mistake_rate,
                seed=args.seed,
            )
            rec = result.record
            if rec:
                rec["id"] = f"{sample_id}_agent{ai + 1}_search"
            return (ai, rec)

        records_by_ai: dict[int, dict] = {}
        if len(agent_jobs) > 1:
            with ThreadPoolExecutor(max_workers=len(agent_jobs)) as ap:
                for fut in as_completed([ap.submit(_run_agent, j) for j in agent_jobs]):
                    ai, rec = fut.result()
                    if rec:
                        records_by_ai[ai] = rec
        else:
            for j in agent_jobs:
                ai, rec = _run_agent(j)
                if rec:
                    records_by_ai[ai] = rec

        records = [records_by_ai[ai] for ai in sorted(records_by_ai)]
        return records

    out_path = args.out_jsonl
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []

    def _safe_process(entry):
        try:
            return entry, _process(entry), None
        except Exception as exc:
            import traceback
            return entry, [], (exc, traceback.format_exc())

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_safe_process, e) for e in entries]
            for i, fut in enumerate(as_completed(futs)):
                entry, recs, err = fut.result()
                if err:
                    print(f"WARNING: {entry.get('sample_id', '?')} failed: {err[0]}")
                    print(err[1])
                else:
                    all_records.extend(recs)
                    print(f"[{i + 1}/{len(entries)}] {entry['sample_id']}: {len(recs)} search records", flush=True)
    else:
        for i, entry in enumerate(entries):
            entry, recs, err = _safe_process(entry)
            if err:
                print(f"WARNING: {entry.get('sample_id', '?')} failed: {err[0]}")
                print(err[1])
            else:
                all_records.extend(recs)
                print(f"[{i + 1}/{len(entries)}] {entry['sample_id']}: {len(recs)} search records", flush=True)

    with open(out_path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_records)} records to {out_path}", flush=True)


if __name__ == "__main__":
    main()
