#!/usr/bin/env python3
"""Build judge-agent SFT v3: audition-based tuple selection from a combined search pool.

The judge agent is invoked by the main agent AFTER the search agents have each
returned a shortlist. Its job is to look at the combined pool in one auditory
view (each candidate audio labelled with its wavetable name) and select the N
candidates (1 to 3) that together best capture the target — the judge decides
how many oscillators to use based on what it hears.

Why the judge exists:
- Each search agent only sees its own slice (~48 wavetables). When GTs are
  scattered across slices, each search agent finds 1 GT + some false positives.
  None has the global view to pick the correct combination.
- The judge receives the union of all search shortlists (~8-15 candidates total),
  listens to all of them alongside the target in a single Omni call, writes
  per-candidate reasoning, then picks the final tuple and writes it to a JSON
  file that the main agent consumes.

Build-time simulation:
- Pool construction: simulate search shortlists the same way the main agent
  does (reuses _simulate_shortlist logic via pool from search agents).
- Selected tuple: GT-if-in-pool + CLAP-best-proxy per osc slot. Same oracle
  picking used by the main agent's synthetic judge output file.
- Per-candidate reasoning: Omni Stage 1 listens to target + all pool audios,
  writes a sentence per candidate; Stage 2 formats into final selection text.

Output:
- JSONL, one record per sample, task_type="judge". Each record represents the
  judge's full conversation: receives pool + target → renders probes → listens
  to all at once → per-candidate reasoning → selects tuple → writes output file.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.agent_sft_common import (  # type: ignore
    DawFarmRolloutCtx,
    assert_valid_ms_swift_multiturn_record,
    build_clap_shortlist_data,
    build_name_embedding_map,
    build_render_probes_snippet,
    ensure_candidate_probes_for_names,
    extract_gt_wavetable_names,
    is_clap_selected,
    load_index_rows,
    load_manifest_entries,
    load_wavetable_lib,
    select_probe_rows_by_name,
    _bash_tool_response,
    _read_tool_response_audio,
    _tool_call as _tool_call_common,
    _wrap_as_bash,
)
from scripts.build_main_agent_sft_v2 import (  # type: ignore
    _check_server_reachable,
    _llm_post,
)


def _b64(path: str | Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _tool_call(name: str, arguments: dict) -> dict:
    return _tool_call_common(name, arguments)


# Tools available to the judge agent at inference (matches main-agent dispatch prompt)
_JUDGE_TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Run a shell command (used to render probes and write the selection file).",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
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
]


def _simulate_search_pool(
    sample_id: str,
    gt_names: list[str],
    name_to_idx_full: dict[str, int],
    idx_to_name_full: dict[int, str],
    name_to_emb: dict[str, np.ndarray],
    total_named: int,
    n_agents: int,
    slice_size: int,
    candidates_per_agent: int,
    force_miss: bool,
    seed: int,
) -> list[str]:
    """Simulate the pool of candidates the main agent would receive from search agents.

    Same slice-rotation + GT-coverage logic the main agent uses, then each slice
    contributes 2-4 candidates via CLAP-similarity selection. Returns the pooled
    (deduplicated) candidate names across all slices.
    """
    sid_seed = int(hashlib.sha1(sample_id.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed + sid_seed)
    stride = max(1, total_named // n_agents)
    base_offset = rng.randrange(stride)

    def _compute_slices(base: int) -> list[int]:
        starts = []
        for i in range(n_agents):
            start = (base + i * stride) % total_named
            if start + slice_size > total_named:
                start = max(0, total_named - slice_size)
            starts.append(start)
        return starts

    slice_starts = _compute_slices(base_offset)
    gt_idxs = [name_to_idx_full[n] for n in gt_names if n in name_to_idx_full]
    if gt_idxs and not force_miss:
        step = max(1, stride // 4)
        tried = 0
        while not any(s <= gi < (s + slice_size) for s in slice_starts for gi in gt_idxs) and tried < total_named:
            base_offset = (base_offset + step) % stride
            slice_starts = _compute_slices(base_offset)
            tried += step

    pool: list[str] = []
    for start in slice_starts:
        end = min(start + slice_size, total_named)
        slice_names = [idx_to_name_full[i] for i in range(start, end) if i in idx_to_name_full]
        # Pick CLAP-selected names (matching search agent's selection logic)
        picks: list[str] = []
        for n in slice_names:
            if is_clap_selected(n, gt_names, name_to_emb, threshold=0.97):
                picks.append(n)
        # Pad with top non-GT by CLAP to GT so pool isn't tiny
        if len(picks) < candidates_per_agent:
            non_picks = [n for n in slice_names if n not in picks]
            # Rank by best-CLAP-to-any-GT
            def _rank(n: str) -> float:
                if n not in name_to_emb:
                    return 0.0
                best = 0.0
                for gt in gt_names:
                    if gt in name_to_emb:
                        sim = float(name_to_emb[n] @ name_to_emb[gt])
                        if sim > best:
                            best = sim
                return best
            non_picks.sort(key=_rank, reverse=True)
            for n in non_picks[: candidates_per_agent - len(picks)]:
                picks.append(n)
        picks = picks[:candidates_per_agent]
        for n in picks:
            if n not in pool:
                pool.append(n)

    return pool


def _omni_single_audio_call(
    audio_path: Path,
    prompt_text: str,
    omni_server: str,
    omni_model: str,
    max_tokens: int = 160,
    temperature: float = 0.4,
    timeout: float = 60.0,
    fallback: str = "",
) -> str:
    """Send ONE audio + ONE text prompt, return the completion. Used in place
    of multi-audio batch calls to avoid position-confusion hallucinations."""
    content = [
        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{_b64(audio_path)}"}},
        {"type": "text", "text": prompt_text},
    ]
    try:
        r = _llm_post(
            f"{omni_server}/v1/chat/completions",
            {"model": omni_model,
             "messages": [{"role": "user", "content": content}],
             "max_tokens": max_tokens, "temperature": temperature},
            timeout=timeout,
        )
        return r["choices"][0]["message"]["content"].strip()
    except Exception:
        return fallback


def omni_judge_audition(
    target_wav: Path,
    pool: list[str],
    candidate_audio: dict[str, Path],
    n_osc_slots: int,
    omni_server: str,
    omni_model: str,
    timeout: float = 300.0,
) -> str:
    """Single-audio policy: describe target + each pool candidate in isolation,
    then synthesize per-candidate assessments + a recommendation via a text-only
    call. Avoids the multi-audio position-confusion that the old batch-audition
    call was prone to."""
    valid_names = [n for n in pool if n in candidate_audio]
    if not valid_names:
        return f"RECOMMENDATION: [] : (no candidates)"

    target_desc = _omni_single_audio_call(
        audio_path=target_wav,
        prompt_text=(
            "This is a target sound we're trying to recreate with a Vital synth. "
            "Describe its TIMBRE in one or two sentences: harmonic content, brightness, "
            "texture, attack shape, and any distinctive effects. Focus on timbral "
            "qualities only — do not mention note pattern."
        ),
        omni_server=omni_server,
        omni_model=omni_model,
        max_tokens=180,
        fallback="A target sound.",
    )

    cand_descs: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(8, len(valid_names))) as ex:
        def _describe(name: str) -> tuple[str, str]:
            desc = _omni_single_audio_call(
                audio_path=candidate_audio[name],
                prompt_text=(
                    "This is a wavetable rendered through a default Vital preset "
                    "(probe: 4 triads over ~10s; note pattern is fixed across all probes). "
                    "In ONE sentence, describe only this wavetable's timbral character: "
                    "harmonic content, brightness, texture, attack shape. Ignore note pattern."
                ),
                omni_server=omni_server,
                omni_model=omni_model,
                max_tokens=120,
                fallback="Candidate wavetable.",
            )
            return name, desc
        futures = [ex.submit(_describe, n) for n in valid_names]
        for fut in as_completed(futures):
            name, desc = fut.result()
            cand_descs[name] = desc

    cand_block = "\n".join(
        f"  '{n}': {cand_descs.get(n, 'Candidate wavetable.')}" for n in valid_names
    )
    synth_prompt = (
        f"You are auditioning wavetable candidates for a Vital recreation.\n\n"
        f"Target timbre (described from an isolated listen):\n  {target_desc}\n\n"
        f"Candidates (each described in isolation):\n{cand_block}\n\n"
        f"Write ONE short sentence per candidate describing how its raw timbre relates to the "
        f"target and whether it could serve as a building block once filters, envelopes, and "
        f"effects are applied. Then on a final line, recommend which candidates (1 to 3) "
        f"together would best capture the target.\n\n"
        f"Format per line: '<name>': <one-sentence assessment>\n"
        f"Final line: 'RECOMMENDATION: [name1, name2, ...]: <one sentence on why this combination works>'\n"
        f"Use the EXACT names above. No snake_case. Do not mention note pattern."
    )
    try:
        r = _llm_post(
            f"{omni_server}/v1/chat/completions",
            {"model": omni_model,
             "messages": [{"role": "user", "content": synth_prompt}],
             "max_tokens": 900, "temperature": 0.4},
            timeout=timeout,
        )
        return r["choices"][0]["message"]["content"].strip()
    except Exception:
        lines = [f"'{n}': {cand_descs.get(n, 'Candidate wavetable.')}" for n in valid_names]
        return "\n".join(lines) + "\nRECOMMENDATION: [] : (fallback — synth call failed)"


@dataclass
class JudgeResult:
    """Return type for build_judge_record(): the SFT record + the verdict/tuple
    that the main agent would see when it reads the judge's output file."""
    record: dict | None
    verdict: str                  # "good", "partial_match", or "no_match"
    tuple: list[str] | None      # selected wavetable names (None if no_match)
    n_osc_slots: int = 0
    reasoning: str = ""
    missing_character: str = ""
    locked_slots: dict[int, str] | None = None   # osc_idx → name for slots confirmed good (partial_match)
    unfilled_oscs: list[int] | None = None        # osc indices still needing candidates (partial_match)


# Verdict constants — judge emits one of these in its output JSON.
_VERDICT_GOOD = "good"
_VERDICT_PARTIAL = "partial_match"
_VERDICT_NO_MATCH = "no_match"


def _derive_missing_character(
    target_wav: Path | str,
    omni_server: str,
    omni_model: str,
    timeout: float = 30.0,
) -> str:
    """Single-audio Omni call: in 3-6 words, name the dominant timbral quality
    of the target. Used when the judge's pool lacks GT — produces a perceptual
    hint for the re-search recommendation. Falls back to a static phrase."""
    if not omni_server:
        return "timbral character of the target"
    prompt_text = (
        "In 3-6 words, name the dominant timbral quality of this synth sound — "
        "the single phrase a sound designer would use to describe its core "
        "character. Examples: 'metallic FM buzz', 'breathy vocal pad', "
        "'punchy plucked bass', 'glassy resonant bell'. Respond with the "
        "phrase only — no preamble, no quotes."
    )
    desc = _omni_single_audio_call(
        audio_path=Path(str(target_wav)),
        prompt_text=prompt_text,
        omni_server=omni_server,
        omni_model=omni_model,
        max_tokens=40,
        temperature=0.4,
        timeout=timeout,
        fallback="timbral character of the target",
    )
    # Strip surrounding quotes/punctuation if present.
    desc = desc.strip().strip(".'\" ")
    if not desc or len(desc.split()) < 2:
        return "timbral character of the target"
    return desc[:120]


def stage2_format_judge_response(
    omni_observations: str,
    pool: list[str],
    selected_tuple: list[str],
    n_osc_slots: int,
    gt_names_in_pool: list[str],
    verdict: str,
    missing_character: str,
    stage2_server: str,
    stage2_model: str,
    timeout: float = 180.0,
) -> str:
    """Stage 2 text model formats Omni's observations into the final judge response.

    Branches on verdict:
      - "good": existing template. Final line: SELECTED: [names]: <reason>.
      - "partial_match": new template. Lock the filled slots, recommend re-search
        for the unfilled ones.
        Final line: PARTIAL_MATCH: [locked names]: <missing_character>: <reason>.
      - "no_match": template. Stage-2 instructed NOT to pick a tuple.
        Final line: NO_MATCH: <missing_character>: <re-search recommendation>.
    """
    if verdict == _VERDICT_NO_MATCH:
        return _stage2_no_match_response(
            omni_observations=omni_observations,
            pool=pool,
            missing_character=missing_character,
            stage2_server=stage2_server,
            stage2_model=stage2_model,
            timeout=timeout,
        )
    if verdict == _VERDICT_PARTIAL:
        return _stage2_partial_match_response(
            omni_observations=omni_observations,
            pool=pool,
            selected_tuple=selected_tuple,
            n_osc_slots=n_osc_slots,
            gt_names_in_pool=gt_names_in_pool,
            missing_character=missing_character,
            stage2_server=stage2_server,
            stage2_model=stage2_model,
            timeout=timeout,
        )
    # verdict == "good"
    gt_hint = (
        f"Ground truth (hidden from the judge at inference): the target actually uses "
        f"{', '.join(repr(g) for g in gt_names_in_pool)} from the pool. "
        f"Your selection below MUST match this — if you're picking a subset, pick those."
    ) if gt_names_in_pool else ""
    selected_list_str = ", ".join(repr(n) for n in selected_tuple)
    prompt = (
        f"You are a music production AI formatting the judge agent's final response.\n\n"
        f"Omni listened to the target + all pool candidates and wrote:\n"
        f"---\n{omni_observations}\n---\n\n"
        f"Pool candidates: {json.dumps(pool)}\n"
        f"The ORACLE has pre-determined that the correct selection is: [{selected_list_str}]\n"
        f"{gt_hint}\n\n"
        f"Write exactly {len(pool) + 1} lines:\n"
        f"  - For each candidate, one line: '<name>': <one-sentence assessment>. "
        f"For selected candidates, explain why they fit; for others, explain why they "
        f"don't fit as well.\n"
        f"  - Final line: 'SELECTED: [{selected_list_str}]: <one-sentence explanation "
        f"of why this combination specifically captures the target's character>'\n\n"
        f"Use the EXACT names. Natural language. No snake_case, no numeric values."
    )
    try:
        r = _llm_post(
            f"{stage2_server}/v1/chat/completions",
            {"model": stage2_model,
             "messages": [{"role": "user", "content": prompt}],
             "max_tokens": 800, "temperature": 0.6},
            timeout=timeout,
        )
        out = r["choices"][0]["message"]["content"].strip()
    except Exception:
        per_cand = "\n".join(f'"{n}": Candidate evaluated.' for n in pool)
        return f"{per_cand}\nSELECTED: [{selected_list_str}]: Best combination from pool for the target."
    # Defensive: if Stage-2 emitted a NO_MATCH line for a good verdict, strip it.
    if "NO_MATCH:" in out:
        out_lines = [ln for ln in out.splitlines() if not ln.strip().upper().startswith("NO_MATCH:")]
        out = "\n".join(out_lines).strip()
        if "SELECTED:" not in out:
            out = out + f"\nSELECTED: [{selected_list_str}]: Best combination from pool for the target."
    return out


def _stage2_no_match_response(
    omni_observations: str,
    pool: list[str],
    missing_character: str,
    stage2_server: str,
    stage2_model: str,
    timeout: float = 180.0,
) -> str:
    """Stage-2 prompt for the no_match verdict: explicitly recommend re-search,
    no tuple pick. Output ends with `NO_MATCH: <missing_character>: <reason>`."""
    prompt = (
        f"You are a music production AI formatting the judge agent's final response.\n\n"
        f"Omni listened to the target + all pool candidates and wrote:\n"
        f"---\n{omni_observations}\n---\n\n"
        f"Pool candidates: {json.dumps(pool)}\n"
        f"The ORACLE has determined the pool DOES NOT contain any wavetable that "
        f"captures the target's defining character — specifically the "
        f"'{missing_character}' quality.\n\n"
        f"Write exactly {len(pool) + 1} lines:\n"
        f"  - For each candidate, one line: '<name>': <one-sentence assessment of "
        f"why it's not a good match for the target despite any partial overlap>.\n"
        f"  - Final line: 'NO_MATCH: {missing_character}: <one-sentence "
        f"recommendation that the main agent re-dispatch search across "
        f"unexplored library regions to find wavetables with this character>'.\n\n"
        f"Do NOT pick any selection. Do NOT emit a SELECTED line. The pool is "
        f"insufficient — the correct response is to recommend re-search.\n"
        f"Use the EXACT names. Natural language. No snake_case, no numeric values."
    )
    try:
        r = _llm_post(
            f"{stage2_server}/v1/chat/completions",
            {"model": stage2_model,
             "messages": [{"role": "user", "content": prompt}],
             "max_tokens": 800, "temperature": 0.6},
            timeout=timeout,
        )
        out = r["choices"][0]["message"]["content"].strip()
    except Exception:
        per_cand = "\n".join(f'"{n}": Does not carry the {missing_character} of the target.' for n in pool)
        return (
            f"{per_cand}\nNO_MATCH: {missing_character}: Pool lacks any wavetable "
            f"with this character — recommending re-search across unexplored regions."
        )
    # Defensive: if Stage-2 leaked a SELECTED line, strip it and substitute NO_MATCH.
    if "SELECTED:" in out and "NO_MATCH:" not in out:
        out_lines = [ln for ln in out.splitlines() if not ln.strip().upper().startswith("SELECTED:")]
        out = "\n".join(out_lines).rstrip()
        out += (
            f"\nNO_MATCH: {missing_character}: Pool lacks any wavetable with "
            f"this character — recommending re-search across unexplored regions."
        )
    elif "NO_MATCH:" not in out:
        # Stage-2 produced no terminator — append one.
        out += (
            f"\nNO_MATCH: {missing_character}: Pool lacks any wavetable with "
            f"this character — recommending re-search across unexplored regions."
        )
    return out


def _stage2_partial_match_response(
    omni_observations: str,
    pool: list[str],
    selected_tuple: list[str],
    n_osc_slots: int,
    gt_names_in_pool: list[str],
    missing_character: str,
    stage2_server: str,
    stage2_model: str,
    timeout: float = 180.0,
) -> str:
    """Stage-2 prompt for partial_match: lock the filled slots, recommend
    re-search for the unfilled ones.
    Output ends with `PARTIAL_MATCH: [locked names]: <missing_character>: <reason>`."""
    locked_list_str = ", ".join(repr(n) for n in selected_tuple)
    gt_hint = (
        f"Ground truth (hidden from the judge at inference): the pool contains "
        f"{', '.join(repr(g) for g in gt_names_in_pool)} which are correct for some "
        f"oscillator slots. Your locked selection MUST include these."
    ) if gt_names_in_pool else ""
    prompt = (
        f"You are a music production AI formatting the judge agent's final response.\n\n"
        f"Omni listened to the target + all pool candidates and wrote:\n"
        f"---\n{omni_observations}\n---\n\n"
        f"Pool candidates: {json.dumps(pool)}\n"
        f"The ORACLE has determined that the pool contains SOME but NOT ALL needed "
        f"wavetables.\n"
        f"Locked (confirmed good): [{locked_list_str}]\n"
        f"Missing character for unfilled slot(s): '{missing_character}'\n"
        f"{gt_hint}\n\n"
        f"Write exactly {len(pool) + 1} lines:\n"
        f"  - For each candidate, one line: '<name>': <one-sentence assessment>. "
        f"For locked candidates, explain why they fit their slot well. "
        f"For others, explain why they don't fill the remaining slot(s).\n"
        f"  - Final line: 'PARTIAL_MATCH: [{locked_list_str}]: {missing_character}: "
        f"<one-sentence explanation that some slots are filled but the pool lacks "
        f"a wavetable with this character for the remaining slot(s) — "
        f"recommend re-search>'\n\n"
        f"Do NOT emit a SELECTED line (not all slots are filled). "
        f"Do NOT emit a NO_MATCH line (some slots ARE filled). "
        f"Use the EXACT names. Natural language."
    )
    try:
        r = _llm_post(
            f"{stage2_server}/v1/chat/completions",
            {"model": stage2_model,
             "messages": [{"role": "user", "content": prompt}],
             "max_tokens": 800, "temperature": 0.6},
            timeout=timeout,
        )
        out = r["choices"][0]["message"]["content"].strip()
    except Exception:
        per_cand = "\n".join(
            f'"{n}": {"Locked — good match for its slot." if n in selected_tuple else "Does not fill the remaining slot."}'
            for n in pool
        )
        return (
            f"{per_cand}\nPARTIAL_MATCH: [{locked_list_str}]: {missing_character}: "
            f"Some slots filled but pool lacks a wavetable with this character "
            f"for the remaining slot(s). Recommending re-search."
        )
    # Defensive: strip any SELECTED or NO_MATCH lines Stage-2 might have leaked.
    if "SELECTED:" in out or "NO_MATCH:" in out:
        out_lines = [
            ln for ln in out.splitlines()
            if not ln.strip().upper().startswith("SELECTED:")
            and not ln.strip().upper().startswith("NO_MATCH:")
        ]
        out = "\n".join(out_lines).rstrip()
    if "PARTIAL_MATCH:" not in out:
        out += (
            f"\nPARTIAL_MATCH: [{locked_list_str}]: {missing_character}: "
            f"Some slots filled but pool lacks a wavetable with this character "
            f"for the remaining slot(s). Recommending re-search."
        )
    return out


def _extract_final_reasoning(stage2_text: str) -> str:
    """Pull the SELECTED:, PARTIAL_MATCH:, or NO_MATCH: line's justification
    for use in the output JSON file. Format-tolerant."""
    for line in reversed(stage2_text.splitlines()):
        line = line.strip()
        upper = line.upper()
        if upper.startswith("SELECTED:") or upper.startswith("NO_MATCH:") or upper.startswith("PARTIAL_MATCH:"):
            parts = line.split(":", 2)
            if len(parts) >= 3:
                return parts[2].strip()
            return line
    # Fallback: use the last non-empty line
    for line in reversed(stage2_text.splitlines()):
        if line.strip():
            return line.strip()[:200]
    return "Best combination selected from pool."


def build_judge_record(
    *,
    sample_id: str,
    target_audio_path: Path,
    target_preset: dict,
    gt_wavetable_names: list[str],
    pool: list[str],
    candidate_audio: dict[str, Path],
    name_to_emb: dict[str, np.ndarray],
    active_oscs: list[int],
    omni_server: str,
    omni_model: str,
    stage2_server: str,
    stage2_model: str,
    judge_output_dir: Path,
    probe_audio_dir: Path | None = None,
    midi_path: str | None = None,
    dawfarm: "DawFarmRolloutCtx | None" = None,
) -> JudgeResult:
    """Build one SFT judge record. Returns empty JudgeResult if the pool is empty."""
    if not pool:
        return JudgeResult(record=None, verdict="no_match", tuple=None)

    messages: list[dict] = []
    audio_assets: list[str] = [str(target_audio_path)]

    n_osc_slots = len(active_oscs)
    if probe_audio_dir is not None:
        probe_out_dir = str(probe_audio_dir / sample_id)
    else:
        probe_out_dir = f"/tmp/judge_probes/{sample_id}"
    display_probe_dir = (
        dawfarm.cw("", "judge_probes") if dawfarm is not None else probe_out_dir
    )

    pool_str = ", ".join(f'"{n}"' for n in pool)
    from scripts.agent_sft_common import make_agent_id  # type: ignore
    judge_agent_id = make_agent_id(sample_id, "wavetable_judge")
    judge_output_file = judge_output_dir / f"{judge_agent_id}.md"
    judge_output_file.parent.mkdir(parents=True, exist_ok=True)

    # User prompt — mirrors what the main agent's Agent tool_call sends.
    # No file path — the framework captures the agent's result automatically.
    messages.append({
        "role": "user",
        "content": (
            f"<audio>\n"
            f"Pool candidates from search agents: [{pool_str}].\n"
            f"The target may use up to 3 active oscillators. Render probes for each "
            f"candidate, listen alongside the target, and select the candidates (1 to 3) "
            f"that together best capture the target's character. "
            f"Return your selection as JSON with keys: "
            f"tuple (list of chosen names), n_osc_slots (how many you chose), reasoning."
        ),
    })

    # Render probes for all pool candidates
    render_cmd = _wrap_as_bash(build_render_probes_snippet(
        names=list(pool), out_dir=display_probe_dir,
        midi_path=midi_path,
    ))
    messages.append({
        "role": "assistant",
        "content": f"Rendering probes for all {len(pool)} pool candidates in one batch.",
    })
    messages.append(_tool_call("Bash", {"command": render_cmd}))

    import re as _re
    def _slugify_j(s: str) -> str:
        return (_re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_") or "unnamed")[:80]

    rendered_entries = []
    audio_read_paths: list[str] = []
    audio_host_paths: list[str] = []
    for idx, name in enumerate(pool):
        fname = f"wt_{idx:04d}_{_slugify_j(name)}.wav"
        display_path = f"{display_probe_dir}/{fname}"
        host_path = f"{probe_out_dir}/{fname}"
        audio_assets.append(host_path)
        audio_read_paths.append(display_path)
        audio_host_paths.append(host_path)
        rendered_entries.append({"name": name, "out": display_path})
    if dawfarm is not None:
        _rres = dawfarm.real_exec(render_cmd, "judge probe render",
                                  timeout=max(dawfarm.exec_timeout, 600.0))
        try:
            _real_rendered = {e["name"] for e in json.loads(_rres.stdout)["rendered"]}
        except Exception as exc:
            raise RuntimeError(f"unparseable judge probe stdout: {exc}: {_rres.stdout[:300]}")
        if _real_rendered != set(pool):
            raise RuntimeError(
                f"judge probe render mismatch: missing={sorted(set(pool) - _real_rendered)[:5]}")
        dawfarm.fetch_dir(display_probe_dir, probe_out_dir)
        _missing = [p for p in audio_host_paths if not Path(p).exists()]
        if _missing:
            raise RuntimeError(f"judge probe fetch incomplete: missing {len(_missing)}")
        messages.append(_bash_tool_response(_rres.stdout))
    else:
        _render_stdout = json.dumps({"status": "ok", "rendered": rendered_entries}) + "\n"
        messages.append(_bash_tool_response(_render_stdout))

    # Read each rendered probe — sequential read calls
    if audio_read_paths:
        messages.append({"role": "assistant", "content": "Listening to all pool candidates."})
        for rp in audio_read_paths:
            messages.append(_tool_call("Read", {"file_path": rp}))
            messages.append(_read_tool_response_audio())

    # Verdict computation: three-way —
    #   "good":          every active-osc GT is in pool → full tuple, done
    #   "partial_match": some (but not all) active-osc GTs are in pool →
    #                    lock the filled slots, re-search for the rest
    #   "no_match":      zero active-osc GTs in pool → reject all, full re-search
    wts = target_preset.get("settings", {}).get("wavetables", [])
    needed_gt_names_by_osc: dict[int, str] = {}
    for oi in active_oscs:
        n = wts[oi].get("name", "") if oi < len(wts) else ""
        if n:
            needed_gt_names_by_osc[oi] = n
    needed_gt_names = list(needed_gt_names_by_osc.values())

    filled_oscs = [oi for oi, n in needed_gt_names_by_osc.items() if n in pool]
    unfilled_oscs = [oi for oi, n in needed_gt_names_by_osc.items() if n not in pool]

    if not needed_gt_names or len(filled_oscs) == len(needed_gt_names_by_osc):
        verdict = _VERDICT_GOOD
    elif len(filled_oscs) > 0:
        verdict = _VERDICT_PARTIAL
    else:
        verdict = _VERDICT_NO_MATCH

    # Build-time oracle: GT-in-pool + CLAP-best proxy per osc slot.
    cur_tuple: list[str | None] = [None, None, None]
    used_in_tuple: set[str] = set()
    for osc_idx in active_oscs:
        wt_name = wts[osc_idx].get("name", "") if osc_idx < len(wts) else ""
        if wt_name and wt_name in pool:
            cur_tuple[osc_idx] = wt_name
            used_in_tuple.add(wt_name)
            continue
        gt_emb = name_to_emb.get(wt_name)
        if gt_emb is not None:
            candidates = [n for n in pool if n not in used_in_tuple and n in name_to_emb]
            if candidates:
                best = max(candidates, key=lambda n: float(name_to_emb[n] @ gt_emb))
                cur_tuple[osc_idx] = best
                used_in_tuple.add(best)
                continue
        for n in pool:
            if n not in used_in_tuple:
                cur_tuple[osc_idx] = n
                used_in_tuple.add(n)
                break

    selected_tuple: list[str] = [cur_tuple[oi] for oi in active_oscs if cur_tuple[oi]]
    # good → full tuple; partial → tuple of filled slots only; no_match → None
    locked_slots: dict[int, str] = {}
    if verdict == _VERDICT_GOOD:
        selected_tuple_for_output = selected_tuple
    elif verdict == _VERDICT_PARTIAL:
        locked_slots = {oi: cur_tuple[oi] for oi in filled_oscs if cur_tuple[oi]}
        selected_tuple_for_output = list(locked_slots.values())
    else:
        selected_tuple_for_output = None
    gts_in_pool = [g for g in gt_wavetable_names if g in pool]

    # Missing-character hint (meaningful for partial_match and no_match).
    missing_character = ""
    if verdict in (_VERDICT_PARTIAL, _VERDICT_NO_MATCH):
        missing_character = _derive_missing_character(
            target_wav=target_audio_path,
            omni_server=omni_server,
            omni_model=omni_model,
        )

    # Omni Stage 1: audition target + all pool candidates
    if omni_server:
        omni_response = omni_judge_audition(
            target_wav=target_audio_path,
            pool=pool,
            candidate_audio=candidate_audio,
            n_osc_slots=n_osc_slots,
            omni_server=omni_server,
            omni_model=omni_model,
        )
    else:
        if verdict == _VERDICT_GOOD:
            omni_response = "\n".join(f'"{n}": Candidate evaluated.' for n in pool) + \
                f"\nRECOMMENDATION: [{', '.join(repr(n) for n in selected_tuple)}] : Best combination from pool."
        elif verdict == _VERDICT_PARTIAL:
            locked_names = list(locked_slots.values())
            omni_response = "\n".join(
                f'"{n}": {"Good match for its slot." if n in locked_names else "Does not fill the remaining slot."}'
                for n in pool
            )
        else:
            omni_response = "\n".join(
                f'"{n}": Does not carry the {missing_character} of the target.' for n in pool
            )

    # Stage 2: format with oracle-selected verdict + tuple
    stage2_response = stage2_format_judge_response(
        omni_observations=omni_response,
        pool=pool,
        selected_tuple=selected_tuple,
        n_osc_slots=n_osc_slots,
        gt_names_in_pool=gts_in_pool,
        verdict=verdict,
        missing_character=missing_character,
        stage2_server=stage2_server,
        stage2_model=stage2_model,
    )
    final_reasoning = _extract_final_reasoning(stage2_response)

    # Build closing text based on verdict
    if verdict == _VERDICT_GOOD:
        tuple_str = ", ".join(repr(n) for n in selected_tuple)
        closing = (
            f"Final tuple: [{tuple_str}]. The main agent can now apply the "
            f"chosen wavetables."
        )
    elif verdict == _VERDICT_PARTIAL:
        locked_str = ", ".join(repr(n) for n in locked_slots.values())
        n_unfilled = len(unfilled_oscs)
        closing = (
            f"PARTIAL_MATCH — [{locked_str}] confirmed but the pool still lacks "
            f"a wavetable with the {missing_character} of the target for the "
            f"remaining slot{'s' if n_unfilled > 1 else ''}. Recommending "
            f"re-dispatch search."
        )
    else:
        closing = (
            f"NO_MATCH — none of these {len(pool)} candidates carries "
            f"the {missing_character} of the target. Recommending re-dispatch "
            f"search across unexplored library regions."
        )

    # Single assistant turn: deliberation + closing (avoids adjacent assistant messages)
    final_assistant_content = (
        f"Listening to the target alongside all {len(pool)} pool candidates at once.\n\n"
        f"{stage2_response}\n\n"
        f"{closing}"
    )

    # Persist the final assistant message to disk (build-time) so the main
    # agent's `cat` works.  In claw-code the framework writes the sub-agent's
    # last response to the outputFile — plain text, not JSON.
    with open(judge_output_file, "w") as f:
        f.write(final_assistant_content)
        f.write("\n")

    messages.append({
        "role": "assistant",
        "content": final_assistant_content,
    })

    # Judge correctness: did the selection match the GTs present in the pool
    # (for the subset of osc slots where a GT was available)?
    n_correct = sum(1 for t in selected_tuple if t in gt_wavetable_names)
    judge_correct = n_correct == min(len(gt_wavetable_names), n_osc_slots, len(gts_in_pool))

    record = {
        "id": f"{sample_id}_judge",
        "task_type": "judge",
        "tools": _JUDGE_TOOL_SPECS,
        "messages": messages,
        "audios": audio_assets,
        "meta": {
            "pipeline_version": "v3_judge",
            "sample_id": sample_id,
            "pool": pool,
            "pool_size": len(pool),
            "n_osc_slots": n_osc_slots,
            "active_oscs": active_oscs,
            "gt_wavetable_names": gt_wavetable_names,
            "gts_in_pool": gts_in_pool,
            "verdict": verdict,
            "missing_character": missing_character,
            # selected_tuple stays populated even on partial/no_match for analysis;
            # the OUTPUT JSON's tuple field reflects only confirmed slots.
            "selected_tuple": selected_tuple,
            "locked_slots": locked_slots if verdict == _VERDICT_PARTIAL else {},
            "unfilled_oscs": unfilled_oscs if verdict == _VERDICT_PARTIAL else [],
            "judge_correct": judge_correct,
            "output_file": str(judge_output_file),
        },
    }

    assert_valid_ms_swift_multiturn_record(record)
    return JudgeResult(
        record=record,
        verdict=verdict,
        tuple=selected_tuple_for_output,
        n_osc_slots=n_osc_slots,
        reasoning=final_reasoning,
        missing_character=missing_character,
        locked_slots=locked_slots if verdict == _VERDICT_PARTIAL else None,
        unfilled_oscs=unfilled_oscs if verdict == _VERDICT_PARTIAL else None,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build judge-agent SFT v3. Generates one training record per sample "
            "where the judge audions the combined search pool, picks N best "
            "candidates, and writes the selection to a JSON output file."
        )
    )
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--index-npy", type=Path, default=Path("outputs/wt_retrieval_baseline/wt_index.npz"))
    ap.add_argument("--index-meta", type=Path, default=Path("outputs/wt_retrieval_baseline/wt_index_meta.json"))
    ap.add_argument("--wavetable-lib", type=Path, default=Path("data/wavetable_lib.json"))
    ap.add_argument("--out-jsonl", required=True, type=Path)
    ap.add_argument("--judge-output-dir", type=Path, default=Path("/tmp/judge_outputs"),
                    help="Where the judge's output JSON files land (mirrors main agent's /tmp/agents dir).")
    ap.add_argument("--max-samples", type=int, default=256)
    ap.add_argument("--num-agents", type=int, default=4,
                    help="Number of search agents (for pool-simulation slice count).")
    ap.add_argument("--candidates-per-slice", type=int, default=48,
                    help="Wavetables per search-agent slice (for pool simulation).")
    ap.add_argument("--candidates-per-agent", type=int, default=3,
                    help="Picks per search agent (for pool simulation).")
    ap.add_argument("--force-research-rate", type=float, default=0.30,
                    help="Fraction of samples where GT-oracle rotation is skipped, "
                         "simulating the re-search branch (judge sees fewer GTs in pool).")
    ap.add_argument("--probe-dir", type=Path, default=Path("outputs/agent_sft/candidate_probes"))
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--omni-server", default="")
    ap.add_argument("--omni-model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--stage2-server", default="")
    ap.add_argument("--stage2-model", default="")
    ap.add_argument("--workers", type=int, default=1)
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

    # Dense name↔idx mapping (matches list_wavetables.py — dedup by name)
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

    name_to_emb = build_name_embedding_map(shortlist_data["embeddings"], index_rows)

    candidate_audio: dict[str, Path] = {}
    serial_lock = threading.Lock()

    def _process(entry: dict) -> dict | None:
        sample_id = str(entry["sample_id"])
        target_audio_path = Path(entry.get("gt_wav") or entry.get("gt_probe_wav"))
        target_preset_path = entry.get("target_preset_path")
        if not target_preset_path:
            path_file = entry.get("path_file")
            if path_file:
                with open(path_file) as f:
                    pd = json.load(f)
                target_preset_path = pd.get("target_preset_path")
        if not target_preset_path:
            return None
        gt_names = list(extract_gt_wavetable_names(Path(target_preset_path)))
        if not gt_names:
            return None
        with open(target_preset_path) as f:
            target_preset = json.load(f)

        # Active oscillators (osc_1_on, osc_2_on, osc_3_on > 0.5)
        s = target_preset.get("settings", {}) or {}
        active_oscs = [i - 1 for i in (1, 2, 3) if float(s.get(f"osc_{i}_on", 0) or 0) > 0.5]
        if not active_oscs:
            return None

        sid_seed = int(hashlib.sha1(sample_id.encode()).hexdigest()[:8], 16)
        sample_rng = random.Random(args.seed + sid_seed)
        force_miss = sample_rng.random() < args.force_research_rate

        pool = _simulate_search_pool(
            sample_id=sample_id,
            gt_names=gt_names,
            name_to_idx_full=name_to_idx_full,
            idx_to_name_full=idx_to_name_full,
            name_to_emb=name_to_emb,
            total_named=total_named,
            n_agents=args.num_agents,
            slice_size=args.candidates_per_slice,
            candidates_per_agent=args.candidates_per_agent,
            force_miss=force_miss,
            seed=args.seed,
        )

        with serial_lock:
            ensure_candidate_probes_for_names(
                names=pool,
                wavetable_lib=wavetable_lib,
                selected_rows=selected_by_name,
                out_dir=args.probe_dir,
                cache=candidate_audio,
            )

        result = build_judge_record(
            sample_id=sample_id,
            target_audio_path=target_audio_path,
            target_preset=target_preset,
            gt_wavetable_names=gt_names,
            pool=pool,
            candidate_audio=candidate_audio,
            name_to_emb=name_to_emb,
            active_oscs=active_oscs,
            omni_server=args.omni_server,
            omni_model=args.omni_model,
            stage2_server=stage2_server,
            stage2_model=stage2_model,
            judge_output_dir=args.judge_output_dir,
        )
        return result.record

    out_path = args.out_jsonl
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []

    def _safe_process(entry):
        try:
            return entry, _process(entry), None
        except Exception as exc:
            import traceback
            return entry, None, (exc, traceback.format_exc())

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool_exec:
            futs = [pool_exec.submit(_safe_process, e) for e in entries]
            for i, fut in enumerate(as_completed(futs)):
                entry, rec, err = fut.result()
                if err:
                    print(f"WARNING: {entry.get('sample_id', '?')} failed: {err[0]}")
                    print(err[1])
                elif rec:
                    all_records.append(rec)
                    print(f"[{i + 1}/{len(entries)}] {entry['sample_id']}: pool={rec['meta']['pool_size']}, "
                          f"selected={rec['meta']['selected_tuple']}, correct={rec['meta']['judge_correct']}",
                          flush=True)
    else:
        for i, entry in enumerate(entries):
            entry, rec, err = _safe_process(entry)
            if err:
                print(f"WARNING: {entry.get('sample_id', '?')} failed: {err[0]}")
                print(err[1])
            elif rec:
                all_records.append(rec)
                print(f"[{i + 1}/{len(entries)}] {entry['sample_id']}: pool={rec['meta']['pool_size']}, "
                      f"selected={rec['meta']['selected_tuple']}, correct={rec['meta']['judge_correct']}",
                      flush=True)

    with open(out_path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_records)} judge records to {out_path}", flush=True)


if __name__ == "__main__":
    main()
