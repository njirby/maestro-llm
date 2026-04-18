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
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.agent_sft_common import (
    assert_valid_ms_swift_multiturn_record,
    build_clap_shortlist_data,
    build_gt_similarity_pool,
    build_name_embedding_map,
    ensure_candidate_probes_for_names,
    extract_gt_wavetable_names,
    is_clap_selected,
    load_index_rows,
    load_manifest_entries,
    load_wavetable_lib,
    select_probe_rows_by_name,
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


def omni_batch_compare(
    target_wav: Path,
    candidate_names: list[str],
    candidate_audio: dict[str, Path],
    archetype: str,
    omni_server: str,
    omni_model: str,
) -> str:
    """Omni Stage 1: hear target + N candidates, write per-candidate comparison."""
    content: list[dict] = [
        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{_b64(target_wav)}"}},
    ]
    valid_names = [n for n in candidate_names if n in candidate_audio]
    for name in valid_names:
        content.append(
            {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{_b64(candidate_audio[name])}"}}
        )

    candidate_list = "\n".join(
        f"  Audio {i + 2}: \"{name}\"" for i, name in enumerate(valid_names)
    )
    content.append({"type": "text", "text": (
        f"You are a sound design assistant. Audio 1 is the TARGET sound.\n"
        f"Audios 2-{len(valid_names) + 1} are candidate wavetables rendered through a "
        f"default synthesizer preset.\n\n"
        f"Candidate list:\n{candidate_list}\n\n"
        f"For EACH candidate, write one sentence about how its raw character relates to "
        f"the target — could this wavetable be a building block for recreating the target "
        f"once filters, envelopes, and effects are applied?\n\n"
        f"Format: '\"<name>\": <one sentence>'\n"
        f"Be specific about harmonic character, brightness, texture, and attack shape."
    )})

    try:
        r = _llm_post(
            f"{omni_server}/v1/chat/completions",
            {"model": omni_model, "messages": [{"role": "user", "content": content}],
             "max_tokens": 500, "temperature": 0.4},
            timeout=180.0,
        )
        return r["choices"][0]["message"]["content"].strip()
    except Exception:
        return "\n".join(f'"{n}": Candidate wavetable evaluated.' for n in valid_names)


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
    """
    # Build candidate display with pre-determined labels
    candidate_lines = []
    for name in candidate_names:
        is_gt = name in gt_names_in_batch
        selected = selection_labels.get(name, False)
        label = "Selected" if selected else "Not selected"
        if is_gt and gt_transform_description:
            candidate_lines.append(
                f"  '{name}': {label} (GT — target preset processes this with: {gt_transform_description})"
            )
        else:
            candidate_lines.append(f"  '{name}': {label}")
    candidates_block = "\n".join(candidate_lines)

    prompt = (
        f"You are a sound design assistant evaluating wavetable candidates for a "
        f"the target sound.\n\n"
        f"Audio observations from listening:\n{omni_observations}\n\n"
        f"Candidates and their selection status (pre-determined):\n{candidates_block}\n\n"
        f"Write EXACTLY {len(candidate_names)} lines, one per candidate.\n"
        f"Format: '<name>': <one sentence assessment>. Selected / Not selected.\n\n"
        f"Rules:\n"
        f"- Use the EXACT names above (no variants, no abbreviations).\n"
        f"- For Selected candidates: explain what raw quality makes it compatible "
        f"with the target after synthesis processing (filtering, enveloping, effects).\n"
        f"- For Not selected candidates: explain what makes it incompatible.\n"
        f"- For GT candidates (marked with processing details): reference at least 2 "
        f"specific transforms from the description and explain how they reshape the "
        f"raw wavetable toward the target.\n"
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


_TOOL_SPECS = json.dumps([{
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute shell/Python commands to render and evaluate wavetable candidates.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    },
}], ensure_ascii=False)


def _tool_call(name: str, arguments: dict) -> dict:
    return {"role": "tool_call", "content": json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)}


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
    rng: random.Random,
    candidates_per_batch: int = 8,
    shortlist_dir: Path | None = None,
    clap_threshold: float = 0.92,
) -> dict | None:
    """Build one search agent SFT record with iterative batch listening.

    Uses real bash commands to skills/vital/scripts/list_wavetables.py and
    skills/vital/scripts/render_probes.py. The search agent's task is defined by
    a dense index range [shard_start, shard_end). GT wavetables within the
    range are oracle-forced onto the final shortlist.
    """
    if shard_end <= shard_start:
        return None

    # Names in shard (by dense index)
    shard = [idx_to_name[i] for i in range(shard_start, shard_end) if i in idx_to_name]
    if not shard:
        return None

    gt_set = set(gt_wavetable_names)
    gt_in_shard = [n for n in shard if n in gt_set]

    # Compute processing chain description for GT grounding
    gt_transform_desc = describe_key_transforms(target_preset) if gt_in_shard else ""

    # Arrange batches: GT should NOT appear in batch 1 (teach the model to search,
    # not get lucky). Place GT in batch 2 or later.
    non_gt = [n for n in shard if n not in gt_set]
    rng.shuffle(non_gt)

    # Build batch sequence: batch 1 = all non-GT, batch 2+ = include GT
    all_ordered = []
    gt_inserted = False
    batch_idx = 0
    ptr = 0
    while ptr < len(non_gt) or not gt_inserted:
        batch = []
        # First batch: only non-GT
        if batch_idx == 0:
            batch = non_gt[ptr:ptr + candidates_per_batch]
            ptr += len(batch)
        else:
            # Insert GT candidates into this batch if not yet done
            if not gt_inserted and gt_in_shard:
                batch = list(gt_in_shard)
                gt_inserted = True
                remaining_slots = candidates_per_batch - len(batch)
                batch.extend(non_gt[ptr:ptr + remaining_slots])
                ptr += remaining_slots
            else:
                batch = non_gt[ptr:ptr + candidates_per_batch]
                ptr += len(batch)
        if not batch:
            break
        rng.shuffle(batch)
        all_ordered.append(batch)
        batch_idx += 1

    if not all_ordered:
        return None

    # If GT wasn't in shard at all, that's fine — the agent legitimately finds nothing
    # from this shard that matches. This is realistic for some shards.

    # Build messages
    messages: list[dict] = []
    audio_assets: list[str] = [str(target_audio_path)]
    all_batch_notes: list[str] = []

    # Output directory for rendered probes (real path, reused at inference)
    probe_out_dir = f"/tmp/search_probes/{sample_id}_agent{agent_idx}"

    messages.append({
        "role": "user",
        "content": (
            f"<audio>\nEvaluate wavetables at indices "
            f"{shard_start}-{shard_end - 1} from the library and return a shortlist "
            f"of 2-4 names that could serve as building blocks for recreating the target. "
            f"Use `python skills/vital/scripts/list_wavetables.py --start {shard_start} --end {shard_end}` "
            f"to see the names in your range, and "
            f"`python skills/vital/scripts/render_probes.py --idxs A,B,C --out-dir {probe_out_dir}` "
            f"to render and listen to each candidate."
        ),
    })

    # Step 1: list_wavetables — agent fetches names in its range
    list_cmd = f"python skills/vital/scripts/list_wavetables.py --start {shard_start} --end {shard_end}"
    shard_entries = [
        {"idx": i, "name": idx_to_name[i]}
        for i in range(shard_start, shard_end) if i in idx_to_name
    ]
    messages.append({
        "role": "assistant",
        "content": f"Fetching candidate names at indices {shard_start}-{shard_end - 1}.",
    })
    messages.append(_tool_call("bash", {"command": list_cmd}))
    messages.append({
        "role": "tool_response",
        "content": json.dumps({
            "wavetables": shard_entries,
            "start": shard_start,
            "end": shard_end,
            "count": len(shard_entries),
        }, ensure_ascii=False),
    })

    selected_so_far: list[str] = []  # CLAP-selected candidates, accumulated across batches
    pending_notes: str | None = None

    for bi, batch in enumerate(all_ordered):
        batch_num = bi + 1
        gt_in_batch = [n for n in batch if n in gt_set]

        # Real bash command: render batch candidates
        batch_idxs = [name_to_idx[n] for n in batch if n in name_to_idx]
        idxs_csv = ",".join(str(i) for i in batch_idxs)
        render_cmd = (
            f"python skills/vital/scripts/render_probes.py "
            f"--idxs {idxs_csv} --out-dir {probe_out_dir}"
        )

        import re as _re
        def _slugify(s: str) -> str:
            return (_re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_") or "unnamed")[:80]

        rendered_entries = []
        for name, idx in zip(batch, batch_idxs):
            out_path = f"{probe_out_dir}/wt_{idx:04d}_{_slugify(name)}.wav"
            entry = {"idx": idx, "name": name, "out": out_path}
            if name in candidate_audio:
                audio_assets.append(str(candidate_audio[name]))
                entry["audio"] = "<audio>"
            rendered_entries.append(entry)

        intro = f"Rendering batch {batch_num} (indices {', '.join(str(i) for i in batch_idxs)})."
        if pending_notes:
            intro = f"{pending_notes}\n\n{intro}"
            pending_notes = None

        messages.append({"role": "assistant", "content": intro})
        messages.append(_tool_call("bash", {"command": render_cmd}))
        messages.append({
            "role": "tool_response",
            "content": json.dumps({"status": "ok", "rendered": rendered_entries}, ensure_ascii=False),
        })

        # CLAP-based selection: determine Selected/Not selected for each candidate
        selection_labels: dict[str, bool] = {}
        for name in batch:
            selection_labels[name] = is_clap_selected(
                name, gt_wavetable_names, name_to_emb, threshold=clap_threshold
            )

        # Accumulate selected candidates (builder-controlled, no model parsing needed)
        batch_selected = [n for n in batch if selection_labels[n]]
        for n in batch_selected:
            if n not in selected_so_far:
                selected_so_far.append(n)

        # Omni Stage 1: batch comparison (audio observations)
        if omni_server:
            omni_obs = omni_batch_compare(
                target_wav=target_audio_path,
                candidate_names=batch,
                candidate_audio=candidate_audio,
                archetype=archetype,
                omni_server=omni_server,
                omni_model=omni_model,
            )
        else:
            omni_obs = "\n".join(f'"{n}": Candidate evaluated.' for n in batch)

        # Stage 2: write reasoning ONLY (labels are pre-determined by CLAP).
        # No "Candidates so far" echo — the running shortlist is the agent's own
        # internal state, not something it narrates in-token. The final shortlist
        # is materialised via an explicit file write at the end of the run
        # (claw-code-style handoff).
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

    # Write shortlist output file. This is the claw-code-style file-based handoff —
    # the dispatcher (main agent) will `cat` this file to consume the result.
    #
    # The write is BOTH persisted to disk (for real, so the main agent's SFT builder
    # can reference it at inference time) AND emitted as an explicit bash tool_call
    # in the conversation transcript, so the model learns the full protocol: "do the
    # work, then write the result to the file my dispatcher reads."
    shortlist_path: str | None = None
    if shortlist_dir is not None:
        shortlist_dir.mkdir(parents=True, exist_ok=True)
        shortlist_path = str(shortlist_dir / f"{sample_id}_search_{agent_idx}.json")
        payload = {
            "status": "completed",
            "agentId": f"search_{sample_id}_{agent_idx}",
            "shardStart": shard_start,
            "shardEnd": shard_end,
            "shortlist": selected_so_far,
            "nBatches": len(all_ordered),
        }
        # Persist to disk (so the file actually exists at the path the transcript references)
        with open(shortlist_path, "w") as f:
            json.dump(payload, f)
            f.write("\n")

        # Merge the final narration and the "writing file" intro into ONE assistant
        # turn, then emit the visible tool_call + tool_response + closing assistant.
        # Two back-to-back assistant turns would fail the validator.
        messages.append({
            "role": "assistant",
            "content": (
                f"{final_text}\n\n"
                f"Writing the final shortlist to the output file for the dispatcher "
                f"to consume."
            ),
        })
        write_cmd = (
            f"python - <<'PY'\n"
            f"import json\n"
            f"from pathlib import Path\n"
            f"p = Path({json.dumps(shortlist_path)})\n"
            f"p.parent.mkdir(parents=True, exist_ok=True)\n"
            f"with open(p, 'w') as f:\n"
            f"    json.dump({json.dumps(payload, ensure_ascii=False)}, f)\n"
            f"    f.write('\\n')\n"
            f"print(json.dumps({{'status': 'ok', 'file': str(p)}}))\n"
            f"PY"
        )
        messages.append(_tool_call("bash", {"command": write_cmd}))
        messages.append({
            "role": "tool_response",
            "content": json.dumps({"status": "ok", "file": shortlist_path}, ensure_ascii=False),
        })
        # Closing assistant turn — keeps the validator's last-message-is-assistant
        # invariant, and signals task completion explicitly.
        messages.append({
            "role": "assistant",
            "content": (
                f"Shortlist written. {len(selected_so_far)} candidate(s) flagged for "
                f"the judge agent."
            ),
        })
    else:
        # No shortlist_dir — single final assistant turn with the narration only.
        messages.append({"role": "assistant", "content": final_text})

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
            "final_shortlist": selected_so_far,
            "gt_on_shortlist": [n for n in selected_so_far if n in gt_set],
            "shortlist_output_file": shortlist_path,
        },
    }

    assert_valid_ms_swift_multiturn_record(record)
    return record


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
    ap.add_argument("--shortlist-dir", type=Path, default=None,
        help="Directory to write per-agent shortlist JSON files (used by main agent's cat tool calls).")
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
        rng = random.Random(args.seed + sid_seed)

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

        with serial_lock:
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
            agent_rng = random.Random(args.seed + sid_seed + ai + 1)
            agent_jobs.append((ai, start, end, agent_rng))

        def _run_agent(job):
            ai, start, end, agent_rng = job
            rec = build_search_record(
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
                rng=agent_rng,
                candidates_per_batch=args.candidates_per_batch,
                shortlist_dir=args.shortlist_dir,
            )
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
