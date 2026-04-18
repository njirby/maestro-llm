#!/usr/bin/env python3
"""Build judge-agent SFT v3: audition-based tuple selection from a combined search pool.

The judge agent is invoked by the main agent AFTER the search agents have each
returned a shortlist. Its job is to look at the combined pool in one auditory
view (each candidate audio labelled with its wavetable name) and select the N
candidates that together best capture the target — where N = the number of
active oscillator slots in the target preset.

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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.agent_sft_common import (  # type: ignore
    assert_valid_ms_swift_multiturn_record,
    build_clap_shortlist_data,
    build_name_embedding_map,
    ensure_candidate_probes_for_names,
    extract_gt_wavetable_names,
    is_clap_selected,
    load_index_rows,
    load_manifest_entries,
    load_wavetable_lib,
    select_probe_rows_by_name,
)
from scripts.build_main_agent_sft_v2 import (  # type: ignore
    _check_server_reachable,
    _llm_post,
)


def _b64(path: str | Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _tool_call(name: str, arguments: dict) -> dict:
    return {"role": "tool_call", "content": json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)}


# Tools available to the judge agent at inference (matches main-agent dispatch prompt)
_JUDGE_TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command (used to render probes and write the selection file).",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
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
            if is_clap_selected(n, gt_names, name_to_emb, threshold=0.92):
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


def omni_judge_audition(
    target_wav: Path,
    pool: list[str],
    candidate_audio: dict[str, Path],
    n_osc_slots: int,
    omni_server: str,
    omni_model: str,
    timeout: float = 300.0,
) -> str:
    """Omni listens to target + all pool candidates at once, writes per-candidate
    observations and a final selection recommendation.

    Returns the raw text. Each candidate audio is labelled in the prompt with its
    wavetable name so the model can associate audio-position with name.
    """
    valid_names = [n for n in pool if n in candidate_audio]
    content = [
        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{_b64(target_wav)}"}},
    ]
    for name in valid_names:
        content.append(
            {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{_b64(candidate_audio[name])}"}}
        )
    audio_labels = "\n".join(
        f"  Audio {i + 2}: \"{name}\"" for i, name in enumerate(valid_names)
    )
    content.append({
        "type": "text",
        "text": (
            f"Audio 1 is the TARGET sound we need to match.\n"
            f"Audios 2-{len(valid_names) + 1} are pool candidates returned by search "
            f"agents (each labelled with its wavetable name):\n"
            f"{audio_labels}\n\n"
            f"Listen to each candidate against the target. Write ONE short sentence per "
            f"candidate describing its raw character and whether it could serve as a "
            f"building block for the target once filters, envelopes, and effects are applied.\n\n"
            f"Then on a final line, recommend which {n_osc_slots} candidate(s) together "
            f"would best capture the target — format: "
            f"'RECOMMENDATION: [name1, name2]: <one sentence on why this combination works>'.\n\n"
            f"Format per line: '<name>': <one-sentence assessment>\n"
            f"Use the EXACT names listed above, natural language, no snake_case."
        ),
    })
    try:
        r = _llm_post(
            f"{omni_server}/v1/chat/completions",
            {"model": omni_model,
             "messages": [{"role": "user", "content": content}],
             "max_tokens": 900, "temperature": 0.4},
            timeout=timeout,
        )
        return r["choices"][0]["message"]["content"].strip()
    except Exception:
        # Fallback: trivial per-candidate lines + selection
        lines = [f'"{n}": Candidate evaluated against target.' for n in valid_names]
        return "\n".join(lines) + f"\nRECOMMENDATION: [] : (fallback — audio model unavailable)"


def stage2_format_judge_response(
    omni_observations: str,
    pool: list[str],
    selected_tuple: list[str],
    n_osc_slots: int,
    gt_names_in_pool: list[str],
    stage2_server: str,
    stage2_model: str,
    timeout: float = 180.0,
) -> str:
    """Stage 2 text model formats Omni's observations into the final judge response,
    with the selection pre-determined by the oracle. Stage 2 writes rationale only."""
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
        f"(n_osc_slots = {n_osc_slots})\n"
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
        return r["choices"][0]["message"]["content"].strip()
    except Exception:
        per_cand = "\n".join(f'"{n}": Candidate evaluated.' for n in pool)
        return f"{per_cand}\nSELECTED: [{selected_list_str}]: Best combination from pool for the target."


def _extract_final_reasoning(stage2_text: str) -> str:
    """Pull the SELECTED: line's justification for use in the output JSON file."""
    for line in reversed(stage2_text.splitlines()):
        line = line.strip()
        if line.upper().startswith("SELECTED:"):
            # Format: "SELECTED: [...]: <reasoning>"
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
) -> dict | None:
    """Build one SFT judge record. Returns None if the pool is empty."""
    if not pool:
        return None

    messages: list[dict] = []
    audio_assets: list[str] = [str(target_audio_path)]

    n_osc_slots = len(active_oscs)
    probe_out_dir = f"/tmp/judge_probes/{sample_id}"

    pool_str = ", ".join(f'"{n}"' for n in pool)
    names_arg = ",".join(json.dumps(n) for n in pool)
    judge_agent_id = f"wavetable_judge_{sample_id}"
    judge_output_file = judge_output_dir / f"{judge_agent_id}.json"
    judge_output_file.parent.mkdir(parents=True, exist_ok=True)

    # User prompt — mirrors what the main agent's Agent tool_call sends
    messages.append({
        "role": "user",
        "content": (
            f"<audio>\n"
            f"Pool candidates from search agents: [{pool_str}].\n"
            f"Target uses {n_osc_slots} active oscillator(s). Render probes for each "
            f"candidate, listen alongside the target, and select the {n_osc_slots} "
            f"candidate(s) that together best capture the target's character. "
            f"Write your selection to {judge_output_file} as JSON: "
            f'{{"tuple": [...], "n_osc_slots": {n_osc_slots}, "reasoning": "..."}}.'
        ),
    })

    # Render probes for all pool candidates
    render_cmd = (
        f"python skills/vital/scripts/render_probes.py --names {names_arg} "
        f"--out-dir {probe_out_dir}"
    )
    messages.append({
        "role": "assistant",
        "content": f"Rendering probes for all {len(pool)} pool candidates in one batch.",
    })
    messages.append(_tool_call("bash", {"command": render_cmd}))

    rendered_entries = []
    for name in pool:
        if name in candidate_audio:
            audio_assets.append(str(candidate_audio[name]))
            rendered_entries.append({
                "name": name,
                "out": str(candidate_audio[name]),
                "audio": "<audio>",
            })
    messages.append({
        "role": "tool_response",
        "content": json.dumps({"status": "ok", "rendered": rendered_entries}, ensure_ascii=False),
    })

    # Build-time oracle: GT-in-pool + CLAP-best proxy per osc slot
    wts = target_preset.get("settings", {}).get("wavetables", [])
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
    gts_in_pool = [g for g in gt_wavetable_names if g in pool]

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
        omni_response = "\n".join(f'"{n}": Candidate evaluated.' for n in pool) + \
            f"\nRECOMMENDATION: [{', '.join(repr(n) for n in selected_tuple)}] : Best combination from pool."

    # Stage 2: format with oracle-selected tuple
    stage2_response = stage2_format_judge_response(
        omni_observations=omni_response,
        pool=pool,
        selected_tuple=selected_tuple,
        n_osc_slots=n_osc_slots,
        gt_names_in_pool=gts_in_pool,
        stage2_server=stage2_server,
        stage2_model=stage2_model,
    )
    final_reasoning = _extract_final_reasoning(stage2_response)

    # Assistant narration: the whole deliberation
    messages.append({
        "role": "assistant",
        "content": (
            f"Listening to the target alongside all {len(pool)} pool candidates at once.\n\n"
            f"{stage2_response}\n\n"
            f"Writing selection to output file."
        ),
    })

    # Write output file both on disk (for real) and via bash snippet (for conversation)
    judge_output = {
        "status": "completed",
        "agentId": judge_agent_id,
        "tuple": selected_tuple,
        "n_osc_slots": n_osc_slots,
        "reasoning": final_reasoning,
    }
    with open(judge_output_file, "w") as f:
        json.dump(judge_output, f)
        f.write("\n")

    # Use a simple heredoc bash command to write the output file — matches the style
    # of the main agent's inline python tool-call snippets (portable across harness).
    write_cmd = (
        f"python - <<'PY'\n"
        f"import json\n"
        f"from pathlib import Path\n"
        f"p = Path({json.dumps(str(judge_output_file))})\n"
        f"p.parent.mkdir(parents=True, exist_ok=True)\n"
        f"with open(p, 'w') as f:\n"
        f"    json.dump({json.dumps(judge_output, ensure_ascii=False)}, f)\n"
        f"    f.write('\\n')\n"
        f"print(json.dumps({{'status': 'ok', 'file': str(p)}}))\n"
        f"PY"
    )
    messages.append(_tool_call("bash", {"command": write_cmd}))
    messages.append({
        "role": "tool_response",
        "content": json.dumps({"status": "ok", "file": str(judge_output_file)}, ensure_ascii=False),
    })
    # Closing assistant turn — confirms the selection and signals the main agent
    # can now read the output file.
    tuple_str = ", ".join(repr(n) for n in selected_tuple)
    messages.append({
        "role": "assistant",
        "content": (
            f"Selection written. Final tuple: [{tuple_str}]. The main agent can now read "
            f"{judge_output_file} and apply the chosen wavetables."
        ),
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
            "selected_tuple": selected_tuple,
            "judge_correct": judge_correct,
            "output_file": str(judge_output_file),
        },
    }

    assert_valid_ms_swift_multiturn_record(record)
    return record


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

        return build_judge_record(
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
