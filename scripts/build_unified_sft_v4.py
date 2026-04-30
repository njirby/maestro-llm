#!/usr/bin/env python3
"""Unified top-down SFT pipeline v4 — real subagent calls, no oracle simulation.

Replaces the independent 4-script pipeline with a single top-down build where
the main agent builder calls real ``build_search_record()``,
``build_judge_record()``, and ``build_transcription_record()`` inline.

Subagent outputs feed forward into the main conversation — no oracle forcing.
This eliminates distribution mismatch and ensures file-causality by construction.

Output: 4 JSONL files (main, search, judge, transcription) in one pass.

Usage:
    python scripts/build_unified_sft_v4.py \
        --manifest outputs/smoke_test_v18/manifest.jsonl \
        --out-dir outputs/smoke_v3 --suffix v31 \
        --max-samples 8 --workers 4 --clap-device cuda
"""
from __future__ import annotations

import argparse
import copy
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

# ---------------------------------------------------------------------------
# Import everything we reuse from the existing builders
# ---------------------------------------------------------------------------

from scripts.agent_sft_common import (
    ClapEmbedder,
    assert_valid_ms_swift_multiturn_record,
    build_clap_shortlist_data,
    build_gt_similarity_pool,
    build_list_wavetables_slice_snippet,
    build_list_wavetables_total_snippet,
    build_name_embedding_map,
    build_param_search_snippet,
    build_reaper_render_snippet,
    build_render_probes_snippet,
    build_render_tuple_snippet,
    build_render_verify_snippet,
    ensure_candidate_probes_for_names,
    extract_gt_wavetable_names,
    load_index_rows,
    load_manifest_entries,
    load_wavetable_lib,
    make_agent_id,
    select_probe_rows_by_name,
    simulate_param_search,
    write_agent_manifest,
    _bash_tool_response,
    _BUILD_CHUNK_HELPER,
    _emit_listen_sequence,
    _read_tool_response_audio,
    _REAPY_HELPER,
    _tool_call,
    _WT_DISCOVER_SNIPPET,
    _wrap_as_bash,
)

# Reuse main-agent-v3 helpers (all the subsystem taxonomy, batch construction,
# rendering, Omni/Stage2, diagnosis, etc.)
from scripts.build_main_agent_sft_v3 import (
    SUBSYSTEM_ORDER,
    SubsystemBatch,
    _batch_search_queries,
    _extract_plan_bullet,
    _init_preset,
    _json_key_to_display,
    _json_key_to_reaper_display,
    _JSON_KEY_TO_REAPER,
    _REAPER_PARAM_DUMP,
    _V3_TOOL_SPECS,
    _VITAL_DISPLAY_NAMES,
    build_batch_action_snippet,
    build_batches_from_diff,
    build_diagnosis_subsystem_truth,
    denormalize_batch_params,
    extract_diagnosis_subsystems_mentioned,
    format_subsystem_diff_summary,
    MistakeInfo,
    inject_mistakes,
    omni_stage1_diagnose,
    omni_stage1_verdict,
    presentation_subsystem,
    render_cumulative_audio,
    stage2_batch_check,
    stage2_correction_intro,
    stage2_diagnosis,
    stage2_verdict,
)
from scripts.build_main_agent_sft_v2 import (
    _build_listen_probe_command,
    _check_server_reachable,
    _llm_post,
    _step_remaining_gap,
    llm_post_stats,
)

# Real subagent builders
from scripts.build_search_agent_sft_v2 import (
    SearchResult,
    build_search_record,
)
from scripts.build_judge_agent_sft_v3 import (
    JudgeResult,
    build_judge_record,
)
from scripts.build_transcription_agent_sft_v4 import (
    TranscriptionResult,
    build_transcription_record_v4,
    load_notes_from_midi,
)

from maestro.synth import path_gen as _pg
from maestro.synth.path_gen import _denormalize, _normalize, _param_family
from maestro.synth.preset_gen import generate_preset
from maestro.render.dawdreamer import render_preset_audio, make_probe_notes, notes_from_dicts

import numpy as np

from scripts.preset_perceptual_summary import summarize_residual_delta_perceptual


# ---------------------------------------------------------------------------
# Main builder — modified to call real subagent builders
# ---------------------------------------------------------------------------


def build_record(
    *,
    entry: dict,
    args: argparse.Namespace,
    embedder: ClapEmbedder,
    shortlist_data: dict,
    selected_by_name: dict,
    wavetable_lib: list[dict],
    index_rows: list[dict],
    candidate_audio: dict[str, Path],
    stage2_server: str,
    stage2_model: str,
    serial_lock: threading.Lock,
    notes: list,
) -> tuple[dict | None, list[dict], list[dict], list[dict]]:
    """Build one unified SFT record.

    Returns (main_record, search_records, judge_records, transcription_records).
    main_record is None to skip; subagent lists may be empty.
    """
    import time as _time
    _t0 = _time.monotonic()

    search_records: list[dict] = []
    judge_records: list[dict] = []
    transcription_records: list[dict] = []

    def _log(msg: str) -> None:
        elapsed = _time.monotonic() - _t0
        print(f"  [{sample_id}] {elapsed:6.1f}s  {msg}", flush=True)

    sample_id = str(entry["sample_id"])
    archetype = str(entry.get("archetype", "synth"))
    target_audio_path = Path(entry.get("gt_wav") or entry.get("gt_probe_wav"))
    default_audio_path = Path(entry["default_wav"]) if entry.get("default_wav") else None
    if default_audio_path is None:
        return None, [], [], []

    source_midi_path = entry.get("source_midi_path")
    if source_midi_path and Path(source_midi_path).exists():
        try:
            _midi_notes = load_notes_from_midi(source_midi_path)
            if _midi_notes:
                notes = notes_from_dicts(_midi_notes)
        except Exception:
            pass

    target_preset_path = entry.get("target_preset_path")
    if not target_preset_path:
        path_file = entry.get("path_file")
        if path_file:
            with open(path_file) as f:
                pd = json.load(f)
            target_preset_path = pd.get("target_preset_path")
    if not target_preset_path:
        return None, [], [], []
    with open(target_preset_path) as f:
        target_preset = json.load(f)

    target_modulations = [
        m for m in target_preset.get("settings", {}).get("modulations", [])
        if m.get("source") and m.get("destination")
    ]

    init_preset = _init_preset()

    sid_seed = int(hashlib.sha1(sample_id.encode()).hexdigest()[:8], 16)
    sample_rng = random.Random(int(args.seed) + sid_seed)

    # Random init preset (independent RNG stream)
    _random_init_rate = float(getattr(args, "random_init_rate", 0.0))
    use_random_init = False
    if _random_init_rate > 0.0:
        _init_rng = random.Random(int(args.seed) + sid_seed + 7777)
        if _init_rng.random() < _random_init_rate:
            use_random_init = True
            _gen_rng = random.Random(int(args.seed) + sid_seed + 8888)
            init_preset = generate_preset(archetype, _gen_rng, wavetable_lib=wavetable_lib)
            _random_init_wav = Path(args.probe_dir) / f"{sample_id}_random_init.wav"
            _random_init_wav.parent.mkdir(parents=True, exist_ok=True)
            render_cumulative_audio(init_preset, notes, _random_init_wav)
            default_audio_path = _random_init_wav

    # ---- WT search setup ----
    gt_names_list = list(extract_gt_wavetable_names(Path(target_preset_path)))
    if not gt_names_list:
        return None, [], [], []

    _seen_wt: set[str] = set()
    _unique_wts: list[str] = []
    for wt in wavetable_lib:
        if not isinstance(wt, dict) or "name" not in wt:
            continue
        if wt["name"] in _seen_wt:
            continue
        _seen_wt.add(wt["name"])
        _unique_wts.append(wt["name"])
    idx_to_name_full: dict[int, str] = {i: n for i, n in enumerate(_unique_wts)}
    name_to_idx_full: dict[str, int] = {n: i for i, n in idx_to_name_full.items()}
    total_named = len(_unique_wts)

    active_oscs: list[int] = []
    for osc_idx in range(3):
        on_key = f"osc_{osc_idx + 1}_on"
        level_key = f"osc_{osc_idx + 1}_level"
        osc_on = float(target_preset.get("settings", {}).get(on_key, 0.0)) > 0.5
        osc_level = float(target_preset.get("settings", {}).get(level_key, 0.0)) > 0.01
        if osc_on and osc_level:
            active_oscs.append(osc_idx)
    if not active_oscs:
        active_oscs = [0]

    slice_size = int(getattr(args, "candidates_per_slice", 48))
    n_agents = int(args.num_agents)
    max_rounds = int(getattr(args, "max_search_rounds", 3))
    stride = max(1, total_named // n_agents)

    def _compute_slices(base: int) -> list[int]:
        starts = []
        for i in range(n_agents):
            start = (base + i * stride) % total_named
            if start + slice_size > total_named:
                start = max(0, total_named - slice_size)
            starts.append(start)
        return starts

    gt_idxs = [name_to_idx_full[n] for n in gt_names_list if n in name_to_idx_full]
    force_research_rate = float(getattr(args, "force_research_rate", 0.30))
    force_miss = sample_rng.random() < force_research_rate

    base_offset = sample_rng.randrange(stride)
    slice_starts = _compute_slices(base_offset)
    if gt_idxs and not force_miss:
        step = max(1, stride // 4)
        tried = 0
        while not any(s <= gi < (s + slice_size) for s in slice_starts for gi in gt_idxs) and tried < total_named:
            base_offset = (base_offset + step) % stride
            slice_starts = _compute_slices(base_offset)
            tried += step

    # ---- Begin messages ----
    messages: list[dict] = []
    audio_assets: list[str] = [str(target_audio_path)]

    # Edge case (~5%): user says "recreate this sound" without audio
    no_audio_rate = float(getattr(args, "no_audio_rate", 0.05))
    if sample_rng.random() < no_audio_rate:
        messages.append({
            "role": "user",
            "content": "Recreate this sound in Vital.",
        })
        messages.append({
            "role": "assistant",
            "content": (
                "I don't see an audio clip attached to your message. Please select "
                "the audio item in REAPER that you want me to recreate (click on the "
                "clip so it's highlighted) and try again — I need to hear the target "
                "sound before I can start matching it in Vital."
            ),
        })
        record = {
            "id": sample_id,
            "task_type": "main",
            "tools": _V3_TOOL_SPECS,
            "messages": messages,
            "audios": [],
            "assets": {
                "target_audio": "", "current_audio": "",
                "candidate_audio": [], "selected_candidates": [], "selected_tuples": [],
            },
            "labels": {"gt_wavetable_names": gt_names_list, "applied_wavetables": []},
            "meta": {
                "pipeline_version": "v4_unified",
                "sample_id": sample_id,
                "archetype": archetype,
                "agent": "main",
                "variant": "no_audio_selected",
                "batch_labels": [],
                "diagnosis_subsystems_mentioned": [],
                "diagnosis_subsystems_truth": [],
                "injected_mistakes": [],
                "total_correction_turns": 0,
                "mistake_caught": False,
                "path_complete": False,
                "n_remaining": 0,
                "commentary_mode": "two_stage",
                "num_agents": int(args.num_agents),
                "pool_top_k": int(args.pool_top_k),
                "max_batches": int(args.max_batches),
                "per_param_mistake_rate": float(getattr(args, "per_param_mistake_rate", 0.10) or 0.10),
            },
        }
        assert_valid_ms_swift_multiturn_record(record)
        return record, [], [], []

    messages.append({
        "role": "user",
        "content": "<audio>\nRecreate this sound in Vital.",
    })

    # ---- Skill discovery + load (same as v3) ----
    _skill_name = "vital"
    _skills_root = Path(__file__).resolve().parents[1] / "skills"
    _skill_md_path = _skills_root / _skill_name / "SKILL.md"
    _available_skill_paths = sorted(str(p.relative_to(_skills_root.parent)) for p in _skills_root.glob("*/SKILL.md"))
    try:
        _skill_md_text = _skill_md_path.read_text()
    except Exception:
        _skill_md_text = ""
    _skill_description = ""
    if _skill_md_text.startswith("---"):
        _fm_end = _skill_md_text.find("---", 3)
        if _fm_end != -1:
            _fm = _skill_md_text[3:_fm_end]
            _in_desc = False
            _desc_lines: list[str] = []
            for _line in _fm.splitlines():
                if _line.startswith("description:"):
                    _in_desc = True
                    _desc_lines.append(_line[len("description:"):].strip())
                    continue
                if _in_desc:
                    if _line.startswith((" ", "\t")):
                        _desc_lines.append(_line.strip())
                    else:
                        break
            _skill_description = " ".join(_desc_lines).strip()

    _log("start — skill discovery + transcription")
    messages.append({
        "role": "assistant",
        "content": "Let me see which skills are available for this plugin.",
    })
    messages.append(_tool_call("Bash", {"command": "ls skills/*/SKILL.md"}))
    messages.append(_bash_tool_response("\n".join(_available_skill_paths) + "\n"))

    messages.append({
        "role": "assistant",
        "content": f"The {_skill_name} skill matches. Loading it for the plugin-specific instructions, helper-script paths, and recreation strategy.",
    })
    messages.append(_tool_call("Skill", {"skill": _skill_name, "args": ""}))
    messages.append({
        "role": "tool_response",
        "content": json.dumps({
            "skill": _skill_name,
            "path": str(_skill_md_path),
            "args": "",
            "description": _skill_description,
            "prompt": _skill_md_text,
        }, ensure_ascii=False),
    })

    # Render baseline (needed for batch-rendering diffs later), but skip
    # listening — the model already heard the result via the judge agent.
    _default_render_cmd = _wrap_as_bash(build_render_verify_snippet(
        out_path=str(default_audio_path),
        notes_override=list(notes),
    ))
    messages.append(_tool_call("Bash", {"command": _default_render_cmd}))
    messages.append(_bash_tool_response(
        json.dumps({"rendered": str(default_audio_path), "ok": True})
    ))

    # --- TRANSCRIPTION BLOCK ---
    # Create a REAPER track and dispatch the transcription subagent.
    # Verification and mistake recovery now happen inside the transcription
    # agent itself (v4) — the main agent just dispatches once and proceeds.
    transcription_summary_text = ""
    _trans_output_file: str | None = None
    if source_midi_path and Path(source_midi_path).exists():
        try:
            _trans_notes = load_notes_from_midi(source_midi_path)
        except Exception:
            _trans_notes = []
        if _trans_notes:
            _trans_n_notes = len(_trans_notes)
            _trans_duration_s = round(
                max(n["start_s"] + n["dur_s"] for n in _trans_notes), 2,
            )
            _trans_track_idx = 0
            _trans_track_name = "target_melody"
            _trans_agent_id = make_agent_id(sample_id, "melody_transcription")
            _trans_agent_dir = f"/tmp/agents/{sample_id}"
            _trans_output_file = f"{_trans_agent_dir}/{_trans_agent_id}.json"
            _trans_manifest_file = f"{_trans_agent_dir}/{_trans_agent_id}.manifest.json"

            _trans_payload = {
                "status": "completed",
                "notes": _trans_notes,
                "n_notes": _trans_n_notes,
                "duration_s": _trans_duration_s,
            }
            Path(_trans_output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(_trans_output_file, "w") as _trf:
                json.dump(_trans_payload, _trf)
                _trf.write("\n")

            # Step 1: create a REAPER track + load Vital on it
            _trans_create_snippet = (
                _REAPY_HELPER
                + f"track_name = {json.dumps(_trans_track_name)}\n"
                  f"with reapy.inside_reaper():\n"
                  f"    RPR.InsertTrackAtIndex(0, True)\n"
                  f"    track = RPR.GetTrack(0, 0)\n"
                  f"    RPR.GetSetMediaTrackInfo_String(track, 'P_NAME', track_name, True)\n"
                  f"    RPR.TrackFX_AddByName(track, 'Vital', False, 1)\n"
                  f"print(json.dumps({{'status': 'ok', 'track_idx': 0, 'track_name': track_name}}))\n"
            )
            _trans_create_cmd = _wrap_as_bash(_trans_create_snippet)
            messages.append({
                "role": "assistant",
                "content": "Creating a REAPER track to hold the transcribed MIDI before I search the wavetable library.",
            })
            messages.append(_tool_call("Bash", {"command": _trans_create_cmd}))
            _track_stdout = json.dumps({
                "status": "ok",
                "track_idx": _trans_track_idx,
                "track_name": _trans_track_name,
            }) + "\n"
            messages.append(_bash_tool_response(_track_stdout))

            # Step 2: dispatch transcription subagent (single call)
            _dispatch_prompt = (
                f"Target: {target_audio_path}. Track: {_trans_track_idx}. Write Python "
                f"(reapy → MIDI_InsertNote) that inserts the MIDI "
                f"notes on that track. After inserting, render and listen to verify "
                f"it matches the target melody."
            )
            write_agent_manifest(
                agent_id=_trans_agent_id,
                subagent_type="melody_transcription",
                output_file=_trans_output_file,
                manifest_file=_trans_manifest_file,
                prompt=_dispatch_prompt,
            )
            messages.append({
                "role": "assistant",
                "content": "Dispatching the transcription subagent to listen to the target and populate the track with MIDI notes.",
            })
            messages.append(_tool_call("Agent", {
                "subagent_type": "melody_transcription",
                "description": f"Transcribe target melody to MIDI on track {_trans_track_idx}",
                "prompt": _dispatch_prompt,
                "name": f"transcribe-{sample_id}",
            }))
            messages.append({
                "role": "tool_response",
                "content": json.dumps({
                    "agentId": _trans_agent_id,
                    "subagentType": "melody_transcription",
                    "status": "completed",
                    "outputFile": _trans_output_file,
                    "manifestFile": _trans_manifest_file,
                    "createdAt": f"build-time:{_trans_agent_id}",
                    "startedAt": f"build-time:{_trans_agent_id}",
                    "n_notes": _trans_n_notes,
                    "duration_s": _trans_duration_s,
                }, ensure_ascii=False),
            })

            # Collect the transcription subagent record for training
            _trans_mistake_rate = float(getattr(args, "transcription_mistake_rate", 0.0) or 0.0)
            trans_result = build_transcription_record_v4(
                sample_id=sample_id,
                archetype=archetype,
                target_audio_path=target_audio_path,
                source_midi_path=source_midi_path,
                output_dir=Path(f"/tmp/agents"),
                track_idx=_trans_track_idx,
                mistake_rate=_trans_mistake_rate,
                seed=int(args.seed),
            )
            if trans_result.record:
                transcription_records.append(trans_result.record)

            transcription_summary_text = (
                f"Transcription verified — {_trans_n_notes} notes on track "
                f"{_trans_track_idx}. "
            )

    # Library size check
    messages.append({
        "role": "assistant",
        "content": f"{transcription_summary_text}Checking wavetable library size.",
    })
    messages.append(_tool_call("Bash", {"command": _wrap_as_bash(build_list_wavetables_total_snippet())}))
    messages.append(_bash_tool_response(json.dumps({"total": total_named}) + "\n"))

    agent_out_dir = f"/tmp/agents/{sample_id}"

    _wt_name_to_emb = build_name_embedding_map(shortlist_data["embeddings"], index_rows)

    tuple_audio_dir = Path(args.out_dir) / "tuple_audio" / sample_id
    tuple_audio_dir.mkdir(parents=True, exist_ok=True)
    wt_lib_by_name = {wt["name"]: wt for wt in wavetable_lib if "name" in wt}

    def _build_tuple_preset(tup: list, oscs: list[int], lib: dict, base: dict) -> dict:
        preset = copy.deepcopy(base)
        for oi in oscs:
            name = tup[oi]
            if name and name in lib:
                if oi < len(preset["settings"].get("wavetables", [])):
                    preset["settings"]["wavetables"][oi] = copy.deepcopy(lib[name])
                preset["settings"][f"osc_{oi + 1}_on"] = 1.0
                preset["settings"][f"osc_{oi + 1}_level"] = 0.7
            else:
                preset["settings"][f"osc_{oi + 1}_on"] = 0.0
                preset["settings"][f"osc_{oi + 1}_level"] = 0.0
        return preset

    # Pre-render probes for search agents
    all_slice_names: list[str] = []
    for s in slice_starts:
        for i in range(s, min(s + slice_size, total_named)):
            if i in idx_to_name_full:
                all_slice_names.append(idx_to_name_full[i])
    all_slice_names = list(dict.fromkeys(all_slice_names))
    with serial_lock:
        ensure_candidate_probes_for_names(
            names=all_slice_names,
            wavetable_lib=wavetable_lib,
            selected_rows=selected_by_name,
            out_dir=Path(getattr(args, "probe_dir", "outputs/agent_sft/candidate_probes")),
            cache=candidate_audio,
            notes=notes,
        )

    _log("search loop start")
    # ---- Multi-round search loop (REAL builders, not oracle) ----
    pool: list[str] = []
    rounds_used = 0
    round_offsets_used: list[list[int]] = []
    verdicts_by_round: list[str] = []
    judge_exhausted_fallback = False
    cur_tuple: list[str | None] = [None, None, None]
    missing_character = ""
    _prev_verdict = ""  # tracks last round's verdict for pool-reset decision
    _carried_locked_slots: dict[int, str] = {}  # slots confirmed good from prior partial_match rounds

    _research_prefix = ""

    while rounds_used < max_rounds:
        rounds_used += 1
        round_offsets_used.append(list(slice_starts))

        # Pool management on re-search (round 2+):
        #   no_match     → full reset (nothing was useful)
        #   partial_match → keep pool (locked slots are good, supplement with new candidates)
        if rounds_used > 1:
            if _prev_verdict != "partial_match":
                pool = []

        # Announce round
        if rounds_used == 1:
            intro = (
                f"Library has {total_named} wavetables. Dispatching {n_agents} search "
                f"agents in parallel across slices "
                f"[{', '.join(f'{s}-{s + slice_size - 1}' for s in slice_starts)}]."
            )
        else:
            intro = (
                f"{_research_prefix}"
                f"Expanding to different library regions with {n_agents} more search agents "
                f"in parallel: "
                f"[{', '.join(f'{s}-{s + slice_size - 1}' for s in slice_starts)}]."
            )
            _research_prefix = ""
        messages.append({"role": "assistant", "content": intro})

        # Dispatch search agents via REAL builders
        round_output_files: list[str] = []
        round_agent_ids: list[str] = []
        round_agent_meta: list[tuple[int, int, str, str, str, list[str]]] = []

        # Build search agent call specs (parallel-safe — each agent
        # gets its own shard and output file; shared state is read-only).
        _search_specs: list[tuple[int, int, int, str]] = []
        for ai, start in enumerate(slice_starts):
            end = min(start + slice_size, total_named)
            if end <= start:
                continue
            agent_id = make_agent_id(sample_id, "wavetable_search", rounds_used, ai)
            _search_specs.append((ai, start, end, agent_id))

        def _run_search(spec: tuple[int, int, int, str]) -> tuple[int, str, SearchResult]:
            ai, start, end, agent_id = spec
            sr = build_search_record(
                sample_id=sample_id,
                agent_idx=ai + 1,
                archetype=archetype,
                target_audio_path=target_audio_path,
                target_preset=target_preset,
                gt_wavetable_names=gt_names_list,
                shard_start=start,
                shard_end=end,
                name_to_idx=name_to_idx_full,
                idx_to_name=idx_to_name_full,
                candidate_audio=candidate_audio,
                name_to_emb=_wt_name_to_emb,
                omni_server=args.omni_server,
                omni_model=args.omni_model,
                stage2_server=stage2_server,
                stage2_model=stage2_model,
                candidates_per_batch=int(getattr(args, "candidates_per_batch", 8)),
                shortlist_dir=Path(agent_out_dir),
                clap_threshold=0.97,
                midi_path=_trans_output_file,
            )
            return ai, agent_id, sr

        # Dispatch all search agents in parallel
        _search_results_ordered: list[tuple[int, str, SearchResult]] = []
        with ThreadPoolExecutor(max_workers=len(_search_specs)) as _search_pool:
            _search_futs = [_search_pool.submit(_run_search, sp) for sp in _search_specs]
            for fut in as_completed(_search_futs):
                _search_results_ordered.append(fut.result())
        _search_results_ordered.sort(key=lambda t: t[0])

        for ai, agent_id, search_result in _search_results_ordered:
            start, end = _search_specs[ai][1], _search_specs[ai][2]
            round_agent_ids.append(agent_id)

            if search_result.record:
                search_result.record["id"] = f"{sample_id}_r{rounds_used}_agent{ai + 1}_search"
                search_records.append(search_result.record)

            sl = search_result.shortlist

            out_dir = Path(agent_out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{agent_id}.json"
            manifest_path = out_dir / f"{agent_id}.manifest.json"
            with open(out_path, "w") as f:
                json.dump({
                    "status": "completed",
                    "agentId": agent_id,
                    "shardStart": start,
                    "shardEnd": end,
                    "shortlist": sl,
                }, f)
                f.write("\n")
            write_agent_manifest(
                agent_id=agent_id,
                subagent_type="wavetable_search",
                output_file=str(out_path),
                manifest_file=str(manifest_path),
                extra={"shardStart": start, "shardEnd": end},
            )
            round_output_files.append(str(out_path))
            round_agent_meta.append((start, end, agent_id, str(out_path), str(manifest_path), sl))

        # Emit Agent tool_calls
        for ai_idx, (start, end, agent_id, _out, _manifest, _sl) in enumerate(round_agent_meta):
            _search_prompt_parts = [f"Target: {target_audio_path}."]
            if _trans_output_file:
                _search_prompt_parts.append(f"Transcription MIDI: {_trans_output_file}.")
            _search_prompt_parts.append(
                f"Evaluate wavetables at indices {start}-{end - 1}. "
                f"Scan Vital's data directories for .vitaltable and .vital files to get names in your range, "
                f"swap each into the synth, render, and listen. "
                f"Return a JSON shortlist of 2-4 wavetable names."
            )
            messages.append(_tool_call("Agent", {
                "subagent_type": "wavetable_search",
                "description": f"Evaluate wavetables {start}-{end - 1} for target sound",
                "prompt": "\n".join(_search_prompt_parts),
                "name": f"search-r{rounds_used}-a{ai_idx + 1}",
            }))

        for _start, _end, agent_id, output_file, manifest_file, _sl in round_agent_meta:
            messages.append({
                "role": "tool_response",
                "content": json.dumps({
                    "agentId": agent_id,
                    "subagentType": "wavetable_search",
                    "status": "completed",
                    "outputFile": output_file,
                    "manifestFile": manifest_file,
                    "createdAt": f"build-time:{agent_id}",
                    "startedAt": f"build-time:{agent_id}",
                }, ensure_ascii=False),
            })

        # Read shortlists
        cat_cmd = "cat " + " ".join(round_output_files)
        messages.append({
            "role": "assistant",
            "content": f"Reading shortlists from {len(round_output_files)} search agents.",
        })
        messages.append(_tool_call("Bash", {"command": cat_cmd}))
        cat_output_lines = []
        round_shortlists: list[list[str]] = []
        for out_file in round_output_files:
            try:
                with open(out_file) as f:
                    content = f.read().strip()
                cat_output_lines.append(content)
                parsed = json.loads(content)
                round_shortlists.append(parsed.get("shortlist", []))
            except Exception:
                cat_output_lines.append("")
                round_shortlists.append([])
        messages.append(_bash_tool_response("\n".join(cat_output_lines) + "\n"))

        # Pool in shortlists
        for sl in round_shortlists:
            for name in sl:
                if name not in pool:
                    pool.append(name)

        # Ensure probes exist for pool candidates (judge needs them)
        with serial_lock:
            ensure_candidate_probes_for_names(
                names=pool,
                wavetable_lib=wavetable_lib,
                selected_rows=selected_by_name,
                out_dir=Path(getattr(args, "probe_dir", "outputs/agent_sft/candidate_probes")),
                cache=candidate_audio,
                notes=notes,
            )

        # Judge via REAL builder
        judge_result = build_judge_record(
            sample_id=sample_id,
            target_audio_path=target_audio_path,
            target_preset=target_preset,
            gt_wavetable_names=gt_names_list,
            pool=pool,
            candidate_audio=candidate_audio,
            name_to_emb=_wt_name_to_emb,
            active_oscs=active_oscs,
            omni_server=args.omni_server,
            omni_model=args.omni_model,
            stage2_server=stage2_server,
            stage2_model=stage2_model,
            judge_output_dir=Path(agent_out_dir),
        )
        if judge_result.record:
            judge_result.record["id"] = f"{sample_id}_r{rounds_used}_judge"
            judge_records.append(judge_result.record)

        judge_verdict = judge_result.verdict
        missing_character = judge_result.missing_character

        if judge_verdict == "partial_match" and judge_result.locked_slots:
            # Partial match: write locked slots by their real osc index.
            # Unfilled slots keep whatever cur_tuple had (or None).
            for osc_idx, name in judge_result.locked_slots.items():
                cur_tuple[osc_idx] = name
        elif judge_result.tuple:
            # Good verdict: positional assignment.
            for i, name in enumerate(judge_result.tuple):
                if i < len(active_oscs):
                    cur_tuple[active_oscs[i]] = name
        else:
            # No tuple from judge (no_match) — build best-available from pool
            used_in_tuple: set[str] = set()
            wts = target_preset.get("settings", {}).get("wavetables", [])
            for osc_idx in active_oscs:
                wt_name = wts[osc_idx].get("name", "") if osc_idx < len(wts) else ""
                if wt_name and wt_name in pool:
                    cur_tuple[osc_idx] = wt_name
                    used_in_tuple.add(wt_name)
                    continue
                gt_emb = _wt_name_to_emb.get(wt_name)
                if gt_emb is not None:
                    candidates = [n for n in pool if n not in used_in_tuple and n in _wt_name_to_emb]
                    if candidates:
                        best = max(candidates, key=lambda n: float(_wt_name_to_emb[n] @ gt_emb))
                        cur_tuple[osc_idx] = best
                        used_in_tuple.add(best)
                        continue
                for n in pool:
                    if n not in used_in_tuple:
                        cur_tuple[osc_idx] = n
                        used_in_tuple.add(n)
                        break

        cur_active_names = [cur_tuple[oi] for oi in active_oscs if cur_tuple[oi]]
        n_osc_slots = len(active_oscs)

        verdicts_by_round.append(judge_verdict)
        _prev_verdict = judge_verdict

        # Read judge output file path
        judge_agent_id = make_agent_id(sample_id, "wavetable_judge", rounds_used)
        judge_output_file = Path(agent_out_dir) / f"{judge_agent_id}.json"
        judge_manifest_file = Path(agent_out_dir) / f"{judge_agent_id}.manifest.json"

        # Dispatch judge in main conversation
        pool_str = ", ".join(f"'{n}'" for n in pool[:6])
        if len(pool) > 6:
            pool_str += f", ...+{len(pool) - 6} more"
        if _carried_locked_slots:
            _locked_narration = (
                " Previously locked: "
                + ", ".join(f"osc {oi+1}='{n}'" for oi, n in sorted(_carried_locked_slots.items()))
                + ". Judge should keep these and evaluate new candidates for the remaining slots."
            )
        else:
            _locked_narration = ""
        messages.append({
            "role": "assistant",
            "content": (
                f"Pool has {len(pool)} candidates across {len(round_agent_meta)} slices: "
                f"[{pool_str}]. Dispatching judge agent to audition the combined pool and "
                f"select the best oscillator combination (up to 3).{_locked_narration}"
            ),
        })
        if _carried_locked_slots:
            _locked_prompt_section = (
                f"\nPreviously confirmed (locked) selections from prior round: "
                + json.dumps({str(k): v for k, v in _carried_locked_slots.items()})
                + ". Keep these locked — only evaluate new candidates for the unfilled slots."
            )
        else:
            _locked_prompt_section = ""
        _judge_dispatch_prompt = (
            f"Target: {target_audio_path}.\n"
            f"Pool candidates from search agents: {json.dumps(pool)}.\n"
            f"The target may use up to 3 active oscillators. Swap each candidate "
            f"wavetable into the synth via chunk manipulation, render, and listen "
            f"alongside the target, then select the candidates (1 to 3) that "
            f"together best capture the target. Return your selection as JSON with "
            f"keys: tuple (list of chosen names), n_osc_slots (how many you chose), "
            f"reasoning."
            f"{_locked_prompt_section}"
        )
        # Write judge output file on disk for the cat (always overwrite —
        # stale files from prior builds would show the wrong verdict).
        judge_output_file.parent.mkdir(parents=True, exist_ok=True)
        if judge_verdict == "good":
            _payload_tuple = cur_active_names
        elif judge_verdict == "partial_match" and judge_result.locked_slots:
            _payload_tuple = list(judge_result.locked_slots.values())
        else:
            _payload_tuple = None
        judge_output_payload: dict = {
            "status": "completed",
            "agentId": judge_agent_id,
            "verdict": judge_verdict,
            "missing_character": missing_character,
            "tuple": _payload_tuple,
            "n_osc_slots": n_osc_slots,
            "reasoning": judge_result.reasoning,
        }
        if judge_verdict == "partial_match" and judge_result.locked_slots:
            judge_output_payload["locked_slots"] = {
                str(k): v for k, v in judge_result.locked_slots.items()
            }
            judge_output_payload["unfilled_oscs"] = judge_result.unfilled_oscs or []
        with open(judge_output_file, "w") as jf:
            json.dump(judge_output_payload, jf)
            jf.write("\n")

        write_agent_manifest(
            agent_id=judge_agent_id,
            subagent_type="wavetable_judge",
            output_file=str(judge_output_file),
            manifest_file=str(judge_manifest_file),
            prompt=_judge_dispatch_prompt,
        )
        messages.append(_tool_call("Agent", {
            "subagent_type": "wavetable_judge",
            "description": f"Select best oscillator combination (up to 3) from pool of {len(pool)} candidates",
            "prompt": _judge_dispatch_prompt,
            "name": f"judge-{rounds_used}",
        }))
        messages.append({
            "role": "tool_response",
            "content": json.dumps({
                "agentId": judge_agent_id,
                "subagentType": "wavetable_judge",
                "status": "completed",
                "outputFile": str(judge_output_file),
                "manifestFile": str(judge_manifest_file),
                "createdAt": f"build-time:{judge_agent_id}",
                "startedAt": f"build-time:{judge_agent_id}",
            }, ensure_ascii=False),
        })

        # Read judge verdict
        messages.append({
            "role": "assistant",
            "content": "Reading judge's verdict and selection.",
        })
        messages.append(_tool_call("Bash", {"command": f"cat {judge_output_file}"}))
        with open(judge_output_file) as jf:
            judge_content = jf.read().strip()
        messages.append(_bash_tool_response(judge_content + "\n"))

        # Branch on verdict
        if judge_verdict in ("no_match", "partial_match"):
            if rounds_used >= max_rounds:
                judge_exhausted_fallback = True
                break
            if judge_verdict == "partial_match":
                locked = judge_result.locked_slots or {}
                _carried_locked_slots.update(locked)
                locked_str = ", ".join(
                    f"osc {oi+1}='{n}'" for oi, n in sorted(_carried_locked_slots.items())
                )
                n_unfilled = len(judge_result.unfilled_oscs or [])
                _research_prefix = (
                    f"The judge confirmed [{locked_str}] for "
                    f"{len(_carried_locked_slots)} of {n_osc_slots} slots, but {n_unfilled} "
                    f"slot{'s' if n_unfilled > 1 else ''} still "
                    f"{'need' if n_unfilled > 1 else 'needs'} a wavetable with "
                    f"the {missing_character} of the target. Keeping locked "
                    f"selections and searching for the remaining slot{'s' if n_unfilled > 1 else ''}. "
                )
            else:
                _carried_locked_slots.clear()
                _research_prefix = (
                    f"The judge reports the pool of {len(pool)} candidates doesn't contain "
                    f"any wavetable with the {missing_character} of the target. "
                    f"Expanding search to unexplored library regions. "
                )
            base_offset = (base_offset + stride // 2) % stride
            slice_starts = _compute_slices(base_offset)

            # Pre-render probes for next round
            next_slice_names: list[str] = []
            for s in slice_starts:
                for i in range(s, min(s + slice_size, total_named)):
                    if i in idx_to_name_full:
                        next_slice_names.append(idx_to_name_full[i])
            next_slice_names = list(dict.fromkeys(next_slice_names))
            with serial_lock:
                ensure_candidate_probes_for_names(
                    names=next_slice_names,
                    wavetable_lib=wavetable_lib,
                    selected_rows=selected_by_name,
                    out_dir=Path(getattr(args, "probe_dir", "outputs/agent_sft/candidate_probes")),
                    cache=candidate_audio,
                    notes=notes,
                )
            continue

        # Good verdict — render tuple, listen, break
        tuple_wav = tuple_audio_dir / f"tuple_r{rounds_used}.wav"
        osc_names = {oi: cur_tuple[oi] for oi in active_oscs if cur_tuple[oi]}
        render_cmd = _wrap_as_bash(build_render_tuple_snippet(
            osc_names=osc_names, out_path=str(tuple_wav),
            midi_path=_trans_output_file,
        ))
        render_cumulative_audio(
            _build_tuple_preset(cur_tuple, active_oscs, wt_lib_by_name, init_preset),
            notes, tuple_wav,
        )

        tuple_names_str = ", ".join(f"'{n}'" for n in cur_active_names)
        messages.append({
            "role": "assistant",
            "content": (
                f"Judge confirmed pool is sufficient and selected [{tuple_names_str}]. "
                f"Rendering the tuple as a final pre-apply sanity check."
            ),
        })
        messages.append(_tool_call("Bash", {"command": render_cmd}))
        audio_assets.append(str(tuple_wav))
        _tuple_stdout = json.dumps({
            "status": "ok", "out": str(tuple_wav), "wavetables": cur_active_names,
        }) + "\n"
        _emit_listen_sequence(
            messages, audio_assets, tuple_wav,
            probe_stdout=_tuple_stdout,
            listen_text="Tuple rendered. Listening.",
        )
        break

    # Exhausted fallback
    if judge_exhausted_fallback:
        tuple_wav = tuple_audio_dir / f"tuple_r{rounds_used}_fallback.wav"
        osc_names = {oi: cur_tuple[oi] for oi in active_oscs if cur_tuple[oi]}
        render_cmd = _wrap_as_bash(build_render_tuple_snippet(
            osc_names=osc_names, out_path=str(tuple_wav),
            midi_path=_trans_output_file,
        ))
        render_cumulative_audio(
            _build_tuple_preset(cur_tuple, active_oscs, wt_lib_by_name, init_preset),
            notes, tuple_wav,
        )
        cur_active_names = [cur_tuple[oi] for oi in active_oscs if cur_tuple[oi]]
        tuple_names_str = ", ".join(f"'{n}'" for n in cur_active_names)
        _last_verdict = verdicts_by_round[-1] if verdicts_by_round else "no_match"
        if _last_verdict == "partial_match":
            _exhaust_msg = (
                f"Search budget exhausted after {max_rounds} rounds; the judge found "
                f"partial matches but couldn't fill all slots. Rendering the best-available "
                f"combination [{tuple_names_str}] to make progress before parameter tuning."
            )
        else:
            _exhaust_msg = (
                f"Search budget exhausted after {max_rounds} rounds; the judge still "
                f"reports no_match on the final pool. Rendering the best-available "
                f"combination [{tuple_names_str}] to make progress before parameter tuning."
            )
        messages.append({
            "role": "assistant",
            "content": _exhaust_msg,
        })
        messages.append(_tool_call("Bash", {"command": render_cmd}))
        audio_assets.append(str(tuple_wav))
        _fb_tuple_stdout = json.dumps({
            "status": "ok", "out": str(tuple_wav), "wavetables": cur_active_names,
        }) + "\n"
        _emit_listen_sequence(
            messages, audio_assets, tuple_wav,
            probe_stdout=_fb_tuple_stdout,
            listen_text="Fallback tuple rendered. Listening.",
        )

    gt_tuple = cur_tuple
    apply_names = [gt_tuple[oi] for oi in active_oscs if gt_tuple[oi]]
    osc_assignments = ", ".join(
        f"oscillator {oi + 1} = '{gt_tuple[oi]}'" for oi in active_oscs if gt_tuple[oi]
    )
    final_verdict = verdicts_by_round[-1] if verdicts_by_round else "good"
    if final_verdict == "good" and not judge_exhausted_fallback:
        selection_text = f"This tuple matches the target well. Applying: {osc_assignments}."
    else:
        selection_text = (
            f"Search budget exhausted; applying best-available combination: "
            f"{osc_assignments}."
        )

    apply_assignments = repr(
        [(oi, gt_tuple[oi]) for oi in active_oscs if gt_tuple[oi]]
    )
    target_mods_literal = repr(target_modulations)
    apply_snippet = (
        _REAPY_HELPER
        + _BUILD_CHUNK_HELPER
        + "import base64\n"
        + _WT_DISCOVER_SNIPPET
        + "name_to_wt = {wt['name']: wt for wt in lib if 'name' in wt}\n"
        'preset = json.load(open("skills/vital/data/init_preset.json"))\n'
        f"for osc_idx, wt_name in {apply_assignments}:\n"
        "    if wt_name in name_to_wt:\n"
        "        preset['settings']['wavetables'][osc_idx] = name_to_wt[wt_name]\n"
        f"preset['settings']['modulations'] = {target_mods_literal}\n"
        "chunk = build_vital_chunk(preset)\n"
        "encoded = base64.b64encode(chunk).decode('ascii')\n"
        "with reapy.inside_reaper():\n"
        "    track = RPR.GetTrack(0, 0)\n"
        "    if not RPR.TrackFX_SetNamedConfigParm(track, 0, 'vst3_chunk', encoded):\n"
        "        RPR.TrackFX_SetNamedConfigParm(track, 0, 'vst_chunk', encoded)\n"
        f"print(json.dumps({{'status': 'ok', 'applied': {json.dumps(apply_names)}}}))"
    )
    messages.append({"role": "assistant", "content": selection_text})
    messages.append(_tool_call("Bash", {"command": _wrap_as_bash(apply_snippet)}))
    _apply_stdout = json.dumps({"status": "ok", "applied": apply_names}) + "\n"
    messages.append(_bash_tool_response(_apply_stdout))

    # ---- DIAGNOSIS + SUBSYSTEM BATCHES (identical to v3) ----
    subsystem_truth_map = build_diagnosis_subsystem_truth(target_preset, init_preset)
    subsystems_truth = [lbl for lbl, _ in SUBSYSTEM_ORDER if lbl in subsystem_truth_map]
    diff_summary = format_subsystem_diff_summary(subsystem_truth_map)

    if args.omni_server:
        stage1_obs = omni_stage1_diagnose(
            str(target_audio_path), target_preset,
            archetype, args.omni_server, args.omni_model,
        )
    else:
        stage1_obs = "Target differs from the starting preset in several subsystems."
    _dt = _time.monotonic()
    diagnosis_text = stage2_diagnosis(
        stage1_obs, diff_summary, subsystems_truth,
        archetype, stage2_server, stage2_model,
    )
    _log(f"diagnosis omni {_time.monotonic()-_dt:.1f}s")

    _diagnosis_text = diagnosis_text

    batches = build_batches_from_diff(target_preset, init_preset)
    batches = batches[:int(args.max_batches)]
    path_complete = True

    cumulative = copy.deepcopy(init_preset)
    for key in ("wavetables", "sample", "lfos", "modulations"):
        if key in target_preset.get("settings", {}):
            cumulative["settings"][key] = copy.deepcopy(target_preset["settings"][key])

    mistake_rng = random.Random(int(args.seed) + sid_seed)
    _per_param_rate = getattr(args, "per_param_mistake_rate", None)
    if _per_param_rate is None:
        _per_param_rate = getattr(args, "mistake_rate", 0.10)
    all_injected_mistakes = inject_mistakes(
        batches, mistake_rng,
        per_param_rate=_per_param_rate,
        init_preset_settings=init_preset.get("settings", {}),
    )
    total_correction_turns = 0

    batch_labels: list[dict] = []
    prior_checks: list[str] = []
    pending_check: str | None = None
    last_batch_audio: Path | None = None
    cumulative_native_overrides: dict[str, float] = {}
    current_reaper_values: dict[int, float] = {}

    batch_audio_dir = Path(args.out_dir) / "batch_audio" / sample_id
    batch_audio_dir.mkdir(parents=True, exist_ok=True)
    _log(f"batch loop start — {len(batches)} batches")

    for bi, b in enumerate(batches):
        batch_before_values: dict[str, float] = {
            name: float(cumulative["settings"].get(name, 0.0) or 0.0)
            for name in set(b.params.keys()) | set(b.params_applied.keys())
        }
        for name, norm in b.params_applied.items():
            native = _denormalize(name, norm)
            cumulative["settings"][name] = native
            cumulative_native_overrides[name] = native

        batch_wav = batch_audio_dir / f"batch_{bi}_{b.subsystem}.wav"
        _bt = _time.monotonic()
        render_cumulative_audio(cumulative, notes, batch_wav)
        _log(f"  batch {bi}/{len(batches)-1} ({b.subsystem}) render {_time.monotonic()-_bt:.1f}s")

        with serial_lock:
            try:
                clap_after = float(embedder.cosine_paths(batch_wav, target_audio_path))
            except Exception:
                clap_after = None

        b.audio_wav = batch_wav
        params_by_idx = {
            _JSON_KEY_TO_REAPER[n]["idx"]: float(v)
            for n, v in b.params_applied.items()
            if n in _JSON_KEY_TO_REAPER
        }
        action_snippet = build_batch_action_snippet(params_by_idx)

        search_queries = _batch_search_queries(
            b.subsystem, list(b.params_applied.keys()), _JSON_KEY_TO_REAPER,
        )

        query_label = " and ".join(search_queries) if len(search_queries) > 1 else search_queries[0]
        intro = f"Searching for {query_label} parameters."
        if bi == 0 and _diagnosis_text:
            intro = f"{_diagnosis_text}\n\n{intro}"
            _diagnosis_text = None
        elif pending_check:
            intro = f"{pending_check}\n\n{intro}"
            pending_check = None
        messages.append({"role": "assistant", "content": intro})

        for sq in search_queries:
            search_snippet = _wrap_as_bash(build_param_search_snippet(sq))
            search_results = simulate_param_search(
                sq, _REAPER_PARAM_DUMP,
                value_overrides=current_reaper_values,
            )
            search_stdout = json.dumps(
                {"query": sq, "count": len(search_results), "params": search_results},
                indent=2, ensure_ascii=False,
            ) + "\n"
            messages.append(_tool_call("Bash", {"command": search_snippet}))
            messages.append(_bash_tool_response(search_stdout))

        messages.append({"role": "assistant", "content": f"Applying {b.subsystem} changes."})
        messages.append(_tool_call("Bash", {"command": action_snippet}))
        _action_stdout = json.dumps({"status": "ok", "applied": len(params_by_idx)}) + "\n"
        messages.append(_bash_tool_response(_action_stdout))

        for name, norm in b.params_applied.items():
            if name in _JSON_KEY_TO_REAPER:
                current_reaper_values[_JSON_KEY_TO_REAPER[name]["idx"]] = float(norm)

        audio_assets.append(str(batch_wav))
        messages.append({"role": "assistant", "content": f"Listening after {b.subsystem} batch."})
        _batch_render_cmd = _wrap_as_bash(build_reaper_render_snippet(out_path=str(batch_wav)))
        messages.append(_tool_call("Bash", {"command": _batch_render_cmd}))
        _emit_listen_sequence(
            messages, audio_assets, batch_wav,
            listen_text=f"Reading {b.subsystem} batch audio.",
        )
        last_batch_audio = batch_wav

        gap = _step_remaining_gap(target_preset, {"cumulative_preset": cumulative})
        is_last = bi == len(batches) - 1

        if args.omni_server:
            plan_bullet = _extract_plan_bullet(diagnosis_text, b.subsystem)
            _mistake_map = {m.param: m for m in (b.mistakes or [])}
            _all_delta_names = sorted(set(b.params.keys()) | set(b.params_applied.keys()))
            param_deltas: list[tuple] = []
            for _dn in _all_delta_names:
                _bv = batch_before_values.get(_dn, 0.0)
                _av = float(cumulative["settings"].get(_dn, 0.0) or 0.0)
                _dm = _mistake_map.get(_dn)
                _tag = _dm.kind if _dm and _dm.kind in ("omission", "spurious") else None
                param_deltas.append((_dn, _bv, _av, _tag))
            _ot = _time.monotonic()
            check_sentence = stage2_batch_check(
                subsystem=b.subsystem,
                plan_bullet=plan_bullet,
                param_deltas=param_deltas,
                prior_checks=prior_checks,
                archetype=archetype,
                stage2_server=stage2_server,
                stage2_model=stage2_model,
                is_final=is_last and not b.mistakes,
                n_params_applied=len(b.params),
            )
            _log(f"  batch {bi}/{len(batches)-1} ({b.subsystem}) omni {_time.monotonic()-_ot:.1f}s")
        else:
            check_sentence = f"{b.subsystem.capitalize()} edits applied consistent with the plan."
        prior_checks.append(check_sentence)
        pending_check = check_sentence

        batch_labels.append({
            "index": bi,
            "subsystem": b.subsystem,
            "param_names": sorted(b.params.keys()),
            "clap_score_after_batch": clap_after,
            "is_correction": False,
        })

        # ---- ITERATIVE CORRECTION (after a batch with mistakes) ----
        if b.mistakes:
            unfixed = list(b.mistakes)
            correction_turn = 0
            max_corr = getattr(args, "max_correction_turns", 3)

            while unfixed and correction_turn < max_corr:
                correction_turn += 1
                total_correction_turns += 1

                unfixed.sort(key=lambda m: m.magnitude, reverse=True)
                if len(unfixed) >= 3 and correction_turn == 1:
                    n_fix = 1
                elif len(unfixed) >= 2:
                    n_fix = mistake_rng.choice([1, 2])
                else:
                    n_fix = 1
                fixing_now = [unfixed.pop(0) for _ in range(n_fix)]

                for m in fixing_now:
                    corr_native = _denormalize(m.param, m.true_value)
                    cumulative["settings"][m.param] = corr_native
                    cumulative_native_overrides[m.param] = corr_native

                corr_wav = batch_audio_dir / f"batch_{bi}_correction_{correction_turn}.wav"
                render_cumulative_audio(cumulative, notes, corr_wav)
                with serial_lock:
                    try:
                        corr_clap = float(embedder.cosine_paths(corr_wav, target_audio_path))
                    except Exception:
                        corr_clap = None

                corr_prefix = f"{pending_check}\n\n" if pending_check else ""
                pending_check = None
                if args.omni_server:
                    corr_intro = corr_prefix + stage2_correction_intro(
                        subsystem=b.subsystem,
                        mistakes_being_fixed=fixing_now,
                        remaining_mistakes=unfixed,
                        archetype=archetype,
                        stage2_server=stage2_server,
                        stage2_model=stage2_model,
                    )
                else:
                    fix_desc = ", ".join(_json_key_to_display(m.param) for m in fixing_now)
                    corr_intro = f"{corr_prefix}Noticed issues in {b.subsystem} — correcting {fix_desc}."
                messages.append({"role": "assistant", "content": corr_intro})

                corr_by_idx: dict[int, float] = {}
                for m in fixing_now:
                    corr_idx = _JSON_KEY_TO_REAPER.get(m.param, {}).get("idx")
                    if corr_idx is not None:
                        corr_by_idx[corr_idx] = float(m.true_value)
                        current_reaper_values[corr_idx] = float(m.true_value)

                if corr_by_idx:
                    messages.append(_tool_call("Bash", {"command": build_batch_action_snippet(corr_by_idx)}))
                    _corr_stdout = json.dumps({"status": "ok", "applied": len(corr_by_idx)}) + "\n"
                    messages.append(_bash_tool_response(_corr_stdout))
                else:
                    messages.append(_tool_call("Bash", {"command": "echo 'no matching REAPER param'"}))
                    messages.append(_bash_tool_response("no matching REAPER param\n"))

                audio_assets.append(str(corr_wav))
                messages.append({"role": "assistant", "content": "Listening to the corrected preset."})
                _corr_render_cmd = _wrap_as_bash(build_reaper_render_snippet(out_path=str(corr_wav)))
                messages.append(_tool_call("Bash", {"command": _corr_render_cmd}))
                _emit_listen_sequence(
                    messages, audio_assets, corr_wav,
                    listen_text="Listening to the corrected preset.",
                )
                last_batch_audio = corr_wav

                if unfixed:
                    pending_check = f"Improved {b.subsystem}, but something still sounds off — listening again."
                else:
                    pending_check = f"The {b.subsystem} region now sits back in line with the plan."

                batch_labels.append({
                    "index": len(batch_labels),
                    "subsystem": "correction",
                    "param_names": [m.param for m in fixing_now],
                    "correction_turn": correction_turn,
                    "mistakes_fixed": [{"param": m.param, "kind": m.kind, "magnitude": round(m.magnitude, 4)} for m in fixing_now],
                    "mistakes_remaining": len(unfixed),
                    "clap_score_after_batch": corr_clap,
                    "is_correction": True,
                })

    if _diagnosis_text:
        messages.append({"role": "assistant", "content": _diagnosis_text})

    # ---- FINAL ASSESSMENT ----
    final_gap = _step_remaining_gap(target_preset, {"cumulative_preset": cumulative})
    final_audio_wav = last_batch_audio

    if args.omni_server and final_audio_wav:
        try:
            verdict_obs = omni_stage1_verdict(
                str(target_audio_path), str(final_audio_wav),
                archetype, args.omni_server, args.omni_model,
            )
        except Exception:
            verdict_obs = "The recreation is close to the target but some timbral details still differ."
    else:
        verdict_obs = "The recreation captures the target's core character."

    fully_converged = final_gap is not None and final_gap["n_remaining"] == 0
    residual_delta_summary = summarize_residual_delta_perceptual(target_preset, cumulative)

    _vt = _time.monotonic()
    verdict_text = stage2_verdict(
        perceptual_obs=verdict_obs,
        residual_delta_summary=residual_delta_summary,
        path_complete=fully_converged,
        archetype=archetype,
        stage2_server=stage2_server,
        stage2_model=stage2_model,
    )
    _log(f"verdict omni {_time.monotonic()-_vt:.1f}s")
    if pending_check:
        verdict_text = f"{pending_check}\n\n{verdict_text}"
    messages.append({"role": "assistant", "content": verdict_text})

    # ---- Record assembly ----
    diagnosis_subs_mentioned = extract_diagnosis_subsystems_mentioned(diagnosis_text)
    mistake_caught = bool(all_injected_mistakes)

    record = {
        "id": sample_id,
        "task_type": "main",
        "tools": _V3_TOOL_SPECS,
        "messages": messages,
        "audios": audio_assets,
        "assets": {
            "target_audio": str(target_audio_path),
            "current_audio": str(default_audio_path),
            "candidate_audio": [],
            "selected_candidates": [],
            "selected_tuples": [],
        },
        "labels": {
            "gt_wavetable_names": gt_names_list,
            "applied_wavetables": [gt_tuple[oi] for oi in active_oscs if gt_tuple[oi]],
            "search_pool_size": len(pool),
            "search_rounds_used": rounds_used,
            "search_slice_starts_per_round": round_offsets_used,
            "search_judge_verdicts": verdicts_by_round,
            "search_final_verdict": verdicts_by_round[-1] if verdicts_by_round else None,
            "search_rounds_exhausted_on_no_match": judge_exhausted_fallback,
        },
        "meta": {
            "pipeline_version": "v4_unified",
            "pipeline_version_notes": (
                "Unified top-down build: real search/judge/transcription builder calls "
                "inline. No oracle forcing — pool composition and judge verdicts come "
                "from actual CLAP thresholding and Omni audition."
            ),
            "sample_id": sample_id,
            "archetype": archetype,
            "start_type": entry.get("start_type", "init"),
            "agent": "main",
            "num_agents": int(args.num_agents),
            "pool_top_k": int(args.pool_top_k),
            "max_batches": int(args.max_batches),
            "per_param_mistake_rate": float(_per_param_rate),
            "commentary_mode": "two_stage",
            "path_complete": path_complete,
            "n_remaining": final_gap["n_remaining"] if final_gap else 0,
            "batch_labels": batch_labels,
            "diagnosis_subsystems_mentioned": diagnosis_subs_mentioned,
            "diagnosis_subsystems_truth": subsystems_truth,
            "injected_mistakes": [
                {"param": m.param, "kind": m.kind, "wrong_value": round(m.wrong_value, 4),
                 "true_value": round(m.true_value, 4), "magnitude": round(m.magnitude, 4)}
                for m in all_injected_mistakes
            ],
            "total_correction_turns": total_correction_turns,
            "mistake_caught": mistake_caught,
            "transcription_output_file": _trans_output_file,
            "random_init": use_random_init,
        },
    }

    assert_valid_ms_swift_multiturn_record(record)
    return record, search_records, judge_records, transcription_records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Unified top-down SFT pipeline v4. Builds all 4 agent JSONL files "
            "in a single pass with real subagent calls (no oracle simulation)."
        ),
    )
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--index-npy", type=Path, default=Path("outputs/wt_retrieval_baseline/wt_index.npz"))
    ap.add_argument("--index-meta", type=Path, default=Path("outputs/wt_retrieval_baseline/wt_index_meta.json"))
    ap.add_argument("--wavetable-lib", type=Path, default=Path("data/wavetable_lib.json"))
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Output directory. JSONL files named <agent>_final<N>_<suffix>.jsonl.")
    ap.add_argument("--suffix", default="v31", help="Version suffix for output filenames.")
    ap.add_argument("--max-samples", type=int, default=256)
    ap.add_argument("--max-batches", type=int, default=16)
    ap.add_argument("--pool-top-k", type=int, default=48)
    ap.add_argument("--num-agents", type=int, default=4)
    ap.add_argument("--candidates-per-slice", type=int, default=48)
    ap.add_argument("--candidates-per-batch", type=int, default=8)
    ap.add_argument("--max-search-rounds", type=int, default=3)
    ap.add_argument("--force-research-rate", type=float, default=0.20,
                    help="Reduced from 0.30 — real CLAP thresholding causes natural misses.")
    ap.add_argument("--no-audio-rate", type=float, default=0.05)
    ap.add_argument("--probe-dir", type=Path, default=Path("outputs/agent_sft/candidate_probes"))
    ap.add_argument("--per-param-mistake-rate", type=float, default=0.10,
        help="Independent per-param mistake probability (default 0.10).")
    ap.add_argument("--mistake-rate", type=float, default=None,
        help="Deprecated alias for --per-param-mistake-rate.")
    ap.add_argument("--max-correction-turns", type=int, default=3,
        help="Max correction iterations per mistaken batch (default 3).")
    ap.add_argument("--transcription-mistake-rate", type=float, default=0.15)
    ap.add_argument("--random-init-rate", type=float, default=0.0,
        help="Fraction of samples starting from a random same-archetype preset instead of factory default.")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--clap-device", default="cuda:0")
    ap.add_argument("--omni-server", default="")
    ap.add_argument("--omni-model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--stage2-server", default="")
    ap.add_argument("--stage2-model", default="")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if args.mistake_rate is not None:
        import warnings
        warnings.warn("--mistake-rate is deprecated, use --per-param-mistake-rate", DeprecationWarning)
        if args.per_param_mistake_rate == 0.10:
            args.per_param_mistake_rate = args.mistake_rate

    stage2_server = args.stage2_server or args.omni_server
    stage2_model = args.stage2_model or args.omni_model

    if args.omni_server:
        _check_server_reachable(args.omni_server, "Omni")
        if stage2_server and stage2_server != args.omni_server:
            _check_server_reachable(stage2_server, "Stage2")

    import time as _wall_time
    _wall_t0 = _wall_time.monotonic()

    entries = load_manifest_entries(Path(args.manifest), max_samples=args.max_samples)
    index_rows = load_index_rows(args.index_meta)
    selected_by_name = select_probe_rows_by_name(index_rows)
    wavetable_lib = load_wavetable_lib(args.wavetable_lib)
    embedder = ClapEmbedder.create(args.clap_device)
    shortlist_data = build_clap_shortlist_data(args.index_npy, index_rows)

    _notes = make_probe_notes("lead", clip_duration_s=10.0)

    # Pre-populate CLAP embedding cache
    clap_paths: list[Path] = []
    for e in entries:
        for k in ("gt_wav", "gt_probe_wav", "default_wav"):
            if e.get(k):
                clap_paths.append(Path(e[k]))
    probe_dir = Path(args.probe_dir)
    if probe_dir.exists():
        for pp in sorted(probe_dir.glob("*.wav")):
            clap_paths.append(pp)
    clap_paths = list(dict.fromkeys(clap_paths))
    print(f"Pre-computing CLAP embeddings for {len(clap_paths)} audio files...", flush=True)
    for p in clap_paths:
        if p.exists():
            try:
                embedder.embed_audio_path(p)
            except Exception as exc:
                print(f"  WARNING: CLAP embed failed for {p.name}: {exc}")
    print(f"CLAP pre-computation done ({len(embedder._cache)} cached).", flush=True)
    if args.clap_device != "cpu":
        try:
            embedder.model = embedder.model.to("cpu")
            embedder.device = "cpu"
            print("CLAP moved to CPU for worker-thread safety.", flush=True)
        except Exception as exc:
            print(f"WARNING: could not move CLAP to CPU: {exc}")

    candidate_audio: dict[str, Path] = {}
    serial_lock = threading.Lock()

    # Accumulate all records
    all_main: list[dict] = []
    all_search: list[dict] = []
    all_judge: list[dict] = []
    all_trans: list[dict] = []

    def _process(entry: dict) -> tuple[dict | None, list[dict], list[dict], list[dict]]:
        return build_record(
            entry=entry,
            args=args,
            embedder=embedder,
            shortlist_data=shortlist_data,
            selected_by_name=selected_by_name,
            wavetable_lib=wavetable_lib,
            index_rows=index_rows,
            candidate_audio=candidate_audio,
            stage2_server=stage2_server,
            stage2_model=stage2_model,
            serial_lock=serial_lock,
            notes=_notes,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    records_by_idx: dict[int, tuple] = {}

    def _record(i: int, result: tuple) -> None:
        main_rec, srecs, jrecs, trecs = result
        if main_rec is None:
            return
        with write_lock:
            records_by_idx[i] = result
            sid = main_rec["meta"]["sample_id"]
            print(
                f"[{len(records_by_idx)}/{len(entries)}] {sid} OK "
                f"(search={len(srecs)}, judge={len(jrecs)}, trans={len(trecs)})",
                flush=True,
            )

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool_exec:
            futs = {pool_exec.submit(_process, e): i for i, e in enumerate(entries)}
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    _record(i, fut.result())
                except Exception as exc:
                    import traceback
                    sid = entries[i].get("sample_id", f"entry_{i}")
                    print(f"WARNING: {sid} failed: {exc}")
                    traceback.print_exc()
    else:
        for i, e in enumerate(entries):
            try:
                _record(i, _process(e))
            except Exception as exc:
                import traceback
                print(f"WARNING: {e.get('sample_id','?')} failed: {exc}")
                traceback.print_exc()

    # Flatten in order
    for i in sorted(records_by_idx):
        main_rec, srecs, jrecs, trecs = records_by_idx[i]
        if main_rec:
            all_main.append(main_rec)
        all_search.extend(srecs)
        all_judge.extend(jrecs)
        all_trans.extend(trecs)

    # Write 4 JSONL files
    n = args.max_samples
    s = args.suffix
    for name, records in [
        (f"main_final{n}_{s}.jsonl", all_main),
        (f"search_final{n}_{s}.jsonl", all_search),
        (f"judge_final{n}_{s}.jsonl", all_judge),
        (f"transcription_final{n}_{s}.jsonl", all_trans),
    ]:
        out_path = args.out_dir / name
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(records):>4} records to {out_path}", flush=True)

    _wall_elapsed = _wall_time.monotonic() - _wall_t0
    _n_ok = len(all_main)
    print(flush=True)
    print(f"=== Timing summary ===", flush=True)
    print(f"  wall clock:      {_wall_elapsed:>8.1f}s  ({_wall_elapsed/60:.1f} min)", flush=True)
    print(f"  records built:   {_n_ok:>8}", flush=True)
    if _n_ok:
        print(f"  per rollout:     {_wall_elapsed/_n_ok:>8.1f}s", flush=True)
        print(f"  throughput:      {_n_ok/_wall_elapsed*3600:>8.1f} rollouts/hr  (at {args.workers} workers)", flush=True)
    print(f"  LLM calls:       {llm_post_stats.summary()}", flush=True)


if __name__ == "__main__":
    main()
