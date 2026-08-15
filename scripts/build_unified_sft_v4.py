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
    assert_valid_oc_record as assert_valid_ms_swift_multiturn_record,
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
    format_param_search_output,
    simulate_param_search,
    write_agent_manifest,
    oc_compat_bash_response as _bash_tool_response,
    _BUILD_CHUNK_HELPER,
    _READ_CHUNK_HELPER,
    oc_compat_emit_listen_sequence as _emit_listen_sequence,
    oc_compat_read_response_audio as _read_tool_response_audio,
    _REAPY_HELPER,
    oc_compat_tool_call as _tool_call,
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
from scripts.opencode_contract import TOOLS as _OC_TOOLS
_OC_TOOLS_JSON = json.dumps(_OC_TOOLS, ensure_ascii=False)

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
from maestro.reaper import dawfarm as _dawfarm
from scripts.agent_sft_common import DawFarmRolloutCtx

import numpy as np

from scripts.preset_perceptual_summary import summarize_residual_delta_perceptual


# ---------------------------------------------------------------------------
# User steer catalog — perceptual directions for post-verdict tweaks
# ---------------------------------------------------------------------------

STEER_CATALOG = [
    {"prompts": ["Make it darker", "Can you darken the tone?", "It's too bright, tone it down"],
     "response": "Darkening the tone by pulling the filter down and reducing high-end presence.",
     "params": {"filter_1_cutoff": ("subtract", 15.0), "eq_high_gain": ("subtract", 2.0)}, "tags": ["tone"]},
    {"prompts": ["Make it brighter", "Open it up more", "I want more presence"],
     "response": "Opening up the filter and boosting the highs for more brightness.",
     "params": {"filter_1_cutoff": ("add", 15.0), "eq_high_gain": ("add", 2.0)}, "tags": ["tone"]},
    {"prompts": ["Add more reverb", "Make it more spacious", "I want it to sound like it's in a big room"],
     "response": "Adding space with more reverb and a longer decay tail.",
     "params": {"reverb_dry_wet": ("add", 0.2), "reverb_decay_time": ("add", 0.15), "reverb_on": ("set", 1.0)}, "tags": ["fx"]},
    {"prompts": ["Make the attack snappier", "It needs more punch", "Tighten up the attack"],
     "response": "Tightening the attack for a snappier transient.",
     "params": {"env_1_attack": ("multiply", 0.3)}, "tags": ["envelope"]},
    {"prompts": ["Soften the attack", "Make it more pad-like", "I want a slower fade-in"],
     "response": "Softening the attack for a gentler fade-in.",
     "params": {"env_1_attack": ("multiply", 2.5)}, "tags": ["envelope"]},
    {"prompts": ["Widen the stereo", "Make it wider", "Spread it out more"],
     "response": "Widening the stereo field with more unison detune and chorus.",
     "params": {"osc_1_unison_detune": ("add", 0.15), "chorus_dry_wet": ("add", 0.15)}, "tags": ["stereo"]},
    {"prompts": ["Make it more aggressive", "Add some grit", "Dirty it up"],
     "response": "Adding grit with more distortion drive.",
     "params": {"distortion_drive": ("add", 0.3), "distortion_on": ("set", 1.0)}, "tags": ["fx"]},
    {"prompts": ["Less reverb", "Make it drier", "Too much space, pull it back"],
     "response": "Pulling back the reverb for a drier sound.",
     "params": {"reverb_dry_wet": ("subtract", 0.2)}, "tags": ["fx"]},
    {"prompts": ["Add more movement", "It's too static, make it evolve", "Can you add some modulation?"],
     "response": "Adding more movement by increasing the modulation depth.",
     "params": {"lfo_1_frequency": ("multiply", 1.5)}, "tags": ["modulation"]},
    {"prompts": ["Turn down the resonance", "The filter is too ringy"],
     "response": "Reducing the filter resonance to tame the ringing.",
     "params": {"filter_1_resonance": ("subtract", 0.15)}, "tags": ["filter"]},
    {"prompts": ["Make the release longer", "Let it ring out more"],
     "response": "Extending the release for a longer tail.",
     "params": {"env_1_release": ("multiply", 2.0)}, "tags": ["envelope"]},
    {"prompts": ["Shorten the release", "Cut it off quicker"],
     "response": "Tightening the release for a cleaner cutoff.",
     "params": {"env_1_release": ("multiply", 0.4)}, "tags": ["envelope"]},
]

from maestro.synth.path_gen import PARAM_RANGES as _STEER_RANGES


def _apply_steer_op(current: float, op: str, value: float, name: str) -> float:
    r = _STEER_RANGES.get(name, {})
    lo, hi = r.get("min", 0.0), r.get("max", 1.0)
    if op == "add":
        result = current + value
    elif op == "subtract":
        result = current - value
    elif op == "multiply":
        result = current * value
    elif op == "set":
        result = value
    else:
        result = current
    return max(lo, min(hi, result))


def _build_steer_turns(
    cumulative: dict,
    notes: list,
    messages: list[dict],
    audio_assets: list[str],
    batch_audio_dir: Path,
    rng: "random.Random",
    sample_id: str,
    ctx: "DawFarmRolloutCtx | None" = None,
) -> None:
    """Append 1-2 user steer turns to the conversation."""
    n_turns = rng.choice([1, 1, 1, 2])
    used_tags: set[str] = set()
    available = list(STEER_CATALOG)

    for turn_i in range(n_turns):
        candidates = [d for d in available if not (set(d["tags"]) & used_tags)]
        if not candidates:
            break
        direction = rng.choice(candidates)
        available.remove(direction)
        used_tags.update(direction["tags"])

        user_text = rng.choice(direction["prompts"])
        messages.append({"role": "user", "content": user_text})

        steer_params: dict[str, float] = {}
        settings = cumulative.get("settings", {})
        for param_name, (op, value) in direction["params"].items():
            current = float(settings.get(param_name, 0.0))
            new_val = _apply_steer_op(current, op, value, param_name)
            steer_params[param_name] = round(new_val, 4) if not isinstance(new_val, int) else new_val
            settings[param_name] = new_val

        messages.append({"role": "assistant", "content": direction["response"]})
        action_snippet = build_batch_action_snippet(steer_params)
        messages.append(_tool_call("Bash", {"command": action_snippet}))
        if ctx is not None:
            _action_stdout = ctx.real_exec(action_snippet, "steer apply").stdout
        else:
            _action_stdout = json.dumps({"status": "ok", "applied": len(steer_params)}) + "\n"
        messages.append(_bash_tool_response(_action_stdout))

        steer_wav = batch_audio_dir / f"steer_{turn_i}.wav"
        if ctx is None:
            render_cumulative_audio(cumulative, notes, steer_wav)
        audio_assets.append(str(steer_wav))
        messages.append({"role": "assistant", "content": "Rendering after the adjustment."})
        _render_cmd = _wrap_as_bash(build_reaper_render_snippet(
            out_path=ctx.cw(steer_wav) if ctx is not None else str(steer_wav)))
        messages.append(_tool_call("Bash", {"command": _render_cmd}))
        if ctx is not None:
            _srres = ctx.real_exec(_render_cmd, "steer render")
            ctx.fetch_wav(ctx.cw(steer_wav), steer_wav)
            _emit_listen_sequence(messages, audio_assets, steer_wav,
                                  probe_stdout=_srres.stdout, display_path=ctx.cw(steer_wav))
        else:
            _emit_listen_sequence(messages, audio_assets, steer_wav)
        messages.append({"role": "assistant", "content": "Done — the adjustment has been applied."})


# ---------------------------------------------------------------------------
# Main builder — modified to call real subagent builders
# ---------------------------------------------------------------------------


def compute_search_partition(total_named: int, slice_size: int = 48) -> list[int]:
    """Contiguous shard starts covering the full library — the single source
    of truth for how search agents partition wavetable indices (imported by
    phase-A embedding preparation; keep in sync with nothing, reuse this)."""
    n_agents = max(1, (total_named + slice_size - 1) // slice_size)
    return [i * slice_size for i in range(n_agents)]


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
    dawfarm_session: "_dawfarm.DawFarmSession | None" = None,
) -> tuple[dict | None, list[dict], list[dict], list[dict]]:
    """Build one unified SFT record.

    Returns (main_record, search_records, judge_records, transcription_records).
    main_record is None to skip; subagent lists may be empty.

    With *dawfarm_session*, all emitted snippets execute inside a real
    daw-farm REAPER container (see DawFarmRolloutCtx); sub-agent env execs
    serialize on the same session via the ctx lock.
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
    factory_init_settings = dict(init_preset.get("settings", {}))

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

    # Partial init: pre-apply 1-4 random GT subsystems so the model
    # sees variable-length rollouts (not always all 7 subsystems).
    pre_applied_subsystems: list[str] = []
    _partial_init_rate = float(getattr(args, "partial_init_rate", 0.0))
    if not use_random_init and _partial_init_rate > 0.0:
        _partial_rng = random.Random(int(args.seed) + sid_seed + 9999)
        if _partial_rng.random() < _partial_init_rate:
            n_pre = _partial_rng.randint(1, 4)
            all_subs = [label for label, _ in SUBSYSTEM_ORDER]
            pre_applied_subsystems = _partial_rng.sample(all_subs, min(n_pre, len(all_subs)))
            truth_map = build_diagnosis_subsystem_truth(target_preset, init_preset)
            for sub in pre_applied_subsystems:
                for param_name in truth_map.get(sub, []):
                    val = target_preset["settings"].get(param_name)
                    if val is not None and isinstance(val, (int, float)):
                        init_preset["settings"][param_name] = val
            if "oscillator" in pre_applied_subsystems:
                init_preset["settings"]["wavetables"] = copy.deepcopy(
                    target_preset["settings"]["wavetables"])
            if "modulation" in pre_applied_subsystems:
                init_preset["settings"]["modulations"] = copy.deepcopy(
                    target_preset["settings"]["modulations"])
            if "lfo" in pre_applied_subsystems:
                init_preset["settings"]["lfos"] = copy.deepcopy(
                    target_preset["settings"]["lfos"])

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
    # Sort by name: the emitted list/render snippets sort discovered
    # wavetable names, so the host index must use the same order or every
    # idx-keyed step (slices, probe filenames) disagrees with the live env.
    _unique_wts.sort()
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
    max_rounds = int(getattr(args, "max_search_rounds", 3))
    # Full contiguous partition: agent count derives from library size, no
    # gaps, no GT-targeted slice adjustment. (The old stride grid left ~1/3
    # of the library unsearchable and teleported one slice onto the GT when
    # needed — a structural label leak: the off-grid slice held the answer.)
    slice_starts = compute_search_partition(total_named, slice_size)
    n_agents = len(slice_starts)

    def _slice_ranges_str(starts: list[int]) -> str:
        return ", ".join(
            f"{s}-{min(s + slice_size, total_named) - 1}" for s in starts
        )

    gt_idxs = [name_to_idx_full[n] for n in gt_names_list if n in name_to_idx_full]
    force_research_rate = float(getattr(args, "force_research_rate", 0.30))
    # Forced re-search now happens at the SHORTLIST level: in round 1 the
    # search agent covering a GT auditions it but doesn't shortlist it (the
    # realistic inference failure — imperfect ears, not bad slicing), so the
    # judge reports no_match and round 2 re-dispatches the same partition.
    force_miss = sample_rng.random() < force_research_rate

    # ---- Begin messages ----
    from scripts import opencode_contract as _oc
    messages: list[dict] = []
    # cwd must be deterministic from the sample alone: ctx is not created yet
    # at this point (and the no-audio edge case returns before it exists).
    # rollout_dir(sample_id) is the same value ctx.cw("", "") would produce.
    from maestro.reaper.dawfarm import rollout_dir as _rollout_dir
    messages.append({
        "role": "system",
        "content": _oc.system_message(
            _oc.AGENT_PROMPTS["main"], cwd=_rollout_dir(sample_id)),
    })
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
            "tools": _OC_TOOLS_JSON,
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
                "num_agents": n_agents,
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

    # ---- daw-farm real-execution mode (mirrors v3; see DawFarmRolloutCtx) ----
    ctx: DawFarmRolloutCtx | None = None
    if dawfarm_session is not None:
        ctx = DawFarmRolloutCtx(
            session=dawfarm_session,
            sample_id=sample_id,
            exec_timeout=float(getattr(args, "daw_farm_timeout", 300.0)),
        )
        _log(f"daw-farm session {dawfarm_session.name} — reset project + sync data")
        _dawfarm.reset_project(dawfarm_session)
        _dawfarm.sync_vital_data(dawfarm_session, getattr(args, "daw_farm_vital_data", None))
        _dawfarm.prepare_sample_dirs(dawfarm_session, sample_id)
        _dawfarm.set_project_tempo(
            dawfarm_session,
            60.0 * float(target_preset.get("settings", {}).get("beats_per_minute", 2.0)))
        if not dawfarm_session.exec_argv(["test", "-d", "/work/skills"]).ok:
            dawfarm_session.put(str(ROOT / "skills"), "/work/skills")

    # Output dirs (needed before the transcription block below).
    tuple_audio_dir = Path(args.out_dir) / "tuple_audio" / sample_id
    tuple_audio_dir.mkdir(parents=True, exist_ok=True)
    search_probe_dir = Path(getattr(args, "search_probe_dir", None)
                            or (Path(args.out_dir) / "search_probe_audio"))
    judge_probe_dir = Path(args.out_dir) / "judge_probe_audio"
    transcription_dir = Path(args.out_dir) / "transcription_audio"

    _log("start — skill discovery + transcription")
    messages.append({
        "role": "assistant",
        "content": "Let me see which skills are available for this plugin.",
    })
    messages.append(_tool_call("Bash", {"command": "ls skills/*/SKILL.md"}))
    if ctx is not None:
        messages.append(_bash_tool_response(
            ctx.real_exec("ls skills/*/SKILL.md", "skill discovery").stdout))
    else:
        messages.append(_bash_tool_response("\n".join(_available_skill_paths) + "\n"))

    messages.append({
        "role": "assistant",
        "content": f"The {_skill_name} skill matches. Loading it for the plugin-specific instructions, helper-script paths, and recreation strategy.",
    })
    messages.append(_tool_call("Skill", {"skill": _skill_name, "args": ""}))
    messages.append({
        "role": "tool_response",
        "content": _oc.skill_output(_skill_name, str(_skill_md_path),
                                    _skill_md_text, _skill_description),
    })

    # Create the REAPER track holding Vital (and later the transcribed MIDI).
    # Unconditional (v3 order): the default render below reads the live
    # preset, and the apply/batch snippets all assume track 0 + Vital.
    _track_name = "target_melody"
    _create_track_snippet = (
        _REAPY_HELPER
        + f"track_name = {json.dumps(_track_name)}\n"
          f"with reapy.inside_reaper():\n"
          f"    RPR.InsertTrackAtIndex(0, True)\n"
          f"    track = RPR.GetTrack(0, 0)\n"
          f"    RPR.GetSetMediaTrackInfo_String(track, 'P_NAME', track_name, True)\n"
          f"    RPR.TrackFX_AddByName(track, 'Vital', False, 1)\n"
          f"print(json.dumps({{'status': 'ok', 'track_idx': 0, 'track_name': track_name}}))\n"
    )
    _create_track_cmd = _wrap_as_bash(_create_track_snippet)
    messages.append({
        "role": "assistant",
        "content": "Creating a REAPER track with Vital loaded to hold the recreation.",
    })
    messages.append(_tool_call("Bash", {"command": _create_track_cmd}))
    if ctx is not None:
        messages.append(_bash_tool_response(
            ctx.real_exec(_create_track_cmd, "track creation").stdout))
        # Align the live Vital with the sample's init preset — a fresh VST3
        # instance boots with the factory patch, not maestro's baseline.
        _dawfarm.apply_vital_preset(dawfarm_session, init_preset)
    else:
        messages.append(_bash_tool_response(json.dumps({
            "status": "ok", "track_idx": 0, "track_name": _track_name,
        }) + "\n"))

    # Render baseline (needed for batch-rendering diffs later), but skip
    # listening — the model already heard the result via the judge agent.
    _default_render_cmd = _wrap_as_bash(build_render_verify_snippet(
        out_path=ctx.cw(default_audio_path) if ctx is not None else str(default_audio_path),
        notes_override=list(notes),
    ))
    messages.append(_tool_call("Bash", {"command": _default_render_cmd}))
    if ctx is not None:
        _dres = ctx.real_exec(_default_render_cmd, "default render",
                              timeout=max(ctx.exec_timeout, 600.0))
        messages.append(_bash_tool_response(_dres.stdout))
    else:
        messages.append(_bash_tool_response(
            json.dumps({"rendered": str(default_audio_path), "ok": True})
        ))

    # --- TRANSCRIPTION BLOCK ---
    # Create a REAPER track and dispatch the transcription subagent.
    # Verification and mistake recovery now happen inside the transcription
    # agent itself (v4) — the main agent just dispatches once and proceeds.
    transcription_summary_text = ""
    _trans_output_file: str | None = None
    _trans_notes_file: str | None = None
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
            # daw-farm mode: /tmp/agents is writable inside the container and
            # the same absolute path exists on both sides (v3 convention);
            # host-repo paths can't be created by the container user.
            _trans_agent_dir = (
                f"/tmp/agents/{sample_id}" if dawfarm_session is not None
                else str(Path(args.out_dir) / "agent_workdir" / sample_id)
            )
            _trans_output_file = f"{_trans_agent_dir}/{_trans_agent_id}.md"
            _trans_manifest_file = f"{_trans_agent_dir}/{_trans_agent_id}.manifest.json"
            _trans_notes_file = f"{_trans_agent_dir}/{sample_id}_notes.json"

            Path(_trans_output_file).parent.mkdir(parents=True, exist_ok=True)
            # MIDI notes file — structured JSON for render functions
            with open(_trans_notes_file, "w") as _tnf:
                json.dump({"notes": _trans_notes, "n_notes": _trans_n_notes, "duration_s": _trans_duration_s}, _tnf)
                _tnf.write("\n")
            if ctx is not None:
                # Same absolute paths in-container so midi_path snippets resolve.
                dawfarm_session.put(_trans_notes_file, _trans_notes_file)

            # Dispatch transcription subagent (track with Vital created above)
            _dispatch_prompt = _oc.transcription_dispatch_prompt(
                target_audio_path, _trans_track_idx)
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
                "run_in_background": True,
            }))
            # (task result is appended inline after the sub-record builds)

            # Collect the transcription subagent record for training
            _trans_mistake_rate = float(getattr(args, "transcription_mistake_rate", 0.0) or 0.0)
            trans_result = build_transcription_record_v4(
                sample_id=sample_id,
                archetype=archetype,
                target_audio_path=target_audio_path,
                source_midi_path=source_midi_path,
                output_dir=transcription_dir,
                track_idx=_trans_track_idx,
                mistake_rate=_trans_mistake_rate,
                seed=int(args.seed),
                dawfarm=ctx,
            )
            if trans_result.record:
                transcription_records.append(trans_result.record)

            # Write the transcription agent's final message to the output file.
            # In claw-code, outputFile = agent's last assistant response (text).
            trans_final_msg = (
                trans_result.record["messages"][-1]["content"]
                if trans_result.record
                else f"Transcription complete. {_trans_n_notes} notes on track 0."
            )
            with open(_trans_output_file, "w") as _trf:
                _trf.write(trans_final_msg)
                _trf.write("\n")
            # opencode contract: the task tool's response IS the subagent's
            # final message, inline — no outputFile cat round-trip.
            from scripts.agent_sft_common import oc_task_result_msg as _oc_task_result
            messages.append(_oc_task_result(_trans_agent_id, trans_final_msg))
            if ctx is not None:
                dawfarm_session.put(_trans_output_file, _trans_output_file)
                # Project state (the MIDI item) was left by the transcription
                # sub-builder's real final-attempt insert above.

            transcription_summary_text = (
                f"Transcription verified — {_trans_n_notes} notes on track "
                f"{_trans_track_idx}. "
            )

    if ctx is not None and _trans_notes_file is None:
        # No source MIDI — the timeline still needs notes for REAPER renders
        # and downstream snippets need a notes JSON (infra, not conversation).
        _fallback_notes = [
            {"pitch": int(p), "velocity": int(v), "start_s": float(s), "dur_s": float(d)}
            for (p, v, s, d) in notes
        ]
        _trans_notes_file = f"/tmp/agents/{sample_id}/{sample_id}_notes.json"
        Path(_trans_notes_file).parent.mkdir(parents=True, exist_ok=True)
        with open(_trans_notes_file, "w") as _tnf:
            json.dump({"notes": _fallback_notes, "n_notes": len(_fallback_notes)}, _tnf)
        dawfarm_session.put(_trans_notes_file, _trans_notes_file)
        _dawfarm.insert_midi_notes(dawfarm_session, _fallback_notes)

    # Library size check
    messages.append({
        "role": "assistant",
        "content": f"{transcription_summary_text}Checking wavetable library size.",
    })
    messages.append(_tool_call("Bash", {"command": _wrap_as_bash(build_list_wavetables_total_snippet())}))
    if ctx is not None:
        _wt_res = ctx.real_exec(_wrap_as_bash(build_list_wavetables_total_snippet()), "wavetable count")
        messages.append(_bash_tool_response(_wt_res.stdout))
        try:
            _container_total = int(json.loads(_wt_res.stdout).get("total", -1))
        except Exception:
            _container_total = -1
        if _container_total != total_named:
            _log(f"WARNING: container wavetable library has {_container_total} "
                 f"entries vs host {total_named} — sync --daw-farm-vital-data")
    else:
        messages.append(_bash_tool_response(json.dumps({"total": total_named}) + "\n"))

    agent_out_dir = (
        f"/tmp/agents/{sample_id}" if dawfarm_session is not None
        else str(Path(args.out_dir) / "agent_workdir" / sample_id)
    )

    _wt_name_to_emb = build_name_embedding_map(shortlist_data["embeddings"], index_rows)

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
                f"agents in parallel across contiguous slices covering the full library "
                f"[{_slice_ranges_str(slice_starts)}]."
            )
        else:
            intro = (
                f"{_research_prefix}"
                f"Re-dispatching {n_agents} search agents across the full library for "
                f"a fresh audition: "
                f"[{_slice_ranges_str(slice_starts)}]."
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
            # Round-1 forced miss: the agent whose shard holds a GT auditions
            # it but leaves it off the shortlist (realistic perceptual miss).
            _fm_names = None
            if force_miss and rounds_used == 1:
                _fm_names = [n for n in gt_names_list
                             if start <= name_to_idx_full.get(n, -1) < end] or None
            # Data-mix control: emit the GT-holding shard's record always,
            # others at --search-record-keep-rate (pool-only otherwise).
            _has_gt = any(start <= name_to_idx_full.get(n, -1) < end for n in gt_names_list)
            _keep_rng = random.Random(int(args.seed) + sid_seed + rounds_used * 101 + ai)
            _keep = _has_gt or _keep_rng.random() < float(getattr(args, "search_record_keep_rate", 1.0))
            if ctx is not None:
                # v3: opencode contract + evidence labels + fixed probe renders.
                # (round-1 forced miss is not supported by evidence labels —
                # rounds still re-dispatch on natural judge no_match verdicts.)
                from scripts.build_search_agent_sft_v3 import build_search_record_v3
                sr = build_search_record_v3(
                    sample_id=sample_id,
                    agent_idx=ai + 1,
                    archetype=archetype,
                    target_audio_path=target_audio_path,
                    gt_wavetable_names=gt_names_list,
                    shard_start=start,
                    shard_end=end,
                    name_to_idx=name_to_idx_full,
                    idx_to_name=idx_to_name_full,
                    embedder=embedder,
                    dawfarm=ctx,
                    midi_path=_trans_notes_file,
                    probe_audio_dir=search_probe_dir,
                    stage2_server=stage2_server,
                    stage2_model=stage2_model,
                    candidates_per_batch=int(getattr(args, "candidates_per_batch", 8)),
                    clap_threshold=0.97,
                    shortlist_dir=None,
                    pool_only=not _keep,
                )
            else:
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
                    midi_path=_trans_notes_file,
                    probe_audio_dir=search_probe_dir,
                    dawfarm=ctx,
                    force_miss_names=_fm_names,
                    merged_stage2=bool(getattr(args, "merged_stage2", False)),
                    pool_only=not _keep,
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
            out_path = out_dir / f"{agent_id}.md"
            manifest_path = out_dir / f"{agent_id}.manifest.json"
            final_msg = (search_result.record["messages"][-1]["content"]
                         if search_result.record else search_result.final_message)
            with open(out_path, "w") as f:
                f.write(final_msg)
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
            _search_prompt_parts = [_oc.search_dispatch_prompt(
                target_audio_path, _trans_notes_file, start, end)]
            messages.append(_tool_call("Agent", {
                "subagent_type": "wavetable_search",
                "description": f"Evaluate wavetables {start}-{end - 1} for target sound",
                "prompt": "\n".join(_search_prompt_parts),
                "name": f"search-r{rounds_used}-a{ai_idx + 1}",
                "run_in_background": True,
            }))

        from scripts.agent_sft_common import oc_task_result_msg as _oc_task_result
        round_shortlists: list[list[str]] = []
        for _start, _end, agent_id, output_file, manifest_file, _sl in round_agent_meta:
            if ctx is not None:
                dawfarm_session.put(output_file, output_file)
            # opencode contract: inline <task_result> response, no cat/grep.
            try:
                with open(output_file) as _of:
                    _final_txt = _of.read().strip()
            except Exception:
                _final_txt = ""
            messages.append(_oc_task_result(agent_id, _final_txt))
            round_shortlists.append(_sl)

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
                            )

        if _carried_locked_slots:
            _locked_prompt_section = (
                f"\nPreviously confirmed (locked) selections from prior round: "
                + json.dumps({str(k): v for k, v in _carried_locked_slots.items()})
                + ". Keep these locked — only evaluate new candidates for the unfilled slots."
            )
        else:
            _locked_prompt_section = ""
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
            probe_audio_dir=judge_probe_dir,
            midi_path=_trans_notes_file,
            dawfarm=ctx,
            locked_prompt_section=_locked_prompt_section,
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
        judge_output_file = Path(agent_out_dir) / f"{judge_agent_id}.md"
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
        _judge_dispatch_prompt = _oc.judge_dispatch_prompt(
            target_audio_path, pool, _locked_prompt_section)
        # Write judge output file on disk for the cat (always overwrite —
        # stale files from prior builds would show the wrong verdict).
        # In claw-code, outputFile contains the agent's final assistant message.
        judge_output_file.parent.mkdir(parents=True, exist_ok=True)
        judge_final_msg = judge_result.record["messages"][-1]["content"] if judge_result.record else ""
        with open(judge_output_file, "w") as jf:
            jf.write(judge_final_msg)
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
            "run_in_background": True,
        }))
        # opencode contract: judge's final message returns inline in the
        # task tool_response — no outputFile cat round-trip.
        if ctx is not None:
            dawfarm_session.put(str(judge_output_file), str(judge_output_file))
        from scripts.agent_sft_common import oc_task_result_msg as _oc_task_result
        messages.append(_oc_task_result(judge_agent_id, judge_final_msg.strip()))

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
                    f"Re-dispatching the search for a fresh audition of the library. "
                )
            # Same full partition — re-search means a fresh audition, not new
            # regions (there are none: the slices already cover everything).

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
                                    )
            continue

        # Good verdict — render tuple, listen, break
        tuple_wav = tuple_audio_dir / f"tuple_r{rounds_used}.wav"
        osc_names = {oi: cur_tuple[oi] for oi in active_oscs if cur_tuple[oi]}
        render_cmd = _wrap_as_bash(build_render_tuple_snippet(
            osc_names=osc_names,
            out_path=ctx.cw(tuple_wav) if ctx is not None else str(tuple_wav),
            midi_path=_trans_notes_file,
        ))
        if ctx is None:
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
        if ctx is not None:
            _tres = ctx.real_exec(render_cmd, "tuple render", timeout=max(ctx.exec_timeout, 600.0))
            ctx.fetch_wav(ctx.cw(tuple_wav), tuple_wav)
            _tuple_stdout = _tres.stdout
        else:
            _tuple_stdout = json.dumps({
                "status": "ok", "out": str(tuple_wav), "wavetables": cur_active_names,
            }) + "\n"
        _emit_listen_sequence(
            messages, audio_assets, tuple_wav,
            probe_stdout=_tuple_stdout,
            display_path=ctx.cw(tuple_wav) if ctx is not None else None,
        )
        break

    # Exhausted fallback
    if judge_exhausted_fallback:
        tuple_wav = tuple_audio_dir / f"tuple_r{rounds_used}_fallback.wav"
        osc_names = {oi: cur_tuple[oi] for oi in active_oscs if cur_tuple[oi]}
        render_cmd = _wrap_as_bash(build_render_tuple_snippet(
            osc_names=osc_names,
            out_path=ctx.cw(tuple_wav) if ctx is not None else str(tuple_wav),
            midi_path=_trans_notes_file,
        ))
        if ctx is None:
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
        if ctx is not None:
            _fbres = ctx.real_exec(render_cmd, "fallback tuple render",
                                   timeout=max(ctx.exec_timeout, 600.0))
            ctx.fetch_wav(ctx.cw(tuple_wav), tuple_wav)
            _fb_tuple_stdout = _fbres.stdout
        else:
            _fb_tuple_stdout = json.dumps({
                "status": "ok", "out": str(tuple_wav), "wavetables": cur_active_names,
            }) + "\n"
        _emit_listen_sequence(
            messages, audio_assets, tuple_wav,
            probe_stdout=_fb_tuple_stdout,
            display_path=ctx.cw(tuple_wav) if ctx is not None else None,
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

    batches = build_batches_from_diff(target_preset, init_preset)
    batches = batches[:int(args.max_batches)]

    apply_assignments = repr(
        [(oi, gt_tuple[oi]) for oi in active_oscs if gt_tuple[oi]]
    )
    target_mods_literal = repr(target_modulations)
    _tmp_batch_params = set()
    for b in batches:
        _tmp_batch_params.update(b.params.keys())
    base_scalar_overrides = {}
    for name, val in target_preset.get("settings", {}).items():
        if isinstance(val, (int, float)) and name not in _tmp_batch_params:
            factory_val = factory_init_settings.get(name)
            if factory_val != val:
                base_scalar_overrides[name] = val
    overrides_literal = repr(base_scalar_overrides)
    apply_snippet = (
        _REAPY_HELPER
        + _READ_CHUNK_HELPER
        + _BUILD_CHUNK_HELPER
        + "import base64\n"
        + _WT_DISCOVER_SNIPPET
        + "name_to_wt = {wt['name']: wt for wt in lib if 'name' in wt}\n"
        "preset = read_vital_preset()\n"
        f"for osc_idx, wt_name in {apply_assignments}:\n"
        "    if wt_name in name_to_wt:\n"
        "        preset['settings']['wavetables'][osc_idx] = name_to_wt[wt_name]\n"
        f"preset['settings']['modulations'] = {target_mods_literal}\n"
        f"_lfos = {repr(target_preset.get('settings', {}).get('lfos'))}\n"
        "if _lfos is not None:\n"
        "    preset['settings']['lfos'] = _lfos\n"
        f"for _k, _v in {overrides_literal}.items():\n"
        "    preset['settings'][_k] = _v\n"
        "chunk = build_vital_chunk(preset)\n"
        "encoded = base64.b64encode(chunk).decode('ascii')\n"
        "with reapy.inside_reaper():\n"
        "    track = RPR.GetTrack(0, 0)\n"
        "    if not RPR.TrackFX_SetNamedConfigParm(track, 0, 'vst_chunk', encoded):\n"
        "        RPR.TrackFX_SetNamedConfigParm(track, 0, 'vst3_chunk', encoded)\n"
        f"print(json.dumps({{'status': 'ok', 'applied': {json.dumps(apply_names)}}}))"
    )
    messages.append({"role": "assistant", "content": selection_text})
    messages.append(_tool_call("Bash", {"command": _wrap_as_bash(apply_snippet)}))
    if ctx is not None:
        _apply_stdout = ctx.real_exec(_wrap_as_bash(apply_snippet), "wavetable apply").stdout
    else:
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

    path_complete = True

    cumulative = copy.deepcopy(init_preset)
    for key in ("wavetables", "sample", "lfos", "modulations"):
        if key in target_preset.get("settings", {}):
            cumulative["settings"][key] = copy.deepcopy(target_preset["settings"][key])
    batch_param_names = set()
    for b in batches:
        batch_param_names.update(b.params.keys())
    for name, val in target_preset.get("settings", {}).items():
        if isinstance(val, (int, float)) and name not in batch_param_names:
            cumulative["settings"][name] = val

    mistake_rng = random.Random(int(args.seed) + sid_seed)
    _per_param_rate = getattr(args, "per_param_mistake_rate", None)
    if _per_param_rate is None:
        _per_param_rate = getattr(args, "mistake_rate", 0.10)
    all_injected_mistakes, max_mistakes_drawn = inject_mistakes(
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

    # Mid-conversation re-transcription decision
    _retranscribe_rate = float(getattr(args, "retranscribe_rate", 0.0))
    retranscribe_after_batch: int | None = None
    _correct_notes = None
    _correct_notes_dicts = None
    if _retranscribe_rate > 0 and len(batches) >= 3 and mistake_rng.random() < _retranscribe_rate:
        retranscribe_after_batch = mistake_rng.randint(1, len(batches) - 2)
        # Mutate the current notes so early batches render with wrong MIDI
        _correct_notes = notes  # save correct DawDreamer-format notes
        _correct_notes_dicts = _trans_notes if '_trans_notes' in dir() else None
        from scripts.agent_sft_common import apply_transcription_mutations
        _retrans_rng = random.Random(int(args.seed) + sid_seed + 5555)
        _mut_result = apply_transcription_mutations(
            load_notes_from_midi(source_midi_path) if source_midi_path else [],
            _retrans_rng,
        )
        if _mut_result is not None:
            _wrong_notes_dicts, _mut_infos, _mut_narration = _mut_result
            notes = notes_from_dicts(_wrong_notes_dicts)
            if ctx is not None:
                # Back-date the fiction: put the wrong notes on the real
                # timeline so early batch renders are genuinely wrong.
                _dawfarm.insert_midi_notes(dawfarm_session, _wrong_notes_dicts)
            _log(f"retranscribe after batch {retranscribe_after_batch} ({len(_mut_infos)} mutations)")
        else:
            retranscribe_after_batch = None

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
        if ctx is None:
            render_cumulative_audio(cumulative, notes, batch_wav)
            _log(f"  batch {bi}/{len(batches)-1} ({b.subsystem}) render {_time.monotonic()-_bt:.1f}s")
        # daw-farm mode: audio comes from a real REAPER render after the
        # apply below; CLAP is computed once the wav exists (post-listen).

        b.audio_wav = batch_wav
        params_native = {
            n: _denormalize(n, v)
            for n, v in b.params_applied.items()
        }
        action_snippet = build_batch_action_snippet(params_native)

        search_queries = _batch_search_queries(
            b.subsystem, list(b.params_applied.keys()), _JSON_KEY_TO_REAPER,
        )

        def _q_label(q):
            """Human label for a lookup: exact-name lists read as a count."""
            if isinstance(q, str):
                return f"{q} parameters"
            return (f"the {len(q)} {b.subsystem} parameters I need"
                    if len(q) > 1 else f"'{q[0]}'")

        query_label = " and ".join(_q_label(q) for q in search_queries)
        intro = f"Looking up {query_label}."
        if bi == 0 and _diagnosis_text:
            intro = f"{_diagnosis_text}\n\n{intro}"
            _diagnosis_text = None
        elif pending_check:
            intro = f"{pending_check}\n\n{intro}"
            pending_check = None
        messages.append({"role": "assistant", "content": intro})

        for sq in search_queries:
            search_snippet = _wrap_as_bash(build_param_search_snippet(sq))
            messages.append(_tool_call("Bash", {"command": search_snippet}))
            if ctx is not None:
                messages.append(_bash_tool_response(
                    ctx.real_exec(search_snippet, f"param search {sq!r}").stdout))
            else:
                search_results = simulate_param_search(
                    sq, _REAPER_PARAM_DUMP,
                    value_overrides=current_reaper_values,
                    max_results=10_000,
                )
                # identical formatting to the live snippet (env-exactness)
                search_stdout = format_param_search_output(
                    sq, search_results, total=len(search_results))
                messages.append(_bash_tool_response(search_stdout))

        messages.append({"role": "assistant", "content": f"Applying {b.subsystem} changes."})
        messages.append(_tool_call("Bash", {"command": action_snippet}))
        if ctx is not None:
            _action_stdout = ctx.real_exec(action_snippet, f"{b.subsystem} apply").stdout
        else:
            _action_stdout = json.dumps({"status": "ok", "applied": len(params_native)}) + "\n"
        messages.append(_bash_tool_response(_action_stdout))

        for name, norm in b.params_applied.items():
            if name in _JSON_KEY_TO_REAPER:
                current_reaper_values[_JSON_KEY_TO_REAPER[name]["idx"]] = float(norm)

        audio_assets.append(str(batch_wav))
        messages.append({"role": "assistant", "content": f"Listening after {b.subsystem} batch."})
        _batch_render_cmd = _wrap_as_bash(build_reaper_render_snippet(
            out_path=ctx.cw(batch_wav) if ctx is not None else str(batch_wav)))
        messages.append(_tool_call("Bash", {"command": _batch_render_cmd}))
        if ctx is not None:
            _bt = _time.monotonic()
            _rres = ctx.real_exec(_batch_render_cmd, f"{b.subsystem} render")
            ctx.fetch_wav(ctx.cw(batch_wav), batch_wav)
            _log(f"  batch {bi}/{len(batches)-1} ({b.subsystem}) reaper render {_time.monotonic()-_bt:.1f}s")
            _emit_listen_sequence(messages, audio_assets, batch_wav,
                                  probe_stdout=_rres.stdout, display_path=ctx.cw(batch_wav))
        else:
            _emit_listen_sequence(messages, audio_assets, batch_wav)
        last_batch_audio = batch_wav

        # CLAP vs GT (wav exists in both modes now)
        with serial_lock:
            try:
                clap_after = float(embedder.cosine_paths(batch_wav, target_audio_path))
            except Exception:
                clap_after = None

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
                if ctx is None:
                    render_cumulative_audio(cumulative, notes, corr_wav)

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

                corr_native: dict[str, float] = {}
                for m in fixing_now:
                    corr_native[m.param] = _denormalize(m.param, m.true_value)

                if corr_native:
                    _corr_action_cmd = build_batch_action_snippet(corr_native)
                    messages.append(_tool_call("Bash", {"command": _corr_action_cmd}))
                    if ctx is not None:
                        _corr_stdout = ctx.real_exec(_corr_action_cmd, "correction apply").stdout
                    else:
                        _corr_stdout = json.dumps({"status": "ok", "applied": len(corr_native)}) + "\n"
                    messages.append(_bash_tool_response(_corr_stdout))
                else:
                    messages.append(_tool_call("Bash", {"command": "echo 'no matching REAPER param'"}))
                    messages.append(_bash_tool_response("no matching REAPER param\n"))

                audio_assets.append(str(corr_wav))
                messages.append({"role": "assistant", "content": "Listening to the corrected preset."})
                _corr_render_cmd = _wrap_as_bash(build_reaper_render_snippet(
                    out_path=ctx.cw(corr_wav) if ctx is not None else str(corr_wav)))
                messages.append(_tool_call("Bash", {"command": _corr_render_cmd}))
                if ctx is not None:
                    _crres = ctx.real_exec(_corr_render_cmd, "correction render")
                    ctx.fetch_wav(ctx.cw(corr_wav), corr_wav)
                    _emit_listen_sequence(
                        messages, audio_assets, corr_wav,
                        probe_stdout=_crres.stdout, display_path=ctx.cw(corr_wav),
                    )
                else:
                    _emit_listen_sequence(
                        messages, audio_assets, corr_wav,
                    )
                last_batch_audio = corr_wav
                with serial_lock:
                    try:
                        corr_clap = float(embedder.cosine_paths(corr_wav, target_audio_path))
                    except Exception:
                        corr_clap = None

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

        # ---- MID-CONVERSATION RE-TRANSCRIPTION ----
        if retranscribe_after_batch is not None and bi == retranscribe_after_batch:
            _retrans_agent_id = make_agent_id(sample_id, "melody_retranscription")
            _retrans_agent_dir = (
                f"/tmp/agents/{sample_id}" if dawfarm_session is not None
                else str(Path(args.out_dir) / "agent_workdir" / sample_id)
            )
            _retrans_output_file = f"{_retrans_agent_dir}/{_retrans_agent_id}.md"
            _retrans_manifest_file = f"{_retrans_agent_dir}/{_retrans_agent_id}.manifest.json"

            _retrans_prefix = f"{pending_check}\n\n" if pending_check else ""
            pending_check = None
            messages.append({
                "role": "assistant",
                "content": (
                    _retrans_prefix
                    + "Listening closely, some of the MIDI notes don't match the target melody — "
                    "the transcription sounds off now that effects are applied. Re-transcribing."
                ),
            })

            _retrans_prompt = (
                f"Target: {target_audio_path}. Track: 0. The previous transcription has "
                f"errors — re-listen and write corrected MIDI notes."
            )
            write_agent_manifest(
                agent_id=_retrans_agent_id,
                subagent_type="melody_transcription",
                output_file=_retrans_output_file,
                manifest_file=_retrans_manifest_file,
                prompt=_retrans_prompt,
            )
            messages.append(_tool_call("Agent", {
                "subagent_type": "melody_transcription",
                "description": "Re-transcribe target melody — previous transcription has errors",
                "prompt": _retrans_prompt,
                "name": f"retranscribe-{sample_id}",
                "run_in_background": True,
            }))
            messages.append({
                "role": "tool_response",
                "content": json.dumps({
                    "status": "completed",
                    "outputFile": _retrans_output_file,
                }),
            })

            retrans_result = build_transcription_record_v4(
                sample_id=sample_id,
                archetype=archetype,
                target_audio_path=target_audio_path,
                source_midi_path=source_midi_path,
                output_dir=transcription_dir,
                track_idx=0,
                mistake_rate=0.0,
                seed=int(args.seed) + 6666,
                dawfarm=ctx,
            )
            if retrans_result.record:
                retrans_result.record["id"] = f"{sample_id}_retranscription"
                transcription_records.append(retrans_result.record)

            retrans_final_msg = (
                retrans_result.record["messages"][-1]["content"]
                if retrans_result.record
                else "Re-transcription complete."
            )
            with open(_retrans_output_file, "w") as _rtf:
                _rtf.write(retrans_final_msg)
                _rtf.write("\n")
            if ctx is not None:
                dawfarm_session.put(_retrans_output_file, _retrans_output_file)

            # Restore correct notes for remaining batches. In real mode the
            # sub-builder's final insert already replaced the timeline MIDI.
            notes = _correct_notes
            _log(f"  re-transcription complete, restored correct notes")

            # Re-render current cumulative with corrected notes (with an
            # explicit render tool_call — a listen without one teaches the
            # model that audio appears for free).
            _retrans_wav = batch_audio_dir / f"batch_{bi}_retranscribed.wav"
            if ctx is None:
                render_cumulative_audio(cumulative, notes, _retrans_wav)
            audio_assets.append(str(_retrans_wav))
            _retrans_render_cmd = _wrap_as_bash(build_reaper_render_snippet(
                out_path=ctx.cw(_retrans_wav) if ctx is not None else str(_retrans_wav)))
            messages.append(_tool_call("Bash", {"command": _retrans_render_cmd}))
            if ctx is not None:
                _rtres = ctx.real_exec(_retrans_render_cmd, "retranscribed render")
                ctx.fetch_wav(ctx.cw(_retrans_wav), _retrans_wav)
                _emit_listen_sequence(
                    messages, audio_assets, _retrans_wav,
                    probe_stdout=_rtres.stdout, display_path=ctx.cw(_retrans_wav),
                )
            else:
                _emit_listen_sequence(
                    messages, audio_assets, _retrans_wav,
                )
            last_batch_audio = _retrans_wav
            pending_check = "Re-transcription confirmed — the melody now matches the target. Continuing with parameter tuning."

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
    _final_clap = None
    if batch_labels:
        _last_clap = [l.get("clap_score_after_batch") for l in batch_labels if l.get("clap_score_after_batch") is not None]
        if _last_clap:
            _final_clap = _last_clap[-1]
    verdict_text = stage2_verdict(
        perceptual_obs=verdict_obs,
        residual_delta_summary=residual_delta_summary,
        path_complete=fully_converged,
        archetype=archetype,
        stage2_server=stage2_server,
        stage2_model=stage2_model,
        final_clap_score=_final_clap,
    )
    _log(f"verdict omni {_time.monotonic()-_vt:.1f}s")
    if pending_check:
        verdict_text = f"{pending_check}\n\n{verdict_text}"
    messages.append({"role": "assistant", "content": verdict_text})

    # ---- USER STEER TURNS (optional post-verdict tweaks) ----
    _steer_rate = float(getattr(args, "steer_rate", 0.0))
    steer_applied = False
    if _steer_rate > 0 and sample_rng.random() < _steer_rate:
        _steer_rng = random.Random(int(args.seed) + sid_seed + 3333)
        _build_steer_turns(
            cumulative=cumulative,
            notes=notes,
            messages=messages,
            audio_assets=audio_assets,
            batch_audio_dir=batch_audio_dir,
            rng=_steer_rng,
            sample_id=sample_id,
            ctx=ctx,
        )
        steer_applied = True

    # ---- Record assembly ----
    diagnosis_subs_mentioned = extract_diagnosis_subsystems_mentioned(diagnosis_text)
    mistake_caught = bool(all_injected_mistakes)

    record = {
        "id": sample_id,
        "task_type": "main",
        "tools": _OC_TOOLS_JSON,
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
            "num_agents": n_agents,
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
            "daw_farm_session": dawfarm_session.name if dawfarm_session is not None else None,
            "determinism_clap": entry.get("determinism_clap"),
        },
    }

    # Extended metadata for production tracking
    n_turns = len(messages)
    n_tool_calls = sum(1 for m in messages if m.get("role") == "tool_call")
    n_audio_clips = len(audio_assets)
    n_batches_applied = len(batch_labels)
    n_search_rounds = rounds_used
    wall_time_s = round(_time.monotonic() - _t0, 2)
    gt_wt_names = gt_names_list
    applied_wt_names = [gt_tuple[oi] for oi in active_oscs if gt_tuple[oi]]

    n_subsystem_batches = sum(1 for l in batch_labels if not l.get("is_correction"))
    record["meta"].update({
        "n_turns": n_turns,
        "n_tool_calls": n_tool_calls,
        "n_audio_clips": n_audio_clips,
        "n_batches_applied": n_batches_applied,
        "n_subsystem_batches": n_subsystem_batches,
        "n_correction_turns": total_correction_turns,
        "n_search_rounds": n_search_rounds,
        "max_mistakes_drawn": max_mistakes_drawn,
        "pre_applied_subsystems": pre_applied_subsystems,
        "retranscribe_after_batch": retranscribe_after_batch,
        "steer_applied": steer_applied,
        "wall_time_s": wall_time_s,
        "gt_wavetable_names": gt_wt_names,
        "applied_wavetable_names": applied_wt_names,
    })

    # Contract assertion (D1 class): every subagent record's opening user text
    # (minus the <audio> tag) must appear verbatim among the main record's task
    # dispatch prompts. Checked record -> dispatch (not the reverse): pool-only
    # search agents are dispatched without keeping a record, and their prompt
    # comes from the same canonical builder. A mismatch means the deployed
    # subagent would receive wording it never saw in training — the failure
    # that broke the first POC.
    _dispatch_prompts = set()
    for _m in messages:
        if _m.get("role") != "tool_call":
            continue
        try:
            _tc = json.loads(_m["content"])
        except Exception:
            continue
        if _tc.get("name") == "task":
            _dispatch_prompts.add((_tc.get("arguments") or {}).get("prompt", "").strip())
    for _r in (list(search_records) + list(judge_records) + list(transcription_records)):
        _msgs = _r.get("messages") or []
        if len(_msgs) < 2:
            continue
        _opener = str(_msgs[1].get("content", "")).replace("<audio>\n", "", 1).strip()
        if _opener not in _dispatch_prompts:
            raise RuntimeError(
                f"{sample_id}: subagent record {_r.get('id')} opener does not match "
                f"any task dispatch prompt (contract drift). opener[:120]="
                f"{_opener[:120]!r}")

    assert_valid_ms_swift_multiturn_record(record)
    return record, search_records, judge_records, transcription_records


# ---------------------------------------------------------------------------
# Setup context (importable by run_sft_production.py)
# ---------------------------------------------------------------------------


def setup_build_context(
    *,
    manifest_path: Path,
    index_npy: Path = Path("outputs/wt_retrieval_baseline/wt_index.npz"),
    index_meta: Path = Path("outputs/wt_retrieval_baseline/wt_index_meta.json"),
    wavetable_lib_path: Path = Path("data/wavetable_lib.json"),
    probe_dir: Path = Path("outputs/agent_sft/candidate_probes"),
    clap_device: str = "cpu",
    clap_cache_path: Path | None = None,
    max_samples: int | None = None,
) -> dict:
    """Load all shared resources needed by build_record().

    Returns a dict with keys that can be unpacked into build_record() calls:
      entries, embedder, shortlist_data, selected_by_name, wavetable_lib,
      index_rows, notes
    """
    entries = load_manifest_entries(manifest_path, max_samples=max_samples or 999_999_999)
    index_rows = load_index_rows(index_meta)
    selected_by_name = select_probe_rows_by_name(index_rows)
    wavetable_lib = load_wavetable_lib(wavetable_lib_path)
    embedder = ClapEmbedder.create(clap_device, cache_path=clap_cache_path)
    shortlist_data = build_clap_shortlist_data(index_npy, index_rows)

    _notes = make_probe_notes("lead", clip_duration_s=10.0)

    if clap_cache_path and len(embedder._cache) > 0:
        print(f"Loaded {len(embedder._cache)} pre-computed CLAP embeddings from {clap_cache_path}", flush=True)

    clap_paths: list[Path] = []
    for e in entries:
        for k in ("gt_wav", "gt_probe_wav", "default_wav"):
            if e.get(k):
                clap_paths.append(Path(e[k]))
    if probe_dir.exists():
        for pp in sorted(probe_dir.glob("*.wav")):
            clap_paths.append(pp)
    clap_paths = list(dict.fromkeys(clap_paths))
    uncached = [p for p in clap_paths if str(p.resolve()) not in embedder._cache]
    if uncached:
        print(f"Computing CLAP embeddings for {len(uncached)} uncached audio files "
              f"(skipping {len(clap_paths) - len(uncached)} already cached)...", flush=True)
        for p in uncached:
            if p.exists():
                try:
                    embedder.embed_audio_path(p)
                except Exception as exc:
                    print(f"  WARNING: CLAP embed failed for {p.name}: {exc}")
        print(f"CLAP pre-computation done ({len(embedder._cache)} cached).", flush=True)
    else:
        print(f"All {len(clap_paths)} audio files already in CLAP cache.", flush=True)

    if clap_device != "cpu":
        try:
            embedder.model = embedder.model.to("cpu")
            embedder.device = "cpu"
            print("CLAP moved to CPU for worker-thread safety.", flush=True)
        except Exception as exc:
            print(f"WARNING: could not move CLAP to CPU: {exc}")

    return {
        "entries": entries,
        "embedder": embedder,
        "shortlist_data": shortlist_data,
        "selected_by_name": selected_by_name,
        "wavetable_lib": wavetable_lib,
        "index_rows": index_rows,
        "notes": _notes,
    }


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
    ap.add_argument("--num-agents", type=int, default=None,
        help="DEPRECATED and ignored: search agent count is derived from "
             "library size (ceil(total/candidates-per-slice)).")

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
    ap.add_argument("--retranscribe-rate", type=float, default=0.0,
        help="Fraction of samples where transcription is re-done mid-conversation after hearing wrong notes with effects.")
    ap.add_argument("--steer-rate", type=float, default=0.0,
        help="Fraction of samples with 1-2 user steer turns after the verdict.")
    ap.add_argument("--random-init-rate", type=float, default=0.0,
        help="Fraction of samples starting from a random same-archetype preset instead of factory default.")
    ap.add_argument("--partial-init-rate", type=float, default=0.0,
        help="Fraction of samples starting with 1-4 GT subsystems pre-applied.")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--clap-device", default="cuda:0")
    ap.add_argument("--search-probe-dir", type=Path, default=None,
                    help="reuse probe wavs from another run (paths key the CLAP cache)")
    ap.add_argument("--clap-cache", type=Path, default=None,
                    help="Pre-computed CLAP cache .npz (from precompute_clap_cache.py)")
    ap.add_argument("--omni-server", default="")
    ap.add_argument("--omni-model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--stage2-server", default="")
    ap.add_argument("--stage2-model", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--daw-farm", default="",
        help="Execute rollout snippets in real daw-farm REAPER sessions. "
             "Spec: docker[:name1,name2] or k8s[:pod1,pod2] (empty names = discover). "
             "Empty flag = simulate tool responses (legacy behaviour).")
    ap.add_argument("--daw-farm-vital-data", default=str(ROOT / "data/prepared/wavetable_lib_vital_dir"),
        help="Host Vital data dir synced into each session (use the generation "
             "library from scripts/export_wavetable_lib_dir.py).")
    ap.add_argument("--search-record-keep-rate", type=float, default=0.25,
        help="Fraction of non-GT-shard search-agent records to emit as SFT "
             "records (the GT shard's record is always kept; dropped shards "
             "run pool-only: identical shortlists, no omni narration). "
             "1.0 = legacy behaviour, every search record emitted.")
    ap.add_argument("--recycle-containers", type=lambda v: v.lower() != "false",
                    default=True,
                    help="docker-restart + assert-clean each container before every sample (default true)")
    ap.add_argument("--merged-stage2", action=argparse.BooleanOptionalAction, default=True,
        help="Search agents: single merged stage-2 call per batch instead of "
             "synthesize+notes. Default on (+32%% throughput, judge-identical "
             "quality); --no-merged-stage2 restores the split calls.")
    ap.add_argument("--daw-farm-timeout", type=float, default=300.0,
        help="Per-snippet exec timeout in daw-farm mode (seconds).")
    args = ap.parse_args()

    # Absolute paths everywhere: agent workdir files are pushed into the
    # container at their host paths, and snippets resolve them in-container —
    # both require absolute paths.
    args.out_dir = args.out_dir.resolve()


    if args.mistake_rate is not None:
        import warnings
        warnings.warn("--mistake-rate is deprecated, use --per-param-mistake-rate", DeprecationWarning)
        if args.per_param_mistake_rate == 0.10:
            args.per_param_mistake_rate = args.mistake_rate

    stage2_server = args.stage2_server or args.omni_server
    stage2_model = args.stage2_model or args.omni_model

    if args.omni_server:
        omni_urls = [u.strip() for u in args.omni_server.split(",") if u.strip()]
        if len(omni_urls) > 1:
            from scripts.build_main_agent_sft_v2 import init_llm_router
            init_llm_router(omni_urls, args.omni_model)
            for u in omni_urls:
                _check_server_reachable(u, f"Omni({u})")
        else:
            _check_server_reachable(omni_urls[0], "Omni")
        if stage2_server and stage2_server != omni_urls[0]:
            _check_server_reachable(stage2_server, "Stage2")

    import time as _wall_time
    _wall_t0 = _wall_time.monotonic()

    ctx = setup_build_context(
        manifest_path=Path(args.manifest),
        index_npy=args.index_npy,
        index_meta=args.index_meta,
        wavetable_lib_path=args.wavetable_lib,
        probe_dir=args.probe_dir,
        clap_device=args.clap_device,
        clap_cache_path=args.clap_cache,
        max_samples=args.max_samples,
    )
    entries = ctx["entries"]
    embedder = ctx["embedder"]
    shortlist_data = ctx["shortlist_data"]
    selected_by_name = ctx["selected_by_name"]
    wavetable_lib = ctx["wavetable_lib"]
    index_rows = ctx["index_rows"]
    _notes = ctx["notes"]

    candidate_audio: dict[str, Path] = {}
    serial_lock = threading.Lock()

    # Accumulate all records
    all_main: list[dict] = []
    all_search: list[dict] = []
    all_judge: list[dict] = []
    all_trans: list[dict] = []

    dawfarm_pool: "_dawfarm.DawFarmPool | None" = None
    if args.daw_farm:
        dawfarm_pool = _dawfarm.DawFarmPool.from_spec(args.daw_farm)
        # Warm the shared candidate-describe probe cache with env-rendered
        # audio (fixed archetype melody) before the worker fan-out.
        from scripts.agent_sft_common import warm_candidate_probe_cache_dawfarm
        _all_names = sorted({wt["name"] for wt in wavetable_lib
                             if isinstance(wt, dict) and wt.get("name")})
        with dawfarm_pool.acquire() as _ws:
            _dawfarm.sync_vital_data(_ws, getattr(args, "daw_farm_vital_data", None))
            _n = warm_candidate_probe_cache_dawfarm(
                _ws, _all_names,
                Path(getattr(args, "probe_dir", "outputs/agent_sft/candidate_probes")),
                candidate_audio)
            print(f"probe cache warmed: {_n} env-rendered, "
                  f"{len(candidate_audio)} total cached", flush=True)
        if len(dawfarm_pool.sessions) < args.workers:
            print(f"NOTE: {args.workers} workers > {len(dawfarm_pool.sessions)} "
                  f"daw-farm sessions — workers will queue for sessions.", flush=True)

    def _process(entry: dict) -> tuple[dict | None, list[dict], list[dict], list[dict]]:
        kwargs = dict(
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
        if dawfarm_pool is not None:
            with dawfarm_pool.acquire() as _sess:
                # Container hygiene (2026-08-15 policy): recycle before every
                # rollout, then assert pristine state — fail loudly if dirty.
                if bool(getattr(args, "recycle_containers", True)):
                    if hasattr(_sess, "recycle"):
                        _sess.recycle()
                    else:
                        _dawfarm.reset_project(_sess)
                        _sess.exec_bash(
                            "rm -rf /tmp/agents /tmp/search_probes /tmp/gate "
                            "&& find /tmp -name 'wt_*.wav' -delete", timeout=60.0)
                    _dawfarm.sync_vital_data(
                        _sess, getattr(args, "daw_farm_vital_data", None))
                _dawfarm.assert_clean(_sess)
                return build_record(dawfarm_session=_sess, **kwargs)
        return build_record(**kwargs)

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
    from scripts.agent_sft_common import format_pipeline_timings
    print(f"  --- aggregate category timings (thread-seconds, all workers) ---", flush=True)
    print(format_pipeline_timings(), flush=True)


if __name__ == "__main__":
    main()
