#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maestro.render.vital import SAMPLE_RATE, _load_vital, _render_note_list, make_probe_notes, trim_silence
from scripts.agent_sft_common import (
    ClapEmbedder,
    assert_valid_ms_swift_multiturn_record,
    build_clap_shortlist_data,
    build_disjoint_shards,
    choose_candidate_pool,
    ensure_candidate_probes_for_names,
    extract_gt_wavetable_names,
    load_index_rows,
    load_manifest_entries,
    load_wavetable_lib,
    select_probe_rows_by_name,
)
from scripts.build_wavetable_retrieval_baseline import _build_probe_preset, _load_init_preset

_BASH_TOOL_SPEC = json.dumps(
    [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute shell/Python commands for parameter search, rendering, and audio audition.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to execute.",
                        }
                    },
                    "required": ["command"],
                },
            },
        }
    ],
    ensure_ascii=False,
)


def _step_selected(step: dict, family_filter: set[str]) -> bool:
    if not step.get("search_keyword"):
        return False
    if not family_filter:
        return True
    return str(step.get("primary_family") or "").strip().lower() in family_filter


def _build_search_tool_command(
    target_audio_path: Path,
    current_audio_path: Path | None,
    source_midi_path: Path | None,
    shard_assets: list[dict[str, str]],
    wavetable_lib_path: Path,
    probe_archetype: str,
    probe_tail_s: float,
    trim_min_duration_s: float,
) -> str:
    payload = {
        "target_audio": str(target_audio_path),
        "current_audio": str(current_audio_path) if current_audio_path else None,
        "source_midi": str(source_midi_path) if source_midi_path else None,
        "wavetable_lib": str(wavetable_lib_path),
        "probe_archetype": probe_archetype,
        "probe_tail_s": float(probe_tail_s),
        "trim_min_duration_s": float(trim_min_duration_s),
        "candidates": [
            {
                "candidate_id": item["candidate_id"],
                "wavetable_name": item["wavetable_name"],
                "source_wavetable_idx": int(item["source_wavetable_idx"]),
                "frame_idx": int(item["frame_idx"]),
                "render_path": item["render_path"],
            }
            for item in shard_assets
        ],
    }
    return (
        "python - <<'PY'\n"
        "import json\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import soundfile as sf\n"
        "import pretty_midi\n"
        "from maestro.render.vital import SAMPLE_RATE, _load_vital, _render_note_list, make_probe_notes, trim_silence\n"
        "from scripts.build_wavetable_retrieval_baseline import _build_probe_preset, _load_init_preset, _load_wavetable_lib\n"
        "\n"
        f"payload = json.loads('''{json.dumps(payload, ensure_ascii=False)}''')\n"
        "target, _ = sf.read(payload['target_audio'], always_2d=True)\n"
        "target = target.mean(axis=1).astype(np.float32)\n"
        "wavetable_lib = _load_wavetable_lib(Path(payload['wavetable_lib']))\n"
        "init_preset = _load_init_preset()\n"
        "notes = None\n"
        "if payload.get('source_midi'):\n"
        "    try:\n"
        "        pm = pretty_midi.PrettyMIDI(payload['source_midi'])\n"
        "        non_drum = [i for i in pm.instruments if not i.is_drum and i.notes]\n"
        "        inst = max(non_drum, key=lambda x: len(x.notes)) if non_drum else None\n"
        "        if inst is None:\n"
        "            with_notes = [i for i in pm.instruments if i.notes]\n"
        "            inst = max(with_notes, key=lambda x: len(x.notes)) if with_notes else None\n"
        "        if inst is not None:\n"
        "            notes = sorted(inst.notes, key=lambda n: (n.start, n.pitch))\n"
        "    except Exception:\n"
        "        notes = None\n"
        "if not notes:\n"
        "    notes = make_probe_notes(payload['probe_archetype'])\n"
        "synth = _load_vital()\n"
        "\n"
        "def cos(a,b):\n"
        "    m=min(len(a),len(b))\n"
        "    if m<=0:\n"
        "        return 0.0\n"
        "    aa=a[:m]\n"
        "    bb=b[:m]\n"
        "    an=np.linalg.norm(aa)+1e-12\n"
        "    bn=np.linalg.norm(bb)+1e-12\n"
        "    return float(np.dot(aa,bb)/(an*bn))\n"
        "\n"
        "rows=[]\n"
        "for c in payload['candidates']:\n"
        "    wt = wavetable_lib[int(c['source_wavetable_idx'])]\n"
        "    preset = _build_probe_preset(init_preset, wt, int(c['frame_idx']))\n"
        "    synth.load_json(json.dumps(preset))\n"
        "    audio = _render_note_list(synth, notes, SAMPLE_RATE, tail_s=float(payload['probe_tail_s']))\n"
        "    audio = trim_silence(audio, SAMPLE_RATE, min_duration_s=float(payload['trim_min_duration_s']))\n"
        "    wav = Path(c['render_path'])\n"
        "    wav.parent.mkdir(parents=True, exist_ok=True)\n"
        "    sf.write(wav, audio.T, SAMPLE_RATE)\n"
        "    x=audio.mean(axis=0).astype(np.float32)\n"
        "    s=cos(x,target)\n"
        "    rows.append((s,c['candidate_id'],c['wavetable_name'],str(wav)))\n"
        "rows.sort(reverse=True)\n"
        "for s,cid,name,wav in rows:\n"
        "    print(f\"{cid}\\t{name}\\t{s:.4f}\\t{wav}\")\n"
        "PY"
    )


def _reason_for_cosine(score: float, include_similarity_scores: bool = True) -> str:
    conf = 0.5 * (score + 1.0)
    if conf >= 0.75:
        return (
            f"Strong waveform match ({score:.3f}); envelope and harmonic profile follow target closely."
            if include_similarity_scores
            else "Strong waveform match; envelope and harmonic profile follow target closely."
        )
    if conf >= 0.60:
        return (
            f"Moderate waveform match ({score:.3f}); similar harmonic structure but texture differs."
            if include_similarity_scores
            else "Moderate waveform match; similar harmonic structure but texture differs."
        )
    return (
        f"Weak waveform match ({score:.3f}); spectral profile diverges but included for shard coverage."
        if include_similarity_scores
        else "Weak waveform match; spectral profile diverges but included for shard coverage."
    )


def _build_assistant_proposals(
    shard_names: list[str],
    gt_names: set[str],
    id_map: dict[str, str],
    score_map: dict[str, float],
    proposals_per_agent: int,
    include_similarity_scores: bool = True,
) -> tuple[list[str], list[dict]]:
    ranked = sorted(
        shard_names,
        key=lambda n: float(score_map.get(n, 0.0)),
        reverse=True,
    )
    selected = ranked[: max(1, int(proposals_per_agent))]

    proposals = []
    for n in selected:
        score = float(score_map.get(n, 0.0))
        conf = max(0.05, min(0.99, 0.5 * (score + 1.0)))
        proposals.append(
            {
                "candidate_id": id_map[n],
                "wavetable_name": n,
                "confidence": round(conf, 3),
                "reason": _reason_for_cosine(score, include_similarity_scores=include_similarity_scores),
            }
        )
    return [id_map[n] for n in ranked], proposals


def _build_audio_tags(n: int) -> str:
    return "\n".join(["<audio>"] * max(0, int(n)))


def _build_audition_bundle_tool_command(
    paths: list[str],
    target_path: str,
    include_similarity_scores: bool = True,
) -> str:
    payload = {"paths": paths, "target": target_path}
    sim_line = (
        "            item['cosine_vs_target'] = round(_cos(x, tgt), 4)\n"
        if include_similarity_scores
        else ""
    )
    sim_none_line = (
        "            item['cosine_vs_target'] = None\n"
        if include_similarity_scores
        else ""
    )
    return (
        "python - <<'PY'\n"
        "import json, numpy as np\n"
        "from pathlib import Path\n"
        "import soundfile as sf\n"
        f"payload = json.loads('''{json.dumps(payload, ensure_ascii=False)}''')\n"
        "def _cos(a, b):\n"
        "    m = min(len(a), len(b))\n"
        "    if m <= 0: return 0.0\n"
        "    aa, bb = a[:m], b[:m]\n"
        "    return float(np.dot(aa, bb) / (np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-12))\n"
        "tgt, _ = sf.read(payload['target'], always_2d=True)\n"
        "tgt = tgt.mean(axis=1).astype(np.float32)\n"
        "bundle = []\n"
        "for p in payload['paths']:\n"
        "    item = {'path': p, 'exists': Path(p).exists()}\n"
        "    if item['exists']:\n"
        "        try:\n"
        "            x, sr = sf.read(p, always_2d=True)\n"
        "            x = x.mean(axis=1).astype(np.float32)\n"
        "            item['duration_s'] = round(float(len(x) / max(1, sr)), 4)\n"
        f"{sim_line}"
        "        except Exception:\n"
        "            item['duration_s'] = None\n"
        f"{sim_none_line}"
        "    bundle.append(item)\n"
        "print(json.dumps({'audition_bundle': bundle}, ensure_ascii=False))\n"
        "PY"
    )


def _load_notes_for_render(source_midi_path: Path | None, probe_archetype: str):
    if source_midi_path and source_midi_path.exists():
        try:
            pm = pretty_midi.PrettyMIDI(str(source_midi_path))
            non_drum = [i for i in pm.instruments if not i.is_drum and i.notes]
            inst = max(non_drum, key=lambda x: len(x.notes)) if non_drum else None
            if inst is None:
                with_notes = [i for i in pm.instruments if i.notes]
                inst = max(with_notes, key=lambda x: len(x.notes)) if with_notes else None
            if inst is not None:
                return sorted(inst.notes, key=lambda n: (n.start, n.pitch))
        except Exception:
            pass
    return make_probe_notes(probe_archetype)


def _ensure_rendered_candidate_audio(
    *,
    shard_assets: list[dict[str, str]],
    wavetable_lib: list[dict],
    notes,
    probe_tail_s: float,
    trim_min_duration_s: float,
    synth,
    init_preset: dict,
) -> None:
    for row in shard_assets:
        out = Path(row["render_path"])
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        wt = wavetable_lib[int(row["source_wavetable_idx"])]
        preset = _build_probe_preset(init_preset, wt, int(row["frame_idx"]))
        synth.load_json(json.dumps(preset))
        audio = _render_note_list(synth, notes, SAMPLE_RATE, tail_s=float(probe_tail_s))
        audio = trim_silence(audio, SAMPLE_RATE, min_duration_s=float(trim_min_duration_s))
        sf.write(str(out), audio.T, SAMPLE_RATE)


def _wav_mono(path: Path) -> np.ndarray:
    x, _ = sf.read(path, always_2d=True)
    return x.mean(axis=1).astype(np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    m = min(len(a), len(b))
    if m <= 0:
        return 0.0
    aa = a[:m]
    bb = b[:m]
    an = np.linalg.norm(aa) + 1e-12
    bn = np.linalg.norm(bb) + 1e-12
    return float(np.dot(aa, bb) / (an * bn))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build search-agent SFT dataset (wavetable proposal shards).")
    ap.add_argument("--manifest", required=True, help="Path to manifest.jsonl with path_file fields.")
    ap.add_argument("--index-npy", type=Path, default=Path("outputs/wt_retrieval_baseline/wt_index.npz"))
    ap.add_argument("--index-meta", type=Path, default=Path("outputs/wt_retrieval_baseline/wt_index_meta.json"))
    ap.add_argument("--wavetable-lib", type=Path, default=Path("data/wavetable_lib.json"))
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--max-samples", type=int, default=256)
    ap.add_argument("--max-steps-per-sample", type=int, default=2)
    ap.add_argument("--step-family-filter", default="osc", help="Comma-separated primary families to include.")

    ap.add_argument("--candidate-source", choices=["all", "clap_topn", "oracle_mix8"], default="oracle_mix8")
    ap.add_argument("--candidate-limit", type=int, default=24)
    ap.add_argument("--oracle-hard-pool", type=int, default=64)
    ap.add_argument("--num-agents", type=int, default=4)
    ap.add_argument("--proposals-per-agent", type=int, default=3)
    ap.add_argument("--listen-top-k", type=int, default=2)

    ap.add_argument("--probe-archetype", default="lead")
    ap.add_argument("--probe-tail-s", type=float, default=1.0)
    ap.add_argument("--trim-min-duration-s", type=float, default=0.5)
    ap.add_argument("--probe-dir", type=Path, default=Path("outputs/agent_sft/candidate_probes"))
    ap.add_argument("--search-render-dir", type=Path, default=Path("outputs/agent_sft/search_agent_renders"))

    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--clap-device", default="cuda:0")
    ap.add_argument(
        "--hide-similarity-scores",
        action="store_true",
        help="Hide numeric similarity fields from model-visible tool responses/reasons.",
    )
    args = ap.parse_args()
    expose_similarity_scores = not bool(args.hide_similarity_scores)

    family_filter = {t.strip().lower() for t in str(args.step_family_filter).split(",") if t.strip()}
    probe_dir_abs = args.probe_dir.resolve()
    search_render_dir_abs = args.search_render_dir.resolve()
    wavetable_lib_abs = args.wavetable_lib.resolve()

    manifest_entries = load_manifest_entries(Path(args.manifest), max_samples=args.max_samples)
    index_rows = load_index_rows(args.index_meta)
    selected_by_name = select_probe_rows_by_name(index_rows)
    universe_names = sorted(selected_by_name.keys(), key=lambda x: x.lower())
    wavetable_lib = load_wavetable_lib(args.wavetable_lib)
    embedder = ClapEmbedder.create(args.clap_device)
    shortlist_data = build_clap_shortlist_data(args.index_npy, index_rows)
    synth = _load_vital()
    init_preset = _load_init_preset()

    candidate_audio: dict[str, Path] = {}
    records: list[dict] = []

    for entry in manifest_entries:
        sample_id = str(entry["sample_id"])
        target_audio_path = Path(entry.get("gt_wav") or entry.get("gt_probe_wav"))
        current_audio_path = Path(entry.get("default_wav")) if entry.get("default_wav") else None

        path_file = Path(entry["path_file"])
        with open(path_file) as f:
            path_data = json.load(f)
        source_midi_raw = entry.get("source_midi_path") or path_data.get("source_midi_path")
        source_midi_path = Path(source_midi_raw) if source_midi_raw else None

        gt_names = set(extract_gt_wavetable_names(Path(path_data["target_preset_path"])))
        if not gt_names:
            continue

        used_steps = 0
        for step in path_data.get("iterations", []):
            if used_steps >= int(args.max_steps_per_sample):
                break
            if not _step_selected(step, family_filter):
                continue

            step_num = int(step.get("step", used_steps + 1))
            iter_wavs = entry.get("iter_wavs") or []
            if step_num - 1 < len(iter_wavs):
                current_audio_path = Path(iter_wavs[step_num - 1])

            candidate_names = choose_candidate_pool(
                sample_id=f"{sample_id}_step{step_num}",
                query_audio_path=target_audio_path,
                gt_names=sorted(gt_names),
                universe_names=universe_names,
                candidate_source=args.candidate_source,
                candidate_limit=int(args.candidate_limit),
                oracle_hard_pool=int(args.oracle_hard_pool),
                seed=int(args.seed),
                clap_embedder=embedder,
                selected_rows_meta=index_rows,
                shortlist_data=shortlist_data,
            )
            if not candidate_names:
                continue

            ensure_candidate_probes_for_names(
                names=candidate_names,
                wavetable_lib=wavetable_lib,
                selected_rows=selected_by_name,
                out_dir=probe_dir_abs,
                cache=candidate_audio,
                probe_archetype=args.probe_archetype,
                probe_tail_s=args.probe_tail_s,
                trim_min_duration_s=args.trim_min_duration_s,
            )

            id_map = {name: f"C{i}" for i, name in enumerate(candidate_names, start=1)}
            candidate_assets_full = [
                {
                    "candidate_id": id_map[name],
                    "wavetable_name": name,
                    "audio_path": str(candidate_audio[name]),
                    "source_wavetable_idx": int(selected_by_name[name]["source_wavetable_idx"]),
                    "frame_idx": int(selected_by_name[name].get("frame_idx", 0)),
                }
                for name in candidate_names
            ]

            clap_scores = {
                n: embedder.cosine_paths(candidate_audio[n], target_audio_path)
                for n in candidate_names
            }

            shards = build_disjoint_shards(candidate_names, args.num_agents)
            for agent_idx, shard in enumerate(shards, start=1):
                if not shard:
                    continue
                shard_assets = [
                    {
                        "candidate_id": id_map[name],
                        "wavetable_name": name,
                        "audio_path": str(candidate_audio[name]),
                        "source_wavetable_idx": int(selected_by_name[name]["source_wavetable_idx"]),
                        "frame_idx": int(selected_by_name[name].get("frame_idx", 0)),
                        "render_path": str(
                            search_render_dir_abs
                            / f"{sample_id}_s{step_num:02d}_a{agent_idx:02d}"
                            / f"{id_map[name]}_{name.replace('/', '_')}.wav"
                        ),
                    }
                    for name in shard
                ]
                notes_for_render = _load_notes_for_render(
                    source_midi_path=source_midi_path,
                    probe_archetype=str(entry.get("archetype", args.probe_archetype)),
                )
                _ensure_rendered_candidate_audio(
                    shard_assets=shard_assets,
                    wavetable_lib=wavetable_lib,
                    notes=notes_for_render,
                    probe_tail_s=float(args.probe_tail_s),
                    trim_min_duration_s=float(args.trim_min_duration_s),
                    synth=synth,
                    init_preset=init_preset,
                )
                target_mono = _wav_mono(target_audio_path)
                shard_scores_by_name: dict[str, float] = {}
                for row in shard_assets:
                    wav_path = Path(row["render_path"])
                    score = _cosine(_wav_mono(wav_path), target_mono)
                    shard_scores_by_name[row["wavetable_name"]] = score
                ranked_ids, proposals = _build_assistant_proposals(
                    shard,
                    gt_names,
                    id_map,
                    shard_scores_by_name,
                    args.proposals_per_agent,
                    include_similarity_scores=expose_similarity_scores,
                )
                selected_ids = [p["candidate_id"] for p in proposals]
                gt_candidate_ids = [id_map[n] for n in shard if n in gt_names]
                tool_rows = []
                for cid in ranked_ids:
                    name = next((n for n in shard if id_map[n] == cid), "")
                    if not name:
                        continue
                    render_path = next(
                        (item["render_path"] for item in shard_assets if item["candidate_id"] == cid),
                        "",
                    )
                    if expose_similarity_scores:
                        tool_rows.append(f"{cid}\t{name}\t{float(shard_scores_by_name.get(name, 0.0)):.4f}\t{render_path}")
                    else:
                        tool_rows.append(f"{cid}\t{name}\t{render_path}")
                tool_result = "\n".join(tool_rows)
                tool_call_id = f"tc_search_{sample_id}_s{step_num:02d}_a{agent_idx:02d}"
                tool_cmd = _build_search_tool_command(
                    target_audio_path=target_audio_path,
                    current_audio_path=current_audio_path,
                    source_midi_path=source_midi_path,
                    shard_assets=shard_assets,
                    wavetable_lib_path=wavetable_lib_abs,
                    probe_archetype=str(entry.get("archetype", args.probe_archetype)),
                    probe_tail_s=float(args.probe_tail_s),
                    trim_min_duration_s=float(args.trim_min_duration_s),
                )
                listen_ids = ranked_ids[: max(1, int(args.listen_top_k))]
                listen_rows = [
                    row for row in shard_assets if row["candidate_id"] in set(listen_ids)
                ]
                listen_rows = sorted(listen_rows, key=lambda x: listen_ids.index(x["candidate_id"]))
                listened_audio_paths = [row["render_path"] for row in listen_rows]

                candidate_lines = [
                    f"- {row['candidate_id']}: {row['wavetable_name']}"
                    for row in shard_assets
                ]
                user_text = (
                    f"{_build_audio_tags(1)}\n"
                    + f"Search task for sample={sample_id}, step={step_num}, agent=sa_{agent_idx}/{len(shards)}.\n"
                    + "Audio mapping: this clip is the target reference.\n"
                    + ("Source MIDI: " + str(source_midi_path) + "\n" if source_midi_path else "")
                    + "Candidate shard:\n"
                    + "\n".join(candidate_lines)
                    + "\nRun tool-based shard search, then audition top candidates, then return proposals JSON."
                )

                search_tool_payload = json.dumps(
                    {"name": "bash", "arguments": {"command": tool_cmd}},
                    ensure_ascii=False,
                )
                search_tool_response = json.dumps({"stdout": tool_result}, ensure_ascii=False)
                audition_tool_payload = json.dumps(
                    {
                        "name": "bash",
                        "arguments": {
                            "command": _build_audition_bundle_tool_command(
                                listened_audio_paths,
                                str(target_audio_path),
                                include_similarity_scores=expose_similarity_scores,
                            )
                        },
                    },
                    ensure_ascii=False,
                )
                audition_tool_response = json.dumps(
                    {
                        "audition_candidates": [
                            {
                                "candidate_id": row["candidate_id"],
                                "wavetable_name": row["wavetable_name"],
                                "path": row["render_path"],
                            }
                            for row in listen_rows
                        ]
                    },
                    ensure_ascii=False,
                )

                messages = [{"role": "user", "content": user_text}]
                if current_audio_path:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": (
                                "<audio>\n"
                                "Baseline reference: this is the current working clip before applying new candidate choices.\n"
                                "I will run shard search now, inspect candidate scores, and then audition top candidates."
                            ),
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": "I will run shard search now, inspect candidate scores, and then audition top candidates.",
                        }
                    )
                messages.extend(
                    [
                        {"role": "tool_call", "content": search_tool_payload},
                        {"role": "tool_response", "content": search_tool_response},
                        {
                            "role": "assistant",
                            "content": (
                                "Search results are in. I will audition the top candidates "
                                f"({', '.join(listen_ids)}) before finalizing proposals."
                            ),
                        },
                        {"role": "tool_call", "content": audition_tool_payload},
                        # Single tool_response with all auditioned candidates bundled.
                        # Each entry carries one <audio> tag matched to listened_audio_paths in order.
                        {
                            "role": "tool_response",
                            "content": json.dumps(
                                {
                                    "status": "ok",
                                    "auditioned": len(listen_rows),
                                    "audition_results": [
                                        (
                                            {
                                                "candidate_id": row["candidate_id"],
                                                "wavetable_name": row["wavetable_name"],
                                                "audio_preview": "<audio>",
                                                "cosine_vs_target": round(float(shard_scores_by_name.get(row["wavetable_name"], 0.0)), 4),
                                            }
                                            if expose_similarity_scores
                                            else {
                                                "candidate_id": row["candidate_id"],
                                                "wavetable_name": row["wavetable_name"],
                                                "audio_preview": "<audio>",
                                            }
                                        )
                                        for row in listen_rows
                                    ],
                                },
                                ensure_ascii=False,
                            ),
                        },
                    ]
                )
                messages.append({"role": "assistant", "content": json.dumps({"proposals": proposals}, ensure_ascii=False)})

                audios = [
                    str(target_audio_path),
                    *([str(current_audio_path)] if current_audio_path else []),
                    *listened_audio_paths,
                ]

                record = {
                    "id": f"{sample_id}_step{step_num:02d}_sa{agent_idx:02d}",
                    "task_type": "search",
                    "tools": _BASH_TOOL_SPEC,
                    "messages": messages,
                    "audios": audios,
                    "assets": {
                        "target_audio": str(target_audio_path),
                        "current_audio": str(current_audio_path) if current_audio_path else None,
                        "source_midi_path": str(source_midi_path) if source_midi_path else None,
                        "candidate_audio": shard_assets,
                    },
                    "labels": {
                        "ranking": ranked_ids,
                        "selected": selected_ids,
                        "gt_candidate_ids": gt_candidate_ids,
                    },
                    "meta": {
                        "sample_id": sample_id,
                        "archetype": str(entry.get("archetype", "synth")),
                        "step": step_num,
                        "agent_id": f"sa_{agent_idx}",
                        "num_agents": int(args.num_agents),
                        "search_keyword": str(step.get("search_keyword") or ""),
                        "primary_family": str(step.get("primary_family") or ""),
                        "support_family": str(step.get("support_family") or ""),
                        "candidate_source": args.candidate_source,
                        "score_visibility": "hidden" if not expose_similarity_scores else "explicit",
                    },
                }
                assert_valid_ms_swift_multiturn_record(record)
                records.append(record)

            used_steps += 1

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} search records to {out_path}")


if __name__ == "__main__":
    main()
