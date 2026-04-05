#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

_TOOL_SPECS = json.dumps(
    [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute shell/Python commands for Vital search, edit, and listen passes.",
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
                "name": "spawn_search_agents",
                "description": "Fan out disjoint candidate shards to search workers.",
                "parameters": {"type": "object"},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "collect_search_reports",
                "description": "Collect proposals from search workers.",
                "parameters": {"type": "object"},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "judge_candidates",
                "description": "Rank and select up to K candidate IDs in token space.",
                "parameters": {"type": "object"},
            },
        },
    ],
    ensure_ascii=False,
)


def _build_search_reports(
    shards: list[list[str]],
    id_map: dict[str, str],
    gt_names: set[str],
    clap_scores: dict[str, float],
    proposals_per_agent: int,
) -> list[dict]:
    reports: list[dict] = []
    for i, shard in enumerate(shards, start=1):
        if not shard:
            continue
        ranked = sorted(
            shard,
            key=lambda n: (1 if n in gt_names else 0, float(clap_scores.get(n, 0.0))),
            reverse=True,
        )
        selected = ranked[: max(1, int(proposals_per_agent))]
        proposals = []
        for n in selected:
            proposals.append(
                {
                    "candidate_id": id_map[n],
                    "wavetable_name": n,
                    "confidence": round(0.9 if n in gt_names else max(0.35, min(0.88, clap_scores[n])), 3),
                    "reason": (
                        "Likely core source candidate from harmonic structure."
                        if n in gt_names
                        else "Possible match in harmonic envelope and brightness."
                    ),
                }
            )
        reports.append({"agent_id": f"sa_{i}", "proposals": proposals})
    return reports


def _build_judge_result(
    candidate_names: list[str],
    id_map: dict[str, str],
    gt_names: set[str],
    clap_scores: dict[str, float],
    select_k: int,
) -> dict:
    ranked = sorted(
        candidate_names,
        key=lambda n: (1 if n in gt_names else 0, float(clap_scores.get(n, 0.0))),
        reverse=True,
    )
    ranking = [id_map[n] for n in ranked]
    selected = ranking[: max(1, int(select_k))]
    return {
        "ranking": ranking,
        "selected": selected,
        "reason": "Selected highest-plausibility source candidates from aggregated search proposals.",
    }


def _step_commentary(step: dict, step_num: int) -> str:
    keyword = str(step.get("search_keyword") or "target controls")
    primary = str(step.get("primary_family") or "synth")
    support = str(step.get("support_family") or "none")
    return (
        f"HEARD: Step {step_num} still has mismatch in tone and movement.\n\n"
        f"HYPOTHESIS: Primary mismatch is in {primary}, with possible interaction from {support}.\n\n"
        f"PLAN: Apply the programmed updates for {keyword}, then listen again."
    )


def _build_listen_probe_command(audio_path: Path) -> str:
    payload = {"path": str(audio_path)}
    return (
        "python - <<'PY'\n"
        "import json\n"
        "from pathlib import Path\n"
        "import soundfile as sf\n"
        f"payload = json.loads('''{json.dumps(payload, ensure_ascii=False)}''')\n"
        "p = Path(payload['path'])\n"
        "out = {'path': str(p), 'exists': p.exists()}\n"
        "if out['exists']:\n"
        "    try:\n"
        "        x, sr = sf.read(p, always_2d=True)\n"
        "        out['duration_s'] = round(float(len(x) / max(1, sr)), 4)\n"
        "    except Exception:\n"
        "        out['duration_s'] = None\n"
        "print(json.dumps({'listen_probe': out}, ensure_ascii=False))\n"
        "PY"
    )


def _tool_call(name: str, arguments: dict) -> dict:
    return {"role": "tool_call", "content": json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build main-agent SFT dataset v2 with search/judge hierarchy.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--index-npy", type=Path, default=Path("outputs/wt_retrieval_baseline/wt_index.npz"))
    ap.add_argument("--index-meta", type=Path, default=Path("outputs/wt_retrieval_baseline/wt_index_meta.json"))
    ap.add_argument("--wavetable-lib", type=Path, default=Path("data/wavetable_lib.json"))
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--max-samples", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=6)

    ap.add_argument("--candidate-source", choices=["all", "clap_topn", "oracle_mix8"], default="oracle_mix8")
    ap.add_argument("--candidate-limit", type=int, default=8)
    ap.add_argument("--oracle-hard-pool", type=int, default=64)
    ap.add_argument("--num-agents", type=int, default=4)
    ap.add_argument("--proposals-per-agent", type=int, default=3)
    ap.add_argument("--select-k", type=int, default=3)

    ap.add_argument("--probe-archetype", default="lead")
    ap.add_argument("--probe-tail-s", type=float, default=1.0)
    ap.add_argument("--trim-min-duration-s", type=float, default=0.5)
    ap.add_argument("--probe-dir", type=Path, default=Path("outputs/agent_sft/candidate_probes"))

    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--clap-device", default="cuda:0")
    args = ap.parse_args()

    entries = load_manifest_entries(Path(args.manifest), max_samples=args.max_samples)
    index_rows = load_index_rows(args.index_meta)
    selected_by_name = select_probe_rows_by_name(index_rows)
    universe_names = sorted(selected_by_name.keys(), key=lambda x: x.lower())
    wavetable_lib = load_wavetable_lib(args.wavetable_lib)
    embedder = ClapEmbedder.create(args.clap_device)
    shortlist_data = build_clap_shortlist_data(args.index_npy, index_rows)

    candidate_audio: dict[str, Path] = {}
    records: list[dict] = []

    for entry in entries:
        sample_id = str(entry["sample_id"])
        target_audio_path = Path(entry.get("gt_wav") or entry.get("gt_probe_wav"))
        default_audio_path = Path(entry.get("default_wav")) if entry.get("default_wav") else None
        iter_wavs = [Path(p) for p in (entry.get("iter_wavs") or [])]

        with open(Path(entry["path_file"])) as f:
            path_data = json.load(f)
        gt_names = set(extract_gt_wavetable_names(Path(path_data["target_preset_path"])))
        if not gt_names:
            continue

        candidate_names = choose_candidate_pool(
            sample_id=sample_id,
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
            out_dir=args.probe_dir,
            cache=candidate_audio,
            probe_archetype=args.probe_archetype,
            probe_tail_s=args.probe_tail_s,
            trim_min_duration_s=args.trim_min_duration_s,
        )

        clap_scores = {n: embedder.cosine_paths(candidate_audio[n], target_audio_path) for n in candidate_names}
        id_map = {name: f"C{i}" for i, name in enumerate(candidate_names, start=1)}
        candidate_assets = [
            {"candidate_id": id_map[name], "wavetable_name": name, "audio_path": str(candidate_audio[name])}
            for name in candidate_names
        ]

        shards = build_disjoint_shards(candidate_names, args.num_agents)
        jobs = [
            {
                "agent_id": f"sa_{i}",
                "candidate_shard": [id_map[n] for n in shard],
                "seed": int(args.seed) + i,
            }
            for i, shard in enumerate(shards, start=1)
            if shard
        ]
        reports = _build_search_reports(shards, id_map, gt_names, clap_scores, args.proposals_per_agent)
        judge_result = _build_judge_result(candidate_names, id_map, gt_names, clap_scores, args.select_k)
        selected_ids = set(judge_result["selected"])
        selected_names = [name for name in candidate_names if id_map[name] in selected_ids]

        messages: list[dict] = []
        used_iter_audio_paths: list[str] = []
        messages.append(
            {
                "role": "user",
                "content": (
                    f"<audio>\nRecreate this {entry.get('archetype', 'synth')} target sound in Vital from default.\n"
                    "Run hierarchical wavetable search, keep <=3 candidates, then continue iterative edits."
                ),
            }
        )

        if default_audio_path is not None:
            messages.append({"role": "assistant", "content": "Listening to current working preset baseline."})
            messages.append(_tool_call("bash", {"command": _build_listen_probe_command(default_audio_path)}))
            messages.append(
                {
                    "role": "tool_response",
                    "content": json.dumps(
                        {"status": "ok", "baseline_audio": "<audio>", "path": str(default_audio_path)},
                        ensure_ascii=False,
                    ),
                }
            )

        messages.append(
            {
                "role": "assistant",
                "content": "Spawning disjoint search shards to gather wavetable proposals in parallel.",
            }
        )
        messages.append(
            _tool_call(
                "spawn_search_agents",
                {
                    "sample_id": sample_id,
                    "target_audio_path": str(target_audio_path),
                    "current_audio_path": str(default_audio_path) if default_audio_path else None,
                    "candidate_universe": [c["candidate_id"] for c in candidate_assets],
                    "num_agents": int(args.num_agents),
                    "shard_strategy": "disjoint_round_robin",
                    "seed": int(args.seed),
                },
            )
        )
        messages.append({"role": "tool_response", "content": json.dumps({"jobs": jobs}, ensure_ascii=False)})

        messages.append({"role": "assistant", "content": "Collecting search-agent reports."})
        messages.append(_tool_call("collect_search_reports", {"sample_id": sample_id, "jobs": jobs}))
        messages.append({"role": "tool_response", "content": json.dumps({"reports": reports}, ensure_ascii=False)})

        messages.append({"role": "assistant", "content": "Judging candidates and selecting up to three for edits."})
        messages.append(
            _tool_call(
                "judge_candidates",
                {
                    "sample_id": sample_id,
                    "target_audio_path": str(target_audio_path),
                    "candidate_audio": candidate_assets,
                    "max_select": int(args.select_k),
                },
            )
        )
        messages.append({"role": "tool_response", "content": json.dumps(judge_result, ensure_ascii=False)})

        for step in path_data.get("iterations", [])[: int(args.max_steps)]:
            step_num = int(step.get("step", 0))
            action_snippet = step.get("action_snippet") or step.get("python_script") or "print('noop')"
            prefix = (
                "Selected candidates: " + ", ".join(selected_names[: int(args.select_k)]) + ".\n\n"
                if step_num == 1 and selected_names
                else ""
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        prefix
                        + _step_commentary(step, step_num)
                        + f"\n\nExecuting step {step_num} parameter updates now."
                    ),
                }
            )
            messages.append(_tool_call("bash", {"command": action_snippet}))
            messages.append({"role": "tool_response", "content": json.dumps({"status": "ok"}, ensure_ascii=False)})

            if step_num - 1 < len(iter_wavs):
                iter_audio = iter_wavs[step_num - 1]
                used_iter_audio_paths.append(str(iter_audio))
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"Listening to updated preset after step {step_num}.",
                    }
                )
                messages.append(_tool_call("bash", {"command": _build_listen_probe_command(iter_audio)}))
                messages.append(
                    {
                        "role": "tool_response",
                        "content": json.dumps(
                            {"status": "ok", "step": step_num, "iter_audio": "<audio>", "path": str(iter_audio)},
                            ensure_ascii=False,
                        ),
                    }
                )

        messages.append({"role": "assistant", "content": "Recreation pass complete."})

        audio_assets = [str(target_audio_path)]
        if default_audio_path is not None:
            audio_assets.append(str(default_audio_path))
        audio_assets.extend(used_iter_audio_paths)

        record = {
            "id": sample_id,
            "task_type": "main",
            "tools": _TOOL_SPECS,
            "messages": messages,
            "audios": audio_assets,
            "assets": {
                "target_audio": str(target_audio_path),
                "current_audio": str(default_audio_path) if default_audio_path else None,
                "candidate_audio": candidate_assets,
                "selected_candidates": [
                    {"candidate_id": id_map[n], "wavetable_name": n, "audio_path": str(candidate_audio[n])}
                    for n in selected_names[: int(args.select_k)]
                ],
            },
            "labels": {
                "judge_ranking": judge_result["ranking"],
                "judge_selected": judge_result["selected"],
                "gt_candidate_ids": [id_map[n] for n in candidate_names if n in gt_names],
            },
            "meta": {
                "sample_id": sample_id,
                "archetype": str(entry.get("archetype", "synth")),
                "agent": "main",
                "num_agents": int(args.num_agents),
                "candidate_source": args.candidate_source,
                "max_steps": int(args.max_steps),
            },
        }
        assert_valid_ms_swift_multiturn_record(record)
        records.append(record)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} main-agent v2 records to {out_path}")


if __name__ == "__main__":
    main()
