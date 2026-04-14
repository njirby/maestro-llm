#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.agent_sft_common import (
    ClapEmbedder,
    assert_valid_ms_swift_multiturn_record,
    build_clap_shortlist_data,
    choose_candidate_pool,
    ensure_candidate_probes_for_names,
    extract_gt_wavetable_names,
    load_index_rows,
    load_manifest_entries,
    load_wavetable_lib,
    select_probe_rows_by_name,
)

_BASH_TOOL_SPEC = json.dumps(
    [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute shell/Python commands for candidate inspection and audition bookkeeping.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to execute."}
                    },
                    "required": ["command"],
                },
            },
        }
    ],
    ensure_ascii=False,
)


def _build_rank_labels(
    ordered_candidate_names: list[str],
    gt_names: set[str],
    clap_scores: dict[str, float],
    topk_select: int,
) -> tuple[list[str], list[str], list[str], dict[str, float]]:
    id_map = {name: f"C{i}" for i, name in enumerate(ordered_candidate_names, start=1)}
    ranked_names = sorted(
        ordered_candidate_names,
        key=lambda n: (1 if n in gt_names else 0, float(clap_scores.get(n, 0.0))),
        reverse=True,
    )
    ranking_ids = [id_map[n] for n in ranked_names]
    selected_ids = ranking_ids[: max(1, int(topk_select))]
    gt_ids = [id_map[n] for n in ordered_candidate_names if n in gt_names]
    score_map = {id_map[n]: float(clap_scores.get(n, 0.0)) for n in ordered_candidate_names}
    return ranking_ids, selected_ids, gt_ids, score_map


def _build_audition_bundle_tool_command(
    candidate_assets: list[dict[str, str]],
    target_audio_path: str,
    include_similarity_scores: bool = True,
) -> str:
    payload = {"candidates": candidate_assets, "target": target_audio_path}
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
        "for c in payload['candidates']:\n"
        "    p = c['audio_path']\n"
        "    item = {\n"
        "        'candidate_id': c['candidate_id'],\n"
        "        'wavetable_name': c['wavetable_name'],\n"
        "        'path': p,\n"
        "        'exists': Path(p).exists(),\n"
        "    }\n"
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Build judge-agent SFT dataset (listwise candidate ranking).")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--index-npy", type=Path, default=Path("outputs/wt_retrieval_baseline/wt_index.npz"))
    ap.add_argument("--index-meta", type=Path, default=Path("outputs/wt_retrieval_baseline/wt_index_meta.json"))
    ap.add_argument("--wavetable-lib", type=Path, default=Path("data/wavetable_lib.json"))
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--max-samples", type=int, default=256)

    ap.add_argument("--candidate-source", choices=["all", "clap_topn", "oracle_mix8"], default="oracle_mix8")
    ap.add_argument("--candidate-limit", type=int, default=8)
    ap.add_argument("--oracle-hard-pool", type=int, default=64)
    ap.add_argument("--select-k", type=int, default=3)
    ap.add_argument("--permutations", type=int, default=3)

    ap.add_argument("--probe-archetype", default="lead")
    ap.add_argument("--probe-tail-s", type=float, default=1.0)
    ap.add_argument("--trim-min-duration-s", type=float, default=0.5)
    ap.add_argument("--probe-dir", type=Path, default=Path("outputs/agent_sft/candidate_probes"))

    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--clap-device", default="cuda:0")
    ap.add_argument(
        "--hide-similarity-scores",
        action="store_true",
        help="Hide numeric similarity fields from model-visible tool responses/reasons.",
    )
    args = ap.parse_args()
    expose_similarity_scores = not bool(args.hide_similarity_scores)

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
        current_audio_path = Path(entry.get("default_wav")) if entry.get("default_wav") else None

        path_file = Path(entry["path_file"])
        with open(path_file) as f:
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

        group_id = f"{sample_id}_judge_group"
        base_seed = (
            int(args.seed) + int(sample_id[-8:], 16)
            if all(c in "0123456789abcdef" for c in sample_id[-8:].lower())
            else int(args.seed)
        )
        for perm_idx in range(max(1, int(args.permutations))):
            rng = random.Random(base_seed + perm_idx)
            ordered = list(candidate_names)
            rng.shuffle(ordered)

            ranking_ids, selected_ids, gt_ids, score_map = _build_rank_labels(
                ordered,
                gt_names,
                clap_scores,
                args.select_k,
            )

            candidate_assets: list[dict[str, str]] = []
            for i, name in enumerate(ordered, start=1):
                candidate_assets.append(
                    {
                        "candidate_id": f"C{i}",
                        "wavetable_name": name,
                        "audio_path": str(candidate_audio[name]),
                    }
                )

            user_text = (
                f"<audio>\nJudge task for sample={sample_id}. This clip is the target reference.\n"
                f"Rank candidates C1..C{len(candidate_assets)} by source plausibility and select up to "
                f"{int(args.select_k)} final candidates."
            )
            messages: list[dict] = [{"role": "user", "content": user_text}]
            if current_audio_path is not None:
                messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "<audio>\nThis is the current working preset. "
                            "I will inspect and audition candidate clips before ranking."
                        ),
                    }
                )
            else:
                messages.append(
                    {
                        "role": "assistant",
                        "content": "I will inspect and audition candidate clips before ranking.",
                    }
                )

            tool_call_id = f"tc_judge_audition_{sample_id}_{perm_idx:02d}"
            tool_call_payload = json.dumps(
                {
                    "name": "bash",
                    "arguments": {
                        "command": _build_audition_bundle_tool_command(
                            candidate_assets,
                            str(target_audio_path),
                            include_similarity_scores=expose_similarity_scores,
                        )
                    },
                    "id": tool_call_id,
                },
                ensure_ascii=False,
            )
            # Single tool_response with all candidate audio previews bundled.
            # Each entry carries one <audio> tag; MS-Swift matches them to the audios list in order.
            audition_results = [
                (
                    {
                        "candidate_id": c["candidate_id"],
                        "wavetable_name": c["wavetable_name"],
                        "audio_preview": "<audio>",
                        "cosine_vs_target": round(float(clap_scores.get(c["wavetable_name"], 0.0)), 4),
                    }
                    if expose_similarity_scores
                    else {
                        "candidate_id": c["candidate_id"],
                        "wavetable_name": c["wavetable_name"],
                        "audio_preview": "<audio>",
                    }
                )
                for c in candidate_assets
            ]
            tool_response_payload = json.dumps(
                {"audition_results": audition_results}, ensure_ascii=False
            )
            messages.append({"role": "tool_call", "content": tool_call_payload})
            messages.append({"role": "tool_response", "content": tool_response_payload})

            # Build a score-informed reason referencing the actual CLAP scores so the
            # model learns to ground rankings in audible similarity values.
            if expose_similarity_scores:
                top_scored = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
                score_summary = ", ".join(f"{cid} ({s:.3f})" for cid, s in top_scored[:4])
                gt_note = (
                    f" GT candidate: {gt_ids[0]}." if gt_ids else ""
                )
                rank_reason = (
                    f"Ranked by CLAP similarity to target. "
                    f"Top scores: {score_summary}.{gt_note} "
                    f"Selected top {len(selected_ids)}."
                )
            else:
                rank_reason = (
                    "Ranked by audible source plausibility from candidate previews. "
                    f"Selected top {len(selected_ids)}."
                )
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "ranking": ranking_ids,
                            "selected": selected_ids,
                            "reason": rank_reason,
                        },
                        ensure_ascii=False,
                    ),
                }
            )

            audios = [
                str(target_audio_path),
                *([str(current_audio_path)] if current_audio_path is not None else []),
                *[c["audio_path"] for c in candidate_assets],
            ]

            record = {
                "id": f"{sample_id}_judge_perm{perm_idx:02d}",
                "task_type": "judge",
                "tools": _BASH_TOOL_SPEC,
                "messages": messages,
                "audios": audios,
                "assets": {
                    "target_audio": str(target_audio_path),
                    "current_audio": str(current_audio_path) if current_audio_path else None,
                    "candidate_audio": candidate_assets,
                },
                "labels": {
                    "ranking": ranking_ids,
                    "selected": selected_ids,
                    "gt_candidate_ids": gt_ids,
                    "scores": score_map,
                },
                "meta": {
                    "sample_id": sample_id,
                    "archetype": str(entry.get("archetype", "synth")),
                    "permutation_group_id": group_id,
                    "permutation_idx": perm_idx,
                    "candidate_source": args.candidate_source,
                    "select_k": int(args.select_k),
                    "score_visibility": "hidden" if not expose_similarity_scores else "explicit",
                },
            }
            assert_valid_ms_swift_multiturn_record(record)
            records.append(record)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} judge records to {out_path}")


if __name__ == "__main__":
    main()
