#!/usr/bin/env python3
"""
Evaluate <=3 wavetable selection using metric deltas (no Omni model).

Per sample:
1) Build candidate pool (e.g., oracle_mix8 with GT guaranteed in pool).
2) Compute baseline similarity S(default_wav, target_wav).
3) For each candidate probe wavetable, compute S(candidate_probe, target_wav).
4) Delta = S(candidate) - S(default).
5) Select <=3 candidates with largest positive deltas.

Outputs per-sample rows + summary and optional audio spotchecks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maestro.render.vital import SAMPLE_RATE, _load_vital, _render_note_list, make_probe_notes, trim_silence
from scripts.build_wavetable_retrieval_baseline import (
    _collapse_row_scores_to_name_scores,
    _embed_clap,
    _extract_gt_wavetable_names,
    _iter_manifest_rows,
    _load_clap,
    _load_init_preset,
    _load_wavetable_lib,
    _resolve_maybe_relative,
    _resolve_target_preset_path,
    _top_name_scores,
)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _slugify(s: str, max_len: int = 80) -> str:
    import re

    out = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_")
    if not out:
        out = "unnamed"
    return out[:max_len]


def _load_index_rows(index_meta: Path) -> list[dict[str, Any]]:
    with open(index_meta) as f:
        meta = json.load(f)
    rows = meta.get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"No rows found in index metadata: {index_meta}")
    return rows


def _select_probe_rows_by_name(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row.get("wavetable_name", "")).strip()
        if not name:
            continue
        frame_idx = int(row.get("frame_idx", 0))
        prev = selected.get(name)
        if prev is None or frame_idx < int(prev.get("frame_idx", 0)):
            selected[name] = row
    return selected


def _ensure_candidate_probes_for_names(
    names: list[str],
    wavetable_lib: list[dict[str, Any]],
    selected_rows: dict[str, dict[str, Any]],
    out_dir: Path,
    probe_archetype: str,
    probe_tail_s: float,
    trim_min_duration_s: float,
    cache: dict[str, Path],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = [n for n in names if n not in cache and n in selected_rows]
    if not missing:
        return

    init_preset = _load_init_preset()
    notes = make_probe_notes(probe_archetype)
    synth = _load_vital()

    for name in missing:
        row = selected_rows[name]
        source_idx = int(row["source_wavetable_idx"])
        frame_idx = int(row.get("frame_idx", 0))
        wt = wavetable_lib[source_idx]
        fname = f"{_slugify(name)}__src{source_idx:04d}_f{frame_idx:03d}.wav"
        path = out_dir / fname
        if not path.exists():
            from scripts.build_wavetable_retrieval_baseline import _build_probe_preset

            preset = _build_probe_preset(init_preset, wt, frame_idx)
            synth.load_json(json.dumps(preset))
            audio = _render_note_list(synth, notes, SAMPLE_RATE, tail_s=probe_tail_s)
            audio = trim_silence(audio, SAMPLE_RATE, min_duration_s=trim_min_duration_s)
            sf.write(path, audio.T, SAMPLE_RATE)
        cache[name] = path


def _build_entries(outputs_root: Path, query_key: str, before_key: str, max_samples: int | None) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for manifest, row in _iter_manifest_rows(outputs_root):
        sample_id = row.get("sample_id")
        if not sample_id:
            continue
        sample_id = str(sample_id)
        archetype = str(row.get("archetype", "unknown"))

        query_audio = row.get(query_key)
        if not isinstance(query_audio, str) or not query_audio.strip():
            query_audio = row.get("gt_wav")
        if not isinstance(query_audio, str) or not query_audio.strip():
            continue
        query_path = _resolve_maybe_relative(query_audio, manifest.parent)
        if not query_path.exists():
            continue

        before_audio = row.get(before_key)
        before_path = None
        if isinstance(before_audio, str) and before_audio.strip():
            p = _resolve_maybe_relative(before_audio, manifest.parent)
            if p.exists():
                before_path = p

        target_preset_path = _resolve_target_preset_path(manifest, row)
        if target_preset_path is None:
            continue

        entries.append(
            {
                "sample_id": sample_id,
                "archetype": archetype,
                "manifest_path": str(manifest),
                "query_audio_path": str(query_path),
                "before_audio_path": str(before_path) if before_path else None,
                "target_preset_path": str(target_preset_path),
            }
        )

    uniq: dict[str, dict[str, Any]] = {}
    for e in entries:
        uniq.setdefault(e["sample_id"], e)
    out = list(uniq.values())
    if max_samples is not None:
        out = out[:max_samples]
    return out


def _maybe_build_clap_shortlist_data(index_npy: Path, index_meta_rows: list[dict[str, Any]]) -> dict[str, Any]:
    idx_npz = np.load(index_npy)
    embeddings = idx_npz["embeddings"].astype(np.float32)
    if len(embeddings) != len(index_meta_rows):
        raise RuntimeError(
            f"Index metadata rows ({len(index_meta_rows)}) != embedding rows ({len(embeddings)})"
        )
    norms = np.linalg.norm(embeddings, axis=1) + 1e-12
    return {"embeddings": embeddings, "norms": norms}


def _clap_shortlist_names(
    query_audio_path: Path,
    shortlist_size: int,
    clap_model: Any,
    clap_processor: Any,
    clap_device: str,
    shortlist_data: dict[str, Any],
    rows_meta: list[dict[str, Any]],
) -> list[str]:
    audio, sr = sf.read(query_audio_path, always_2d=True)
    audio = audio.T.astype(np.float32)
    q = _embed_clap(audio, sr, clap_model, clap_processor, clap_device)
    q_norm = np.linalg.norm(q) + 1e-12
    row_scores = (shortlist_data["embeddings"] @ q) / (shortlist_data["norms"] * q_norm)
    name_scores = _collapse_row_scores_to_name_scores(row_scores, rows_meta)
    ranked = _top_name_scores(name_scores, topn=shortlist_size)
    return [n for n, _ in ranked]


def _build_oracle_mix_candidates(
    gt_names: list[str],
    universe_names: list[str],
    mix_size: int,
    rng: random.Random,
    hard_negative_names: list[str] | None = None,
) -> list[str]:
    mix_size = max(1, int(mix_size))
    universe_set = set(universe_names)
    gt_present = [n for n in list(dict.fromkeys(gt_names)) if n in universe_set]
    if not gt_present:
        return []

    chosen_gts = gt_present[:mix_size]
    exclude = set(gt_present)

    negatives: list[str] = []
    if hard_negative_names:
        for n in hard_negative_names:
            if n in exclude or n not in universe_set:
                continue
            if n not in negatives:
                negatives.append(n)
            if len(negatives) >= max(0, mix_size - len(chosen_gts)):
                break

    if len(negatives) < max(0, mix_size - len(chosen_gts)):
        pool = [n for n in universe_names if n not in exclude and n not in negatives]
        rng.shuffle(pool)
        need = max(0, mix_size - len(chosen_gts) - len(negatives))
        negatives.extend(pool[:need])

    candidates = chosen_gts + negatives[: max(0, mix_size - len(chosen_gts))]
    rng.shuffle(candidates)
    return candidates


def _first_gt_rank(ranked_names: list[str], gt_names: list[str]) -> int | None:
    gt = set(gt_names)
    for i, name in enumerate(ranked_names, start=1):
        if name in gt:
            return i
    return None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a) + 1e-12
    bn = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (an * bn))


def _emit_spotcheck_audio(
    sample_id: str,
    query_audio_path: str,
    before_audio_path: str | None,
    gt_names: list[str],
    selected_names: list[str],
    ranked_names: list[str],
    name_scores: dict[str, float],
    name_deltas: dict[str, float],
    candidate_audio: dict[str, Path],
    out_root: Path,
) -> None:
    out = out_root / sample_id
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(query_audio_path, out / "00_target.wav")
    if before_audio_path and Path(before_audio_path).exists():
        shutil.copyfile(before_audio_path, out / "00_before.wav")

    for i, name in enumerate(selected_names, start=1):
        if name not in candidate_audio:
            continue
        s = float(name_scores.get(name, 0.0))
        d = float(name_deltas.get(name, 0.0))
        fname = f"{i:02d}_selected_{_slugify(name)}_sim{s:.4f}_delta{d:+.4f}.wav"
        shutil.copyfile(candidate_audio[name], out / fname)

    for i, name in enumerate(ranked_names[:3], start=1):
        if name in selected_names or name not in candidate_audio:
            continue
        s = float(name_scores.get(name, 0.0))
        d = float(name_deltas.get(name, 0.0))
        fname = f"{10+i:02d}_rank{i}_{_slugify(name)}_sim{s:.4f}_delta{d:+.4f}.wav"
        shutil.copyfile(candidate_audio[name], out / fname)

    for name in gt_names:
        if name in selected_names or name not in candidate_audio:
            continue
        shutil.copyfile(candidate_audio[name], out / f"90_gt_{_slugify(name)}.wav")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index-npy", type=Path, default=Path("outputs/wt_retrieval_baseline/wt_index.npz"))
    ap.add_argument("--index-meta", type=Path, default=Path("outputs/wt_retrieval_baseline/wt_index_meta.json"))
    ap.add_argument("--wavetable-lib", type=Path, default=Path("data/wavetable_lib.json"))
    ap.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    ap.add_argument("--query-key", default="gt_wav")
    ap.add_argument("--before-key", default="default_wav")
    ap.add_argument("--max-samples", type=int, default=64)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/wt_delta_selector"))

    ap.add_argument("--candidate-source", choices=["all", "clap_topn", "oracle_mix8"], default="oracle_mix8")
    ap.add_argument("--candidate-limit", type=int, default=8)
    ap.add_argument("--oracle-hard-pool", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--select-k", type=int, default=3)
    ap.add_argument("--delta-min", type=float, default=0.0)
    ap.add_argument("--force-top1-if-empty", action="store_true")

    ap.add_argument("--probe-archetype", default="lead")
    ap.add_argument("--probe-tail-s", type=float, default=1.0)
    ap.add_argument("--trim-min-duration-s", type=float, default=0.5)

    ap.add_argument("--clap-device", default="cuda:0")
    ap.add_argument("--spotcheck-dir", type=Path, default=None)
    ap.add_argument("--no-tqdm", action="store_true")
    args = ap.parse_args()

    rows_meta = _load_index_rows(args.index_meta)
    selected_by_name = _select_probe_rows_by_name(rows_meta)
    wavetable_lib = _load_wavetable_lib(args.wavetable_lib)

    entries = _build_entries(args.outputs_root, args.query_key, args.before_key, args.max_samples)
    if not entries:
        raise RuntimeError("No entries found for evaluation.")

    universe_names = sorted(selected_by_name.keys(), key=lambda x: x.lower())
    shortlist_data = None
    if args.candidate_source in {"clap_topn", "oracle_mix8"}:
        shortlist_data = _maybe_build_clap_shortlist_data(args.index_npy, rows_meta)

    clap_model, clap_processor = _load_clap(args.clap_device)

    candidate_probe_dir = args.out_dir / "candidate_probes"
    candidate_audio: dict[str, Path] = {}
    emb_cache: dict[str, np.ndarray] = {}

    def emb_for_path(path: Path) -> np.ndarray:
        key = str(path)
        if key in emb_cache:
            return emb_cache[key]
        audio, sr = sf.read(path, always_2d=True)
        audio = audio.T.astype(np.float32)
        emb = _embed_clap(audio, sr, clap_model, clap_processor, args.clap_device)
        emb_cache[key] = emb
        return emb

    rows: list[dict[str, Any]] = []
    iterator: Any = entries
    if not args.no_tqdm and tqdm is not None:
        iterator = tqdm(entries, desc="Delta eval", unit="sample", dynamic_ncols=True)

    for e in iterator:
        gt_names = _extract_gt_wavetable_names(Path(e["target_preset_path"]))
        if not gt_names:
            continue

        if args.candidate_source == "oracle_mix8":
            sid_seed = int(hashlib.sha1(e["sample_id"].encode("utf-8")).hexdigest()[:8], 16)
            rng = random.Random(int(args.seed) + sid_seed)
            hard_pool = _clap_shortlist_names(
                query_audio_path=Path(e["query_audio_path"]),
                shortlist_size=max(int(args.oracle_hard_pool), int(args.candidate_limit) * 2),
                clap_model=clap_model,
                clap_processor=clap_processor,
                clap_device=args.clap_device,
                shortlist_data=shortlist_data,
                rows_meta=rows_meta,
            )
            candidate_names = _build_oracle_mix_candidates(
                gt_names=gt_names,
                universe_names=universe_names,
                mix_size=int(args.candidate_limit),
                rng=rng,
                hard_negative_names=hard_pool,
            )
        elif args.candidate_source == "clap_topn":
            candidate_names = _clap_shortlist_names(
                query_audio_path=Path(e["query_audio_path"]),
                shortlist_size=max(1, int(args.candidate_limit)),
                clap_model=clap_model,
                clap_processor=clap_processor,
                clap_device=args.clap_device,
                shortlist_data=shortlist_data,
                rows_meta=rows_meta,
            )
        else:
            candidate_names = universe_names[: int(args.candidate_limit)]

        if not candidate_names:
            continue

        _ensure_candidate_probes_for_names(
            names=candidate_names,
            wavetable_lib=wavetable_lib,
            selected_rows=selected_by_name,
            out_dir=candidate_probe_dir,
            probe_archetype=args.probe_archetype,
            probe_tail_s=args.probe_tail_s,
            trim_min_duration_s=args.trim_min_duration_s,
            cache=candidate_audio,
        )

        query_emb = emb_for_path(Path(e["query_audio_path"]))
        before_path = Path(e["before_audio_path"]) if e.get("before_audio_path") else None
        if before_path is not None and before_path.exists():
            before_emb = emb_for_path(before_path)
            baseline_sim = _cosine(before_emb, query_emb)
        else:
            baseline_sim = 0.0

        name_scores: dict[str, float] = {}
        name_deltas: dict[str, float] = {}
        for name in candidate_names:
            cpath = candidate_audio[name]
            cemb = emb_for_path(cpath)
            sim = _cosine(cemb, query_emb)
            name_scores[name] = sim
            name_deltas[name] = sim - baseline_sim

        ranked = sorted(candidate_names, key=lambda n: name_deltas[n], reverse=True)
        selected = [n for n in ranked if name_deltas[n] > float(args.delta_min)][: max(1, int(args.select_k))]
        if not selected and args.force_top1_if_empty and ranked:
            selected = [ranked[0]]

        gt_set = set(gt_names)
        gt_in_pool = any(n in gt_set for n in candidate_names)
        selected_has_gt = any(n in gt_set for n in selected)
        selected_top1_is_gt = bool(selected and selected[0] in gt_set)
        best_gt_rank = _first_gt_rank(ranked, gt_names)

        if args.spotcheck_dir:
            _emit_spotcheck_audio(
                sample_id=e["sample_id"],
                query_audio_path=e["query_audio_path"],
                before_audio_path=e.get("before_audio_path"),
                gt_names=gt_names,
                selected_names=selected,
                ranked_names=ranked,
                name_scores=name_scores,
                name_deltas=name_deltas,
                candidate_audio=candidate_audio,
                out_root=args.spotcheck_dir,
            )

        rows.append(
            {
                "sample_id": e["sample_id"],
                "archetype": e["archetype"],
                "query_audio_path": e["query_audio_path"],
                "before_audio_path": e.get("before_audio_path"),
                "target_preset_path": e["target_preset_path"],
                "gt_wavetable_names": gt_names,
                "candidate_pool_names": candidate_names,
                "candidate_count": len(candidate_names),
                "gt_in_pool": gt_in_pool,
                "baseline_similarity": baseline_sim,
                "ranked_names": ranked,
                "ranked_deltas": [float(name_deltas[n]) for n in ranked],
                "ranked_scores": [float(name_scores[n]) for n in ranked],
                "selected_names": selected,
                "selected_count": len(selected),
                "selected_has_gt": selected_has_gt,
                "selected_top1_is_gt": selected_top1_is_gt,
                "best_gt_rank": best_gt_rank,
            }
        )

    if not args.no_tqdm and tqdm is not None:
        iterator.close()

    rows.sort(key=lambda r: r["sample_id"])
    n = len(rows)
    if n == 0:
        raise RuntimeError("No rows evaluated.")

    gt_in_pool_rate = float(sum(1 for r in rows if r["gt_in_pool"]) / n)
    selected_any_gt_rate = float(sum(1 for r in rows if r["selected_has_gt"]) / n)
    selected_top1_gt_rate = float(sum(1 for r in rows if r["selected_top1_is_gt"]) / n)
    ranks = [int(r["best_gt_rank"]) for r in rows if isinstance(r.get("best_gt_rank"), int)]
    mean_best_gt_rank = float(sum(ranks) / len(ranks)) if ranks else None
    mean_selected_count = float(sum(int(r["selected_count"]) for r in rows) / n)
    mean_baseline_similarity = float(sum(float(r["baseline_similarity"]) for r in rows) / n)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_samples": n,
        "candidate_source": args.candidate_source,
        "candidate_limit": args.candidate_limit,
        "select_k": args.select_k,
        "delta_min": args.delta_min,
        "force_top1_if_empty": bool(args.force_top1_if_empty),
        "metric": "clap_cosine_delta",
        "clap_device": args.clap_device,
        "gt_in_pool_rate": gt_in_pool_rate,
        "selected_any_gt_rate": selected_any_gt_rate,
        "selected_top1_gt_rate": selected_top1_gt_rate,
        "mean_best_gt_rank": mean_best_gt_rank,
        "mean_selected_count": mean_selected_count,
        "mean_baseline_similarity": mean_baseline_similarity,
        "candidate_probe_count": len(selected_by_name),
        "candidate_probes_rendered": len(candidate_audio),
        "spotcheck_dir": str(args.spotcheck_dir) if args.spotcheck_dir else None,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_dir / "delta_rows.jsonl"
    summary_path = args.out_dir / "delta_summary.json"
    with open(rows_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Evaluated samples: {n}")
    print(f"GT in pool rate: {gt_in_pool_rate:.4f}")
    print(f"Selected-any-GT rate: {selected_any_gt_rate:.4f}")
    print(f"Selected-top1-GT rate: {selected_top1_gt_rate:.4f}")
    print(f"Rows: {rows_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
