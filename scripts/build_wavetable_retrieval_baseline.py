#!/usr/bin/env python3
"""
Build and evaluate a naive CLAP-based wavetable retrieval baseline.

Subcommands:
  1) build-wt-index
     Render canonical probe audio for each (wavetable, frame) row and embed with CLAP.
  2) eval-wt-retrieval
     Use task-reference query audio from existing manifests and measure Recall@K.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maestro.render.vital import SAMPLE_RATE, _load_vital, _render_note_list, make_probe_notes, trim_silence
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - cosmetic fallback
    tqdm = None

CLAP_MODEL_ID = "laion/larger_clap_general"
CLAP_SAMPLE_RATE = 48000
DEFAULT_KS = (1, 5, 10)


def _load_wavetable_lib(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        lib = json.load(f)
    if not isinstance(lib, list):
        raise ValueError(f"Expected wavetable library list at {path}")
    return lib


def _load_init_preset() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "maestro" / "synth" / "init_preset.json"
    with open(path) as f:
        return json.load(f)


def _load_clap(device: str) -> tuple[Any, Any]:
    try:
        import torch
        from transformers import ClapModel, ClapProcessor
    except Exception as e:  # pragma: no cover - exercised in integration, not unit
        raise RuntimeError(f"CLAP dependencies unavailable: {e}") from e

    try:
        processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
        model = ClapModel.from_pretrained(CLAP_MODEL_ID).to(device)
        model.eval()
    except Exception as e:  # pragma: no cover - exercised in integration, not unit
        raise RuntimeError(f"Failed to load CLAP model {CLAP_MODEL_ID}: {e}") from e
    return model, processor


def _resolve_clap_devices(clap_device: str, clap_devices: str | None) -> list[str]:
    """
    Resolve target CLAP devices.

    Rules:
      - If --clap-devices is set, use it verbatim (comma-separated).
      - Else if --clap-device == cuda, expand to all visible GPUs (cuda:0..N-1).
      - Else use the single --clap-device value.
    """
    if clap_devices:
        devices = [d.strip() for d in clap_devices.split(",") if d.strip()]
        if not devices:
            raise ValueError("--clap-devices provided but no valid devices parsed.")
        return devices

    if clap_device == "cuda":
        try:
            import torch
        except Exception as e:
            raise RuntimeError(f"--clap-device cuda requested but torch unavailable: {e}") from e
        if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
            raise RuntimeError("--clap-device cuda requested but no CUDA GPUs are visible.")
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]

    return [clap_device]


def _embed_clap(audio: np.ndarray, sample_rate: int, model: Any, processor: Any, device: str) -> np.ndarray:
    import torch

    if audio.ndim == 2:
        mono = audio.mean(axis=0)
    else:
        mono = audio
    mono = mono.astype(np.float32)

    g = gcd(CLAP_SAMPLE_RATE, sample_rate)
    up = CLAP_SAMPLE_RATE // g
    down = sample_rate // g
    mono_48k = resample_poly(mono, up, down).astype(np.float32)

    target_len = CLAP_SAMPLE_RATE * 10
    if len(mono_48k) < target_len:
        mono_48k = np.pad(mono_48k, (0, target_len - len(mono_48k)))
    else:
        mono_48k = mono_48k[:target_len]

    inputs = processor(audio=mono_48k, sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.get_audio_features(**inputs)
    emb = out.pooler_output if hasattr(out, "pooler_output") else out
    return emb.squeeze(0).cpu().numpy().astype(np.float32)


def _dedup_wavetables_by_name(
    wavetable_lib: list[dict[str, Any]],
    include_empty_names: bool = False,
) -> list[tuple[int, dict[str, Any]]]:
    seen: set[str] = set()
    out: list[tuple[int, dict[str, Any]]] = []
    for idx, wt in enumerate(wavetable_lib):
        if not isinstance(wt, dict):
            continue
        name = str(wt.get("name", "")).strip()
        if not include_empty_names and not name:
            continue
        key = name if name else f"__unnamed_{idx}"
        if key in seen:
            continue
        seen.add(key)
        out.append((idx, wt))
    return out


def _n_frames(wavetable: dict[str, Any]) -> int:
    try:
        keyframes = wavetable["groups"][0]["components"][0]["keyframes"]
        return max(1, len(keyframes))
    except (KeyError, IndexError, TypeError):
        return 1


def _build_probe_preset(
    init_preset: dict[str, Any],
    wavetable: dict[str, Any],
    frame_idx: int,
) -> dict[str, Any]:
    preset = copy.deepcopy(init_preset)
    settings = preset["settings"]
    settings["sample_on"] = 0.0

    # Isolate oscillator 1 as a source probe.
    settings["osc_1_on"] = 1.0
    settings["osc_1_level"] = 1.0
    settings["osc_1_wave_frame"] = float(frame_idx)
    settings["osc_1_unison_voices"] = 1.0
    settings["osc_1_unison_detune"] = 0.0
    settings["osc_1_pan"] = 0.0

    settings["osc_2_on"] = 0.0
    settings["osc_2_level"] = 0.0
    settings["osc_3_on"] = 0.0
    settings["osc_3_level"] = 0.0

    # Turn off major FX to keep probe representation stable.
    for k in ("reverb_on", "delay_on", "chorus_on", "distortion_on", "phaser_on", "flanger_on", "compressor_on"):
        if k in settings:
            settings[k] = 0.0

    if "wavetables" in settings and isinstance(settings["wavetables"], list) and settings["wavetables"]:
        settings["wavetables"][0] = copy.deepcopy(wavetable)
    return preset


def _build_probe_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "probe_archetype": args.probe_archetype,
        "tail_s": args.probe_tail_s,
        "trim_min_duration_s": args.trim_min_duration_s,
        "max_frames_per_wt": args.max_frames_per_wt,
        "include_empty_names": args.include_empty_names,
        "clap_model_id": CLAP_MODEL_ID,
    }


def _probe_config_hash(cfg: dict[str, Any]) -> str:
    s = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _index_worker(payload: dict[str, Any]) -> list[tuple[int, dict[str, Any], np.ndarray]]:
    device = str(payload["device"])
    jobs: list[tuple[int, int, int, int]] = payload["jobs"]
    wavetable_lib_path = Path(payload["wavetable_lib_path"])
    probe_archetype = str(payload["probe_archetype"])
    probe_tail_s = float(payload["probe_tail_s"])
    trim_min_duration_s = float(payload["trim_min_duration_s"])
    show_progress = bool(payload.get("show_progress", False))
    progress_position = int(payload.get("progress_position", 0))

    wavetable_lib = _load_wavetable_lib(wavetable_lib_path)
    init_preset = _load_init_preset()
    notes = make_probe_notes(probe_archetype)
    synth = _load_vital()
    model, processor = _load_clap(device)

    out: list[tuple[int, dict[str, Any], np.ndarray]] = []
    iterator: Any = jobs
    if show_progress and tqdm is not None:
        iterator = tqdm(
            jobs,
            desc=f"Index {device}",
            unit="probe",
            position=progress_position,
            leave=True,
            dynamic_ncols=True,
        )
    for job_idx, source_idx, frame_idx, total_frames in iterator:
        wt = wavetable_lib[source_idx]
        name = str(wt.get("name", "")).strip()
        preset = _build_probe_preset(init_preset, wt, frame_idx)
        synth.load_json(json.dumps(preset))
        audio = _render_note_list(synth, notes, SAMPLE_RATE, tail_s=probe_tail_s)
        audio = trim_silence(audio, SAMPLE_RATE, min_duration_s=trim_min_duration_s)
        emb = _embed_clap(audio, SAMPLE_RATE, model, processor, device)
        row = {
            "wavetable_name": name,
            "frame_idx": int(frame_idx),
            "source_wavetable_idx": int(source_idx),
            "n_frames": int(total_frames),
        }
        out.append((int(job_idx), row, emb))
    if show_progress and tqdm is not None:
        iterator.close()
    return out


def _eval_worker(payload: dict[str, Any]) -> list[dict[str, Any]]:
    device = str(payload["device"])
    entries: list[dict[str, Any]] = payload["entries"]
    index_npy = Path(payload["index_npy"])
    index_meta = Path(payload["index_meta"])
    ks: list[int] = payload["ks"]
    topk = int(payload["topk"])
    show_progress = bool(payload.get("show_progress", False))
    progress_position = int(payload.get("progress_position", 0))

    idx_npz = np.load(index_npy)
    embeddings = idx_npz["embeddings"].astype(np.float32)
    with open(index_meta) as f:
        meta = json.load(f)
    rows_meta = meta.get("rows", [])
    if len(rows_meta) != len(embeddings):
        raise RuntimeError(
            f"Index metadata rows ({len(rows_meta)}) != embedding rows ({len(embeddings)})"
        )
    row_norms = np.linalg.norm(embeddings, axis=1) + 1e-12

    model, processor = _load_clap(device)
    out: list[dict[str, Any]] = []
    iterator: Any = entries
    if show_progress and tqdm is not None:
        iterator = tqdm(
            entries,
            desc=f"Eval {device}",
            unit="sample",
            position=progress_position,
            leave=True,
            dynamic_ncols=True,
        )
    for e in iterator:
        try:
            gt_names = _extract_gt_wavetable_names(Path(e["target_preset_path"]))
            if not gt_names:
                continue
            audio, sr = sf.read(e["query_audio_path"], always_2d=True)
            audio = audio.T.astype(np.float32)  # (2, N)
            q = _embed_clap(audio, sr, model, processor, device)
            q_norm = np.linalg.norm(q) + 1e-12
            row_scores = (embeddings @ q) / (row_norms * q_norm)

            name_scores = _collapse_row_scores_to_name_scores(row_scores, rows_meta)
            ranked = _top_name_scores(name_scores, topn=topk)
            ranked_names = [n for n, _ in ranked]
            hits = _compute_recall_hits(gt_names, ranked_names, ks)
            has_init = "Init" in gt_names
            out.append(
                {
                    "sample_id": e["sample_id"],
                    "archetype": e["archetype"],
                    "query_audio_path": e["query_audio_path"],
                    "target_preset_path": e["target_preset_path"],
                    "gt_wavetable_names": gt_names,
                    "osc_count": len(gt_names),
                    "has_init": has_init,
                    "top_names": ranked_names,
                    "top_name_scores": [float(s) for _, s in ranked],
                    **hits,
                }
            )
        except Exception:
            continue
    if show_progress and tqdm is not None:
        iterator.close()
    return out


def _build_wt_index(args: argparse.Namespace) -> None:
    wavetable_lib = _load_wavetable_lib(args.wavetable_lib)
    deduped = _dedup_wavetables_by_name(
        wavetable_lib, include_empty_names=args.include_empty_names
    )
    if args.max_wavetables is not None:
        deduped = deduped[: args.max_wavetables]
    if not deduped:
        raise RuntimeError("No wavetable entries available to index.")

    devices = _resolve_clap_devices(args.clap_device, args.clap_devices)
    device_label = ",".join(devices)

    jobs: list[tuple[int, int, int, int]] = []  # (job_idx, source_wt_idx, frame_idx, n_frames)
    for source_idx, wt in deduped:
        total_frames = _n_frames(wt)
        if args.max_frames_per_wt is None:
            frame_ids = range(total_frames)
        else:
            frame_ids = range(min(total_frames, args.max_frames_per_wt))
        for frame_idx in frame_ids:
            jobs.append((len(jobs), int(source_idx), int(frame_idx), int(total_frames)))

    work_by_device: dict[str, list[tuple[int, int, int, int]]] = {d: [] for d in devices}
    for i, job in enumerate(jobs):
        work_by_device[devices[i % len(devices)]].append(job)

    results: list[tuple[int, dict[str, Any], np.ndarray]] = []
    if len(devices) == 1:
        dev = devices[0]
        results.extend(
            _index_worker(
                {
                    "device": dev,
                    "jobs": work_by_device[dev],
                    "wavetable_lib_path": str(args.wavetable_lib),
                    "probe_archetype": args.probe_archetype,
                    "probe_tail_s": args.probe_tail_s,
                    "trim_min_duration_s": args.trim_min_duration_s,
                    "show_progress": (not args.no_tqdm),
                    "progress_position": 0,
                }
            )
        )
    else:
        payloads = [
            {
                "device": dev,
                "jobs": shard,
                "wavetable_lib_path": str(args.wavetable_lib),
                "probe_archetype": args.probe_archetype,
                "probe_tail_s": args.probe_tail_s,
                "trim_min_duration_s": args.trim_min_duration_s,
                "show_progress": (not args.no_tqdm),
                "progress_position": i,
            }
            for i, (dev, shard) in enumerate(work_by_device.items())
            if shard
        ]
        with ProcessPoolExecutor(max_workers=len(payloads), mp_context=mp.get_context("spawn")) as ex:
            for part in ex.map(_index_worker, payloads):
                results.extend(part)

    results.sort(key=lambda x: x[0])
    rows: list[dict[str, Any]] = []
    embeddings: list[np.ndarray] = []
    for row_idx, (_, row, emb) in enumerate(results):
        row["row_idx"] = row_idx
        rows.append(row)
        embeddings.append(emb)

    emb_matrix = np.stack(embeddings).astype(np.float32)
    args.index_npy.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.index_npy, embeddings=emb_matrix)

    cfg = _build_probe_config(args)
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "clap_model_id": CLAP_MODEL_ID,
        "clap_devices": devices,
        "clap_device_label": device_label,
        "probe_config": cfg,
        "probe_config_hash": _probe_config_hash(cfg),
        "rows": rows,
    }
    with open(args.index_meta, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Indexed rows: {len(rows)}")
    print(f"CLAP devices: {device_label}")
    print(f"Index embeddings: {args.index_npy}")
    print(f"Index metadata: {args.index_meta}")


def _resolve_maybe_relative(path: str, base_dir: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (base_dir / p)


def _resolve_target_preset_path(
    manifest_path: Path,
    row: dict[str, Any],
) -> Path | None:
    base = manifest_path.parent
    sample_id = row.get("sample_id")

    direct = row.get("target_preset_path")
    if isinstance(direct, str) and direct.strip():
        p = _resolve_maybe_relative(direct, base)
        if p.exists():
            return p

    path_file = row.get("path_file")
    if isinstance(path_file, str) and path_file.strip():
        pf = _resolve_maybe_relative(path_file, base)
        if pf.exists():
            try:
                with open(pf) as f:
                    path_obj = json.load(f)
                tpp = path_obj.get("target_preset_path")
                if isinstance(tpp, str) and tpp.strip():
                    p = _resolve_maybe_relative(tpp, pf.parent)
                    if p.exists():
                        return p
            except Exception:
                pass
            if sample_id:
                inferred = pf.parent / f"{sample_id}_target.vital"
                if inferred.exists():
                    return inferred

    if sample_id:
        candidates = [
            base / "paths" / f"{sample_id}_target.vital",
            base / f"{sample_id}_target.vital",
        ]
        for c in candidates:
            if c.exists():
                return c
    return None


def _extract_gt_wavetable_names_from_preset_dict(preset: dict[str, Any]) -> list[str]:
    settings = preset.get("settings", {})
    wts = settings.get("wavetables", [])
    names: list[str] = []
    for osc_idx in (1, 2, 3):
        if float(settings.get(f"osc_{osc_idx}_on", 0.0)) < 0.5:
            continue
        wt = wts[osc_idx - 1] if isinstance(wts, list) and len(wts) >= osc_idx else None
        if not isinstance(wt, dict):
            continue
        name = str(wt.get("name", "")).strip()
        if name:
            names.append(name)
    # Order-preserving dedupe
    return list(dict.fromkeys(names))


def _extract_gt_wavetable_names(preset_path: Path) -> list[str]:
    with open(preset_path) as f:
        preset = json.load(f)
    return _extract_gt_wavetable_names_from_preset_dict(preset)


def _collapse_row_scores_to_name_scores(
    row_scores: np.ndarray,
    rows_meta: list[dict[str, Any]],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for score, row in zip(row_scores.tolist(), rows_meta):
        name = str(row.get("wavetable_name", "")).strip()
        if not name:
            continue
        prev = out.get(name)
        if prev is None or score > prev:
            out[name] = float(score)
    return out


def _top_name_scores(name_scores: dict[str, float], topn: int) -> list[tuple[str, float]]:
    return sorted(name_scores.items(), key=lambda kv: kv[1], reverse=True)[:topn]


def _compute_recall_hits(gt_names: list[str], ranked_names: list[str], ks: list[int]) -> dict[str, int]:
    gt = set(gt_names)
    hits: dict[str, int] = {}
    for k in ks:
        topk = set(ranked_names[:k])
        hits[f"r@{k}"] = 1 if gt and (gt & topk) else 0
    return hits


def _iter_manifest_rows(outputs_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    out: list[tuple[Path, dict[str, Any]]] = []
    for manifest in sorted(outputs_root.rglob("manifest.jsonl")):
        with open(manifest) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out.append((manifest, row))
    return out


def _build_eval_entries(
    outputs_root: Path,
    query_key: str,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for manifest, row in _iter_manifest_rows(outputs_root):
        if "sample_id" not in row:
            continue
        sample_id = str(row["sample_id"])
        archetype = str(row.get("archetype", "unknown"))

        query_audio = row.get(query_key)
        if not isinstance(query_audio, str) or not query_audio.strip():
            if query_key != "gt_wav":
                query_audio = row.get("gt_wav")
        if not isinstance(query_audio, str) or not query_audio.strip():
            continue
        query_path = _resolve_maybe_relative(query_audio, manifest.parent)
        if not query_path.exists():
            continue

        target_preset_path = _resolve_target_preset_path(manifest, row)
        if target_preset_path is None:
            continue

        entries.append(
            {
                "sample_id": sample_id,
                "archetype": archetype,
                "manifest_path": str(manifest),
                "query_audio_path": str(query_path),
                "target_preset_path": str(target_preset_path),
            }
        )

    # Stable order and unique sample IDs to avoid duplicate counting across manifests.
    uniq: dict[str, dict[str, Any]] = {}
    for e in entries:
        uniq.setdefault(e["sample_id"], e)
    out = list(uniq.values())
    if max_samples is not None:
        out = out[:max_samples]
    return out


def _aggregate_eval(rows: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    def _empty_stats() -> dict[str, float]:
        return {f"r@{k}": 0.0 for k in ks}

    if not rows:
        return {
            "n_samples": 0,
            "overall": _empty_stats(),
            "by_archetype": {},
            "by_osc_count": {},
            "by_has_init": {},
        }

    n = len(rows)
    overall_hits = {f"r@{k}": 0 for k in ks}
    by_arch: dict[str, list[dict[str, Any]]] = {}
    by_osc: dict[int, list[dict[str, Any]]] = {}
    by_init: dict[str, list[dict[str, Any]]] = {"has_init": [], "no_init": []}

    for row in rows:
        for k in ks:
            overall_hits[f"r@{k}"] += int(row[f"r@{k}"])
        by_arch.setdefault(row["archetype"], []).append(row)
        by_osc.setdefault(int(row["osc_count"]), []).append(row)
        by_init["has_init" if row["has_init"] else "no_init"].append(row)

    def _avg(bucket: list[dict[str, Any]]) -> dict[str, float]:
        if not bucket:
            return _empty_stats()
        m = len(bucket)
        return {f"r@{k}": float(sum(int(r[f"r@{k}"]) for r in bucket) / m) for k in ks}

    return {
        "n_samples": n,
        "overall": {f"r@{k}": float(overall_hits[f"r@{k}"] / n) for k in ks},
        "by_archetype": {arch: _avg(bucket) for arch, bucket in sorted(by_arch.items())},
        "by_osc_count": {str(osc): _avg(bucket) for osc, bucket in sorted(by_osc.items())},
        "by_has_init": {k: _avg(v) for k, v in by_init.items()},
    }


def _evaluate_wt_retrieval(args: argparse.Namespace) -> None:
    with open(args.index_meta) as f:
        meta = json.load(f)
    devices = _resolve_clap_devices(args.clap_device, args.clap_devices)
    eval_entries = _build_eval_entries(args.outputs_root, args.query_key, args.max_samples)
    if not eval_entries:
        raise RuntimeError(
            f"No evaluation entries found under {args.outputs_root} with query key '{args.query_key}'."
        )

    ks = sorted(set(DEFAULT_KS + tuple(args.extra_k)))
    per_sample_rows: list[dict[str, Any]] = []
    if len(devices) == 1:
        per_sample_rows.extend(
            _eval_worker(
                {
                    "device": devices[0],
                    "entries": eval_entries,
                    "index_npy": str(args.index_npy),
                    "index_meta": str(args.index_meta),
                    "ks": ks,
                    "topk": args.topk,
                    "show_progress": (not args.no_tqdm),
                    "progress_position": 0,
                }
            )
        )
    else:
        shards: dict[str, list[dict[str, Any]]] = {d: [] for d in devices}
        for i, entry in enumerate(eval_entries):
            shards[devices[i % len(devices)]].append(entry)
        payloads = [
            {
                "device": dev,
                "entries": shard,
                "index_npy": str(args.index_npy),
                "index_meta": str(args.index_meta),
                "ks": ks,
                "topk": args.topk,
                "show_progress": (not args.no_tqdm),
                "progress_position": i,
            }
            for i, (dev, shard) in enumerate(shards.items())
            if shard
        ]
        with ProcessPoolExecutor(max_workers=len(payloads), mp_context=mp.get_context("spawn")) as ex:
            for part in ex.map(_eval_worker, payloads):
                per_sample_rows.extend(part)

    per_sample_rows.sort(key=lambda r: r["sample_id"])

    summary = _aggregate_eval(per_sample_rows, ks)
    summary["query_source"] = "task_reference"
    summary["query_key"] = args.query_key
    summary["index_npy"] = str(args.index_npy)
    summary["index_meta"] = str(args.index_meta)
    summary["clap_devices"] = devices
    summary["topk"] = args.topk
    summary["ks"] = ks
    summary["target_r10"] = args.target_r10
    summary["pass_r10"] = summary["overall"].get("r@10", 0.0) >= args.target_r10

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_dir / "retrieval_rows.jsonl"
    summary_path = args.out_dir / "retrieval_summary.json"
    with open(rows_path, "w") as f:
        for row in per_sample_rows:
            f.write(json.dumps(row) + "\n")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Evaluated samples: {len(per_sample_rows)}")
    print("Overall Recall:")
    for k in ks:
        print(f"  R@{k}: {summary['overall'].get(f'r@{k}', 0.0):.4f}")
    print(f"Pass gate (R@10 >= {args.target_r10:.2f}): {summary['pass_r10']}")
    print(f"Rows: {rows_path}")
    print(f"Summary: {summary_path}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build-wt-index", help="Build CLAP wavetable-frame retrieval index")
    build.add_argument("--wavetable-lib", type=Path, default=Path("data/wavetable_lib.json"))
    build.add_argument("--index-npy", type=Path, default=Path("outputs/wt_retrieval_baseline/wt_index.npz"))
    build.add_argument("--index-meta", type=Path, default=Path("outputs/wt_retrieval_baseline/wt_index_meta.json"))
    build.add_argument("--clap-device", default="cuda")
    build.add_argument(
        "--clap-devices",
        default=None,
        help="Comma-separated devices (e.g. cuda:0,cuda:1). If unset and --clap-device cuda, uses all visible GPUs.",
    )
    build.add_argument("--probe-archetype", default="lead")
    build.add_argument("--probe-tail-s", type=float, default=1.0)
    build.add_argument("--trim-min-duration-s", type=float, default=0.5)
    build.add_argument("--max-wavetables", type=int, default=None)
    build.add_argument("--max-frames-per-wt", type=int, default=None)
    build.add_argument("--include-empty-names", action="store_true")
    build.add_argument("--no-tqdm", action="store_true", help="Disable progress bars")

    ev = sub.add_parser("eval-wt-retrieval", help="Evaluate wavetable retrieval Recall@K")
    ev.add_argument("--index-npy", type=Path, default=Path("outputs/wt_retrieval_baseline/wt_index.npz"))
    ev.add_argument("--index-meta", type=Path, default=Path("outputs/wt_retrieval_baseline/wt_index_meta.json"))
    ev.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    ev.add_argument("--query-key", default="gt_wav")
    ev.add_argument("--clap-device", default="cuda")
    ev.add_argument(
        "--clap-devices",
        default=None,
        help="Comma-separated devices (e.g. cuda:0,cuda:1). If unset and --clap-device cuda, uses all visible GPUs.",
    )
    ev.add_argument("--topk", type=int, default=10)
    ev.add_argument("--extra-k", type=int, nargs="*", default=[])
    ev.add_argument("--max-samples", type=int, default=None)
    ev.add_argument("--target-r10", type=float, default=0.80)
    ev.add_argument("--out-dir", type=Path, default=Path("outputs/wt_retrieval_baseline"))
    ev.add_argument("--no-tqdm", action="store_true", help="Disable progress bars")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if args.cmd == "build-wt-index":
        _build_wt_index(args)
    elif args.cmd == "eval-wt-retrieval":
        _evaluate_wt_retrieval(args)
    else:  # pragma: no cover
        raise RuntimeError(f"Unknown command {args.cmd}")


if __name__ == "__main__":
    main()
