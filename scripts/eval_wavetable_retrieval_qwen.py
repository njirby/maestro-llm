#!/usr/bin/env python3
"""
Evaluate wavetable retrieval with pairwise Qwen-Omni audio scoring.

Flow:
1) Build evaluation entries from existing manifests (same as CLAP baseline).
2) Render one canonical probe WAV per wavetable name (from index metadata rows).
3) For each sample, ask Qwen-Omni to score TARGET audio vs each candidate probe.
4) Rank candidates by Qwen score and compute Recall@K against GT wavetable names.

This keeps metrics comparable to scripts/build_wavetable_retrieval_baseline.py
while swapping the scorer from CLAP embedding similarity to Omni judgment.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import random
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maestro.render.vital import SAMPLE_RATE, _load_vital, _render_note_list, make_probe_notes, trim_silence
from scripts.build_wavetable_retrieval_baseline import (
    DEFAULT_KS,
    _aggregate_eval,
    _build_eval_entries,
    _build_probe_preset,
    _collapse_row_scores_to_name_scores,
    _compute_recall_hits,
    _extract_gt_wavetable_names,
    _load_clap,
    _load_init_preset,
    _load_wavetable_lib,
    _top_name_scores,
)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - cosmetic fallback
    tqdm = None


PROMPT_TEMPLATE = (
    "You are scoring whether CANDIDATE could plausibly be one wavetable component of TARGET. "
    "TARGET may include extra oscillators, modulation, and effects; focus on core oscillator timbre only. "
    "Use this rubric: "
    "0.00-0.20 very unlikely (different harmonic family); "
    "0.21-0.45 weak similarity; "
    "0.46-0.70 moderate similarity; "
    "0.71-0.89 strong similarity; "
    "0.90-1.00 near-identical dry oscillator character only. "
    "Use full score range. Most pairs should not be above 0.80. "
    "Return strict JSON only with keys: score, confidence, reason. "
    "confidence: float 0.0-1.0. reason: <=20 words."
)

LISTWISE_PROMPT_TEMPLATE = (
    "You will receive audio clips in this order: TARGET first, then candidates C1..C{n}. "
    "Rank candidates by similarity to TARGET's core oscillator timbre "
    "(ignore FX, modulation, stereo spread, and loudness differences). "
    "Return strict JSON only with keys: ranking, scores, reason. "
    "ranking: list of candidate IDs from most to least similar, e.g. [\"C3\",\"C1\",...]. "
    "scores: object mapping each candidate ID to a float in [0,1], higher is more similar, preferably no ties. "
    "reason: <=25 words."
)


@dataclass
class QwenScore:
    score: float
    confidence: float
    reason: str
    raw_text: str


def _slugify(s: str, max_len: int = 80) -> str:
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
    """Pick one canonical row per wavetable name, preferring lower frame index."""
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


def _render_candidate_probes(
    wavetable_lib: list[dict[str, Any]],
    selected_rows: dict[str, dict[str, Any]],
    out_dir: Path,
    probe_archetype: str,
    probe_tail_s: float,
    trim_min_duration_s: float,
    no_tqdm: bool,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    init_preset = _load_init_preset()
    notes = make_probe_notes(probe_archetype)
    synth = _load_vital()

    items = sorted(selected_rows.items(), key=lambda kv: kv[0].lower())
    iterator: Any = items
    if not no_tqdm and tqdm is not None:
        iterator = tqdm(items, desc="Render candidate probes", unit="wavetable", dynamic_ncols=True)

    out: dict[str, Path] = {}
    for name, row in iterator:
        source_idx = int(row["source_wavetable_idx"])
        frame_idx = int(row.get("frame_idx", 0))
        wt = wavetable_lib[source_idx]
        fname = f"{_slugify(name)}__src{source_idx:04d}_f{frame_idx:03d}.wav"
        path = out_dir / fname
        if not path.exists():
            preset = _build_probe_preset(init_preset, wt, frame_idx)
            synth.load_json(json.dumps(preset))
            audio = _render_note_list(synth, notes, SAMPLE_RATE, tail_s=probe_tail_s)
            audio = trim_silence(audio, SAMPLE_RATE, min_duration_s=trim_min_duration_s)
            sf.write(path, audio.T, SAMPLE_RATE)
        out[name] = path

    if not no_tqdm and tqdm is not None:
        iterator.close()
    return out


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
    """Render and cache only the candidate probe WAVs needed for this sample."""
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
            preset = _build_probe_preset(init_preset, wt, frame_idx)
            synth.load_json(json.dumps(preset))
            audio = _render_note_list(synth, notes, SAMPLE_RATE, tail_s=probe_tail_s)
            audio = trim_silence(audio, SAMPLE_RATE, min_duration_s=trim_min_duration_s)
            sf.write(path, audio.T, SAMPLE_RATE)
        cache[name] = path


def _encode_wav_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _parse_qwen_score(text: str) -> QwenScore:
    s = (text or "").strip()

    # Try direct JSON first.
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            score = float(obj.get("score", 0.0))
            conf = float(obj.get("confidence", 0.0))
            reason = str(obj.get("reason", "")).strip()
            return QwenScore(score=_clamp01(score), confidence=_clamp01(conf), reason=reason, raw_text=s)
    except Exception:
        pass

    # Try extracting JSON object from a fenced or mixed response.
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                score = float(obj.get("score", 0.0))
                conf = float(obj.get("confidence", 0.0))
                reason = str(obj.get("reason", "")).strip()
                return QwenScore(score=_clamp01(score), confidence=_clamp01(conf), reason=reason, raw_text=s)
        except Exception:
            pass

    # Final fallback: first numeric token treated as score in [0,1] or [0,100].
    nums = re.findall(r"[-+]?\d*\.?\d+", s)
    if nums:
        val = float(nums[0])
        if val > 1.0:
            val = val / 100.0
        return QwenScore(score=_clamp01(val), confidence=0.0, reason="", raw_text=s)

    return QwenScore(score=0.0, confidence=0.0, reason="", raw_text=s)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    s = (text or "").strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def _parse_qwen_listwise(
    text: str,
    candidate_ids: list[str],
) -> tuple[list[str], dict[str, float], str]:
    valid = set(candidate_ids)
    obj = _extract_json_object(text)
    ranking: list[str] = []
    scores: dict[str, float] = {}
    reason = ""

    if obj is not None:
        raw_rank = obj.get("ranking", [])
        if isinstance(raw_rank, list):
            for x in raw_rank:
                cid = str(x).strip().upper()
                if cid in valid and cid not in ranking:
                    ranking.append(cid)
        raw_scores = obj.get("scores", {})
        if isinstance(raw_scores, dict):
            for k, v in raw_scores.items():
                cid = str(k).strip().upper()
                if cid not in valid:
                    continue
                try:
                    scores[cid] = _clamp01(float(v))
                except Exception:
                    continue
        reason = str(obj.get("reason", "")).strip()

    # Fill missing ranking entries by descending score, then stable candidate order.
    if len(ranking) < len(candidate_ids):
        by_score = sorted(
            [(cid, scores.get(cid, -1.0)) for cid in candidate_ids if cid not in ranking],
            key=lambda kv: kv[1],
            reverse=True,
        )
        for cid, sc in by_score:
            if sc >= 0.0 and cid not in ranking:
                ranking.append(cid)
        for cid in candidate_ids:
            if cid not in ranking:
                ranking.append(cid)

    # If scores missing, synthesize monotonic scores from ranking.
    if not scores:
        n = max(1, len(ranking) - 1)
        for i, cid in enumerate(ranking):
            scores[cid] = _clamp01(1.0 - (i / n))
    else:
        for cid in candidate_ids:
            scores.setdefault(cid, 0.0)

    return ranking[: len(candidate_ids)], scores, reason


def _cache_key(
    query_audio_path: str,
    candidate_name: str,
    candidate_audio_path: str,
    prompt: str,
    model: str,
) -> str:
    raw = "||".join([query_audio_path, candidate_name, candidate_audio_path, prompt, model])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _load_score_cache(path: Path) -> dict[str, QwenScore]:
    out: dict[str, QwenScore] = {}
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                key = str(row.get("cache_key", "")).strip()
                if not key:
                    continue
                out[key] = QwenScore(
                    score=float(row.get("score", 0.0)),
                    confidence=float(row.get("confidence", 0.0)),
                    reason=str(row.get("reason", "")),
                    raw_text=str(row.get("raw_text", "")),
                )
            except Exception:
                continue
    return out


async def _score_pair_qwen(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    server: str,
    model: str,
    prompt: str,
    target_b64: str,
    candidate_b64: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
    retries: int,
) -> QwenScore:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{target_b64}"}},
                    {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{candidate_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    last_err = ""
    for attempt in range(retries + 1):
        try:
            async with sem:
                resp = await client.post(f"{server.rstrip('/')}/v1/chat/completions", json=payload, timeout=timeout_s)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            return _parse_qwen_score(str(text))
        except Exception as e:  # pragma: no cover - network/runtime dependent
            last_err = str(e)
            if attempt >= retries:
                break
            await asyncio.sleep(min(1.0 * (attempt + 1), 3.0))
    return QwenScore(score=0.0, confidence=0.0, reason=f"request_failed: {last_err[:120]}", raw_text="")


async def _rank_listwise_qwen(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    server: str,
    model: str,
    prompt: str,
    target_b64: str,
    candidate_payload: list[tuple[str, str]],  # (candidate_id, candidate_b64)
    max_tokens: int,
    temperature: float,
    timeout_s: float,
    retries: int,
) -> tuple[list[str], dict[str, float], str, str]:
    content: list[dict[str, Any]] = [
        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{target_b64}"}}
    ]
    for _, b64 in candidate_payload:
        content.append({"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{b64}"}})
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    candidate_ids = [cid for cid, _ in candidate_payload]
    last_err = ""
    last_text = ""
    for attempt in range(retries + 1):
        try:
            async with sem:
                resp = await client.post(f"{server.rstrip('/')}/v1/chat/completions", json=payload, timeout=timeout_s)
            resp.raise_for_status()
            text = str(resp.json()["choices"][0]["message"]["content"])
            last_text = text
            ranking, scores, reason = _parse_qwen_listwise(text, candidate_ids)
            return ranking, scores, reason, text
        except Exception as e:  # pragma: no cover
            last_err = str(e)
            if attempt >= retries:
                break
            await asyncio.sleep(min(1.0 * (attempt + 1), 3.0))

    # Fallback: keep input order, zero scores.
    scores = {cid: 0.0 for cid in candidate_ids}
    return candidate_ids, scores, f"request_failed: {last_err[:120]}", last_text


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

    from scripts.build_wavetable_retrieval_baseline import _embed_clap

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


def _emit_spotcheck_audio(
    sample_id: str,
    query_audio_path: str,
    gt_names: list[str],
    ranked_names: list[str],
    name_scores: dict[str, float],
    candidate_audio: dict[str, Path],
    spotcheck_dir: Path,
    topk: int,
    best_gt_rank: int | None,
) -> None:
    out = spotcheck_dir / sample_id
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(query_audio_path, out / "00_target.wav")

    copied: set[str] = set()
    for i, name in enumerate(ranked_names[:topk], start=1):
        if name not in candidate_audio:
            continue
        copied.add(name)
        score = float(name_scores.get(name, 0.0))
        fname = f"{i:02d}_rank{i}_{_slugify(name)}_score{score:.4f}.wav"
        shutil.copyfile(candidate_audio[name], out / fname)

    for name in gt_names:
        if name in copied or name not in candidate_audio:
            continue
        shutil.copyfile(candidate_audio[name], out / f"90_gt_{_slugify(name)}.wav")

    meta = {
        "sample_id": sample_id,
        "query_audio_path": query_audio_path,
        "gt_wavetable_names": gt_names,
        "best_gt_rank": best_gt_rank,
        "top_ranked_names": ranked_names[:topk],
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


async def _evaluate_qwen(args: argparse.Namespace) -> None:
    rows_meta = _load_index_rows(args.index_meta)
    selected_by_name = _select_probe_rows_by_name(rows_meta)

    wavetable_lib = _load_wavetable_lib(args.wavetable_lib)
    candidate_probe_dir = args.out_dir / "candidate_probes"
    candidate_audio: dict[str, Path] = {}

    eval_entries = _build_eval_entries(args.outputs_root, args.query_key, args.max_samples)
    if not eval_entries:
        raise RuntimeError(
            f"No evaluation entries found under {args.outputs_root} with query key '{args.query_key}'."
        )

    ks = sorted(set(DEFAULT_KS + tuple(args.extra_k)))
    topk = max(args.topk, max(ks))
    prompt_pairwise = PROMPT_TEMPLATE

    score_cache_path = args.out_dir / "qwen_score_cache.jsonl"
    score_cache = _load_score_cache(score_cache_path)
    score_cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_lock = asyncio.Lock()

    # Candidate encoding cache to avoid repeated base64 work.
    candidate_b64: dict[str, str] = {}

    universe_names = sorted(selected_by_name.keys(), key=lambda x: x.lower())
    shortlist_data = None
    clap_model = clap_processor = None
    if args.candidate_source in {"clap_topn", "oracle_mix8"}:
        shortlist_data = _maybe_build_clap_shortlist_data(args.index_npy, rows_meta)
        clap_model, clap_processor = _load_clap(args.clap_device)

    sem = asyncio.Semaphore(max(1, args.qwen_concurrency))
    client = httpx.AsyncClient()

    per_sample_rows: list[dict[str, Any]] = []
    total_candidates_scored = 0
    total_cache_hits = 0
    total_requests = 0

    iterator: Any = eval_entries
    if not args.no_tqdm and tqdm is not None:
        iterator = tqdm(eval_entries, desc="Qwen eval", unit="sample", dynamic_ncols=True)

    try:
        for entry in iterator:
            gt_names = _extract_gt_wavetable_names(Path(entry["target_preset_path"]))
            if not gt_names:
                continue

            if args.candidate_source == "oracle_mix8":
                mix_size = args.candidate_limit if args.candidate_limit is not None else 8
                sid_seed = int(hashlib.sha1(entry["sample_id"].encode("utf-8")).hexdigest()[:8], 16)
                rng = random.Random(int(args.seed) + sid_seed)
                hard_pool = _clap_shortlist_names(
                    query_audio_path=Path(entry["query_audio_path"]),
                    shortlist_size=max(int(args.oracle_hard_pool), int(mix_size) * 2),
                    clap_model=clap_model,
                    clap_processor=clap_processor,
                    clap_device=args.clap_device,
                    shortlist_data=shortlist_data,
                    rows_meta=rows_meta,
                )
                candidate_names = _build_oracle_mix_candidates(
                    gt_names=gt_names,
                    universe_names=universe_names,
                    mix_size=int(mix_size),
                    rng=rng,
                    hard_negative_names=hard_pool,
                )
            elif args.candidate_source == "clap_topn":
                shortlist_size = args.candidate_limit if args.candidate_limit is not None else min(64, len(selected_by_name))
                candidate_names = _clap_shortlist_names(
                    query_audio_path=Path(entry["query_audio_path"]),
                    shortlist_size=max(1, int(shortlist_size)),
                    clap_model=clap_model,
                    clap_processor=clap_processor,
                    clap_device=args.clap_device,
                    shortlist_data=shortlist_data,
                    rows_meta=rows_meta,
                )
            else:
                candidate_names = universe_names
                if args.candidate_limit is not None:
                    candidate_names = candidate_names[: args.candidate_limit]

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

            target_b64 = _encode_wav_b64(Path(entry["query_audio_path"]))
            name_scores: dict[str, float] = {}
            cache_hits_this = 0
            ranked_names: list[str] = []
            listwise_reason = ""
            listwise_raw = ""

            if args.ranking_mode == "listwise":
                candidate_payload: list[tuple[str, str]] = []
                id_to_name: dict[str, str] = {}
                for i, cname in enumerate(candidate_names):
                    cid = f"C{i + 1}"
                    if cname not in candidate_b64:
                        candidate_b64[cname] = _encode_wav_b64(candidate_audio[cname])
                    candidate_payload.append((cid, candidate_b64[cname]))
                    id_to_name[cid] = cname

                prompt_listwise = LISTWISE_PROMPT_TEMPLATE.format(n=len(candidate_payload))
                total_requests += 1
                ranking_ids, score_by_id, listwise_reason, listwise_raw = await _rank_listwise_qwen(
                    client=client,
                    sem=sem,
                    server=args.qwen_server,
                    model=args.qwen_model,
                    prompt=prompt_listwise,
                    target_b64=target_b64,
                    candidate_payload=candidate_payload,
                    max_tokens=args.qwen_max_tokens,
                    temperature=args.qwen_temperature,
                    timeout_s=args.qwen_timeout_s,
                    retries=args.qwen_retries,
                )
                ranked_names = [id_to_name[cid] for cid in ranking_ids if cid in id_to_name]
                for cid, cname in id_to_name.items():
                    name_scores[cname] = float(score_by_id.get(cid, 0.0))
                ranked = [(name, name_scores.get(name, 0.0)) for name in ranked_names]
            else:
                tasks = []

                async def _score_one(candidate_name: str) -> tuple[str, QwenScore, bool]:
                    nonlocal total_requests
                    cpath = candidate_audio[candidate_name]
                    if candidate_name not in candidate_b64:
                        candidate_b64[candidate_name] = _encode_wav_b64(cpath)
                    key = _cache_key(
                        query_audio_path=entry["query_audio_path"],
                        candidate_name=candidate_name,
                        candidate_audio_path=str(cpath),
                        prompt=prompt_pairwise,
                        model=args.qwen_model,
                    )
                    cached = score_cache.get(key)
                    if cached is not None:
                        return candidate_name, cached, True

                    total_requests += 1
                    scored = await _score_pair_qwen(
                        client=client,
                        sem=sem,
                        server=args.qwen_server,
                        model=args.qwen_model,
                        prompt=prompt_pairwise,
                        target_b64=target_b64,
                        candidate_b64=candidate_b64[candidate_name],
                        max_tokens=args.qwen_max_tokens,
                        temperature=args.qwen_temperature,
                        timeout_s=args.qwen_timeout_s,
                        retries=args.qwen_retries,
                    )
                    async with cache_lock:
                        score_cache[key] = scored
                        with open(score_cache_path, "a") as cf:
                            cf.write(
                                json.dumps(
                                    {
                                        "cache_key": key,
                                        "query_audio_path": entry["query_audio_path"],
                                        "candidate_name": candidate_name,
                                        "candidate_audio_path": str(cpath),
                                        "model": args.qwen_model,
                                        "score": scored.score,
                                        "confidence": scored.confidence,
                                        "reason": scored.reason,
                                        "raw_text": scored.raw_text,
                                    }
                                )
                                + "\n"
                            )
                    return candidate_name, scored, False

                for cname in candidate_names:
                    tasks.append(_score_one(cname))

                results = await asyncio.gather(*tasks)
                for cname, scored, from_cache in results:
                    name_scores[cname] = float(scored.score)
                    if from_cache:
                        cache_hits_this += 1
                ranked = _top_name_scores(name_scores, topn=topk)
                ranked_names = [n for n, _ in ranked]

            total_cache_hits += cache_hits_this
            total_candidates_scored += len(candidate_names)

            hits = _compute_recall_hits(gt_names, ranked_names, ks)
            has_init = "Init" in gt_names
            gt_in_pool_names = [n for n in gt_names if n in set(candidate_names)]
            gt_in_pool = bool(gt_in_pool_names)
            best_gt_rank = _first_gt_rank(ranked_names, gt_names)
            selected_name = ranked_names[0] if ranked_names else ""
            selected_is_gt = selected_name in set(gt_names)

            if args.spotcheck_dir:
                _emit_spotcheck_audio(
                    sample_id=entry["sample_id"],
                    query_audio_path=entry["query_audio_path"],
                    gt_names=gt_names,
                    ranked_names=ranked_names,
                    name_scores=name_scores,
                    candidate_audio=candidate_audio,
                    spotcheck_dir=args.spotcheck_dir,
                    topk=max(1, args.spotcheck_topk),
                    best_gt_rank=best_gt_rank,
                )

            per_sample_rows.append(
                {
                    "sample_id": entry["sample_id"],
                    "archetype": entry["archetype"],
                    "query_audio_path": entry["query_audio_path"],
                    "target_preset_path": entry["target_preset_path"],
                    "gt_wavetable_names": gt_names,
                    "osc_count": len(gt_names),
                    "has_init": has_init,
                    "candidate_count": len(candidate_names),
                    "candidate_pool_names": candidate_names,
                    "gt_in_pool": gt_in_pool,
                    "gt_in_pool_names": gt_in_pool_names,
                    "selected_name": selected_name,
                    "selected_is_gt": selected_is_gt,
                    "best_gt_rank": best_gt_rank,
                    "cache_hits": cache_hits_this,
                    "top_names": ranked_names[: args.topk],
                    "top_name_scores": [float(name_scores.get(n, 0.0)) for n in ranked_names[: args.topk]],
                    "ranking_mode": args.ranking_mode,
                    "listwise_reason": listwise_reason if args.ranking_mode == "listwise" else "",
                    "listwise_raw": listwise_raw if args.ranking_mode == "listwise" else "",
                    **hits,
                }
            )

    finally:
        await client.aclose()
        if not args.no_tqdm and tqdm is not None:
            iterator.close()

    per_sample_rows.sort(key=lambda r: r["sample_id"])

    summary = _aggregate_eval(per_sample_rows, ks)
    n_rows = max(1, len(per_sample_rows))
    gt_in_pool_rate = float(sum(1 for r in per_sample_rows if r.get("gt_in_pool")) / n_rows)
    gt_top1_rate = float(sum(1 for r in per_sample_rows if r.get("selected_is_gt")) / n_rows)
    gt_top3_rate = float(
        sum(1 for r in per_sample_rows if isinstance(r.get("best_gt_rank"), int) and int(r["best_gt_rank"]) <= 3) / n_rows
    )
    gt_ranks = [int(r["best_gt_rank"]) for r in per_sample_rows if isinstance(r.get("best_gt_rank"), int)]
    mean_best_gt_rank = float(sum(gt_ranks) / len(gt_ranks)) if gt_ranks else None

    summary.update(
        {
            "query_source": "task_reference",
            "query_key": args.query_key,
            "scorer": f"qwen_omni_{args.ranking_mode}",
            "ranking_mode": args.ranking_mode,
            "qwen_server": args.qwen_server,
            "qwen_model": args.qwen_model,
            "qwen_concurrency": args.qwen_concurrency,
            "candidate_source": args.candidate_source,
            "candidate_limit": args.candidate_limit,
            "oracle_hard_pool": args.oracle_hard_pool,
            "seed": args.seed,
            "candidate_probe_count": len(selected_by_name),
            "candidate_probes_rendered": len(candidate_audio),
            "mean_candidates_per_sample": float(total_candidates_scored / max(1, len(per_sample_rows))),
            "gt_in_pool_rate": gt_in_pool_rate,
            "gt_top1_rate": gt_top1_rate,
            "gt_top3_rate": gt_top3_rate,
            "mean_best_gt_rank": mean_best_gt_rank,
            "cache_hits_total": int(total_cache_hits),
            "requests_sent": int(total_requests),
            "index_npy": str(args.index_npy),
            "index_meta": str(args.index_meta),
            "wavetable_lib": str(args.wavetable_lib),
            "topk": args.topk,
            "ks": ks,
            "target_r10": args.target_r10,
            "pass_r10": summary["overall"].get("r@10", 0.0) >= args.target_r10,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "spotcheck_dir": str(args.spotcheck_dir) if args.spotcheck_dir else None,
            "spotcheck_topk": args.spotcheck_topk,
        }
    )

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
    print(f"Requests sent (non-cache): {total_requests}")
    print(f"Cache hits: {total_cache_hits}")
    print(f"Rows: {rows_path}")
    print(f"Summary: {summary_path}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index-npy", type=Path, default=Path("outputs/wt_retrieval_baseline/wt_index.npz"))
    p.add_argument("--index-meta", type=Path, default=Path("outputs/wt_retrieval_baseline/wt_index_meta.json"))
    p.add_argument("--wavetable-lib", type=Path, default=Path("data/wavetable_lib.json"))
    p.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    p.add_argument("--query-key", default="gt_wav")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--extra-k", type=int, nargs="*", default=[])
    p.add_argument("--max-samples", type=int, default=32)
    p.add_argument("--target-r10", type=float, default=0.80)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/wt_retrieval_qwen_omni_baseline"))

    p.add_argument("--probe-archetype", default="lead")
    p.add_argument("--probe-tail-s", type=float, default=1.0)
    p.add_argument("--trim-min-duration-s", type=float, default=0.5)
    p.add_argument(
        "--ranking-mode",
        choices=["pairwise", "listwise"],
        default="pairwise",
        help="pairwise: one request per candidate. listwise: one request over all candidates in shared context.",
    )

    p.add_argument(
        "--candidate-source",
        choices=["all", "clap_topn", "oracle_mix8"],
        default="all",
        help=(
            "Candidate set source. 'all' scores all indexed wavetable names; "
            "'clap_topn' prefilters with CLAP; "
            "'oracle_mix8' builds a set with at least one GT + distractors."
        ),
    )
    p.add_argument(
        "--candidate-limit",
        type=int,
        default=None,
        help="Optional cap on candidates per sample (applies to all-source directly or as top-N for clap_topn).",
    )
    p.add_argument(
        "--clap-device",
        default="cuda:0",
        help="Used when --candidate-source is clap_topn or oracle_mix8.",
    )
    p.add_argument(
        "--oracle-hard-pool",
        type=int,
        default=64,
        help="When candidate-source=oracle_mix8, sample distractors from CLAP top-N hard negatives first.",
    )
    p.add_argument("--seed", type=int, default=1337, help="Seed for oracle candidate mixing.")

    p.add_argument("--qwen-server", default="http://localhost:8000")
    p.add_argument("--qwen-model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    p.add_argument("--qwen-concurrency", type=int, default=8)
    p.add_argument("--qwen-timeout-s", type=float, default=90.0)
    p.add_argument("--qwen-retries", type=int, default=2)
    p.add_argument("--qwen-max-tokens", type=int, default=80)
    p.add_argument("--qwen-temperature", type=float, default=0.0)
    p.add_argument(
        "--spotcheck-dir",
        type=Path,
        default=None,
        help="Optional directory to emit per-sample audio spotchecks (target + top ranked + GT clips).",
    )
    p.add_argument("--spotcheck-topk", type=int, default=3, help="How many top-ranked candidates to export per sample.")
    p.add_argument("--no-tqdm", action="store_true", help="Disable progress bars")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    asyncio.run(_evaluate_qwen(args))


if __name__ == "__main__":
    main()
