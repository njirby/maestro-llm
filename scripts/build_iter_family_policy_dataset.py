#!/usr/bin/env python3
"""
Build an offline family-selection policy dataset for iterative preset matching.

For each generated target preset, this script simulates an iterative loop:
1) Start from the Vital init preset (with target wavetable/sample/LFO assets copied in)
2) At each step, enumerate candidate family actions (osc/filter/env/lfo/fx/modulation)
3) Apply each candidate to a copy of current preset, render audio, and measure similarity
4) Record per-family gain (delta similarity), pick the best family, update state

This produces training/evaluation data for learning "what to tweak first" based on
measured acoustic gain instead of manual assumptions.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from collections import defaultdict
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maestro.render.vital import SAMPLE_RATE, _load_vital, _render_note_list, make_probe_notes, trim_silence
from maestro.synth.path_gen import _param_family
from maestro.synth.preset_gen import ARCHETYPES, generate_preset
from maestro.synth.wavetable_lib import load_wavetable_lib

CLAP_MODEL_ID = "laion/larger_clap_general"
CLAP_SAMPLE_RATE = 48000
SKIP_KEYS = {"modulations", "lfos", "wavetables", "sample"}


def _load_param_ranges() -> dict[str, dict[str, float]]:
    path = Path(__file__).resolve().parents[1] / "maestro" / "synth" / "param_ranges.json"
    with open(path) as f:
        return json.load(f)


def _load_init_preset() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "maestro" / "synth" / "init_preset.json"
    with open(path) as f:
        return json.load(f)


PARAM_RANGES = _load_param_ranges()
INIT_PRESET = _load_init_preset()


def _extract_scalar_params(settings: dict[str, Any]) -> dict[str, float]:
    return {
        k: v
        for k, v in settings.items()
        if k not in SKIP_KEYS and not isinstance(v, (list, dict))
    }


def _normalize(name: str, native: float) -> float | None:
    r = PARAM_RANGES.get(name)
    if r is None:
        return None
    span = float(r["max"]) - float(r["min"])
    if span == 0:
        return 0.0
    return float((native - float(r["min"])) / span)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.reshape(-1).astype(np.float32)
    bb = b.reshape(-1).astype(np.float32)
    denom = (np.linalg.norm(aa) * np.linalg.norm(bb)) + 1e-12
    return float(np.dot(aa, bb) / denom)


def _extract_spectral_features(audio: np.ndarray, sr: int) -> np.ndarray:
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    n_bands = 6
    n_features = 8 + n_bands
    if mono.size == 0:
        return np.zeros(n_features, dtype=np.float32)

    rms = float(np.sqrt(np.mean(mono ** 2)))
    n_fft = min(4096, mono.size)
    spectrum = np.abs(np.fft.rfft(mono[:n_fft]))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    power = spectrum ** 2
    total_power = power.sum() + 1e-12

    centroid = float((freqs * power).sum() / total_power)
    bandwidth = float(np.sqrt(((freqs - centroid) ** 2 * power).sum() / total_power))
    log_s = np.log(spectrum + 1e-12)
    flatness = float(np.exp(log_s.mean()) / (spectrum.mean() + 1e-12))
    cumsum = np.cumsum(power)
    rolloff_idx = np.searchsorted(cumsum, 0.95 * cumsum[-1])
    rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])
    zcr = float(np.mean(np.abs(np.diff(np.sign(mono)))) / 2.0)

    band_edges = [0, 80, 250, 2000, 4000, 8000, sr // 2]
    band_energies = []
    for lo, hi in zip(band_edges[:-1], band_edges[1:]):
        mask = (freqs >= lo) & (freqs < hi)
        band_energies.append(float(power[mask].sum() / total_power))

    env = np.abs(mono)
    peak_val = env.max()
    attack_time = float(np.argmax(env > 0.8 * peak_val) / sr) if peak_val > 1e-6 else 0.0
    q = max(1, len(mono) // 4)
    rms_start = float(np.sqrt(np.mean(mono[:q] ** 2)) + 1e-12)
    rms_end = float(np.sqrt(np.mean(mono[-q:] ** 2)) + 1e-12)
    decay_ratio = float(np.log(rms_end / rms_start + 1e-6))

    return np.array(
        [rms, centroid, bandwidth, flatness, rolloff, zcr, attack_time, decay_ratio] + band_energies,
        dtype=np.float32,
    )


def _load_clap(device: str) -> tuple[Any, Any] | tuple[None, None]:
    try:
        import torch
        from transformers import ClapModel, ClapProcessor
    except Exception as e:
        print(f"WARNING: CLAP deps unavailable ({e}); falling back to spectral metric", file=sys.stderr)
        return None, None

    try:
        processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
        model = ClapModel.from_pretrained(CLAP_MODEL_ID).to(device)
        model.eval()
        return model, processor
    except Exception as e:
        print(f"WARNING: failed to load CLAP model ({e}); falling back to spectral metric", file=sys.stderr)
        return None, None


def _embed_clap(audio_44k: np.ndarray, model: Any, processor: Any, device: str) -> np.ndarray:
    import torch
    from scipy.signal import resample_poly

    mono_44k = audio_44k.mean(axis=0) if audio_44k.ndim == 2 else audio_44k.copy()
    mono_44k = mono_44k.astype(np.float32)

    g = gcd(CLAP_SAMPLE_RATE, SAMPLE_RATE)
    up, down = CLAP_SAMPLE_RATE // g, SAMPLE_RATE // g
    mono_48k = resample_poly(mono_44k, up, down).astype(np.float32)

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


class SimilarityScorer:
    def __init__(
        self,
        target_audio: np.ndarray,
        metric: str,
        hybrid_clap_weight: float,
        clap_model: Any = None,
        clap_processor: Any = None,
        clap_device: str = "cpu",
    ) -> None:
        self.metric = metric
        self.hybrid_clap_weight = max(0.0, min(1.0, hybrid_clap_weight))
        self.clap_model = clap_model
        self.clap_processor = clap_processor
        self.clap_device = clap_device

        self.target_spec = _extract_spectral_features(target_audio, SAMPLE_RATE)
        self.target_clap = None
        if self._use_clap():
            self.target_clap = _embed_clap(target_audio, clap_model, clap_processor, clap_device)

    def _use_clap(self) -> bool:
        return self.metric in {"clap", "hybrid"} and self.clap_model is not None and self.clap_processor is not None

    def score(self, audio: np.ndarray) -> float:
        spec = _extract_spectral_features(audio, SAMPLE_RATE)
        spec_sim = _cosine_similarity(spec, self.target_spec)
        if self.metric == "spectral" or not self._use_clap():
            return spec_sim

        clap = _embed_clap(audio, self.clap_model, self.clap_processor, self.clap_device)
        clap_sim = _cosine_similarity(clap, self.target_clap)
        if self.metric == "clap":
            return clap_sim
        return float(self.hybrid_clap_weight * clap_sim + (1.0 - self.hybrid_clap_weight) * spec_sim)


def _copy_init_for_target(target_preset: dict[str, Any]) -> dict[str, Any]:
    cur = copy.deepcopy(INIT_PRESET)
    for key in ("wavetables", "sample", "lfos", "modulations"):
        if key in target_preset["settings"]:
            cur["settings"][key] = copy.deepcopy(target_preset["settings"][key])
    return cur


def _render_preset_probe(synth: Any, preset: dict[str, Any], notes: list[Any]) -> np.ndarray:
    synth.load_json(json.dumps(preset))
    audio = _render_note_list(synth, notes, SAMPLE_RATE, tail_s=1.0)
    return trim_silence(audio, SAMPLE_RATE, min_duration_s=0.5)


def _collect_residuals(
    target_settings: dict[str, Any],
    current_settings: dict[str, Any],
    threshold: float,
    min_norm_delta: float,
) -> dict[str, list[tuple[str, float]]]:
    residuals: dict[str, list[tuple[str, float]]] = defaultdict(list)
    t_scalars = _extract_scalar_params(target_settings)
    c_scalars = _extract_scalar_params(current_settings)

    for name, t_native in t_scalars.items():
        c_native = c_scalars.get(name)
        if c_native is None:
            continue
        t_norm = _normalize(name, float(t_native))
        c_norm = _normalize(name, float(c_native))
        if t_norm is None or c_norm is None:
            continue
        diff = abs(t_norm - c_norm)
        if diff > threshold and diff >= min_norm_delta:
            residuals[_param_family(name)].append((name, diff))

    for fam in residuals:
        residuals[fam].sort(key=lambda x: x[1], reverse=True)
    return residuals


def _build_candidate_params(
    residuals: dict[str, list[tuple[str, float]]],
    per_family_budget: int,
) -> dict[str, list[str]]:
    candidates: dict[str, list[str]] = {}
    for fam, items in residuals.items():
        selected = [name for name, _ in items[:per_family_budget]]
        if selected:
            candidates[fam] = selected
    return candidates


def _apply_params_to_target(
    current_preset: dict[str, Any],
    target_settings: dict[str, Any],
    param_names: list[str],
) -> dict[str, Any]:
    out = copy.deepcopy(current_preset)
    for name in param_names:
        if name in target_settings:
            out["settings"][name] = target_settings[name]
    return out


def _step_bucket(step_idx: int, total_steps: int) -> str:
    if total_steps <= 1:
        return "early"
    frac = step_idx / max(1, total_steps - 1)
    if frac < 0.34:
        return "early"
    if frac < 0.67:
        return "mid"
    return "late"


def _aggregate_family_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fam_cand = defaultdict(int)
    fam_wins = defaultdict(int)
    fam_gain_sum = defaultdict(float)
    fam_win_gain_sum = defaultdict(float)
    fam_step_sum = defaultdict(float)
    bucket_gain = defaultdict(lambda: defaultdict(float))
    bucket_count = defaultdict(lambda: defaultdict(int))

    total_steps = 0
    for row in rows:
        steps = row.get("steps", [])
        n_steps = len(steps)
        for s_idx, step in enumerate(steps):
            total_steps += 1
            best = step.get("best_family")
            if best:
                fam_wins[best] += 1
                fam_step_sum[best] += s_idx
                fam_win_gain_sum[best] += float(step.get("best_gain", 0.0))

            bkt = _step_bucket(s_idx, n_steps)
            for cand in step.get("candidates", []):
                fam = cand["family"]
                gain = float(cand.get("gain", 0.0))
                fam_cand[fam] += 1
                fam_gain_sum[fam] += gain
                bucket_gain[fam][bkt] += gain
                bucket_count[fam][bkt] += 1

    families = sorted(set(fam_cand) | set(fam_wins))
    family_stats: dict[str, Any] = {}
    for fam in families:
        c = fam_cand[fam]
        w = fam_wins[fam]
        family_stats[fam] = {
            "candidate_count": c,
            "win_count": w,
            "win_rate": float(w / total_steps) if total_steps else 0.0,
            "avg_gain": float(fam_gain_sum[fam] / c) if c else 0.0,
            "avg_gain_when_won": float(fam_win_gain_sum[fam] / w) if w else 0.0,
            "avg_win_step_index": float(fam_step_sum[fam] / w) if w else None,
            "avg_gain_by_bucket": {
                b: (float(bucket_gain[fam][b] / bucket_count[fam][b]) if bucket_count[fam][b] else 0.0)
                for b in ("early", "mid", "late")
            },
        }

    ranked = sorted(
        family_stats.items(),
        key=lambda kv: (
            kv[1]["avg_gain_by_bucket"]["early"],
            kv[1]["avg_gain"],
            kv[1]["win_rate"],
        ),
        reverse=True,
    )
    return {
        "family_stats": family_stats,
        "recommended_order": [name for name, _ in ranked],
        "n_samples": len(rows),
        "n_steps_total": total_steps,
    }


def run_sample(
    sample_id: str,
    archetype: str,
    rng: random.Random,
    synth: Any,
    wavetable_lib: list[dict[str, Any]],
    metric: str,
    hybrid_clap_weight: float,
    clap_model: Any,
    clap_processor: Any,
    clap_device: str,
    max_steps: int,
    per_family_budget: int,
    threshold: float,
    min_norm_delta: float,
    min_gain_stop: float,
) -> dict[str, Any]:
    target = generate_preset(archetype, rng, wavetable_lib=wavetable_lib)
    current = _copy_init_for_target(target)
    target_settings = target["settings"]
    notes = make_probe_notes(archetype)

    target_audio = _render_preset_probe(synth, target, notes)
    scorer = SimilarityScorer(
        target_audio=target_audio,
        metric=metric,
        hybrid_clap_weight=hybrid_clap_weight,
        clap_model=clap_model,
        clap_processor=clap_processor,
        clap_device=clap_device,
    )

    current_audio = _render_preset_probe(synth, current, notes)
    current_score = scorer.score(current_audio)
    steps: list[dict[str, Any]] = []

    for step_idx in range(max_steps):
        residuals = _collect_residuals(
            target_settings=target_settings,
            current_settings=current["settings"],
            threshold=threshold,
            min_norm_delta=min_norm_delta,
        )
        if not residuals:
            break

        candidate_params = _build_candidate_params(residuals, per_family_budget=per_family_budget)
        if not candidate_params:
            break

        candidates: list[dict[str, Any]] = []
        best_family = None
        best_gain = -1e9
        best_score = current_score
        best_preset = None

        for family, params in candidate_params.items():
            cand_preset = _apply_params_to_target(current, target_settings, params)
            cand_audio = _render_preset_probe(synth, cand_preset, notes)
            cand_score = scorer.score(cand_audio)
            gain = float(cand_score - current_score)

            candidates.append({
                "family": family,
                "param_count": len(params),
                "params": params,
                "score_after": cand_score,
                "gain": gain,
            })
            if gain > best_gain:
                best_gain = gain
                best_family = family
                best_score = cand_score
                best_preset = cand_preset

        candidates.sort(key=lambda x: x["gain"], reverse=True)
        steps.append({
            "step": step_idx + 1,
            "score_before": current_score,
            "score_after": best_score,
            "best_gain": best_gain,
            "best_family": best_family,
            "residual_counts": {fam: len(items) for fam, items in residuals.items()},
            "candidates": candidates,
        })

        if best_preset is None:
            break
        current = best_preset
        current_score = best_score
        if best_gain <= min_gain_stop:
            break

    return {
        "sample_id": sample_id,
        "archetype": archetype,
        "max_steps": max_steps,
        "steps": steps,
        "final_score": current_score,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generate", type=int, default=60, help="Number of generated samples")
    p.add_argument("--archetypes", nargs="+", default=["bass", "lead", "pad"], choices=ARCHETYPES)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--per-family-budget", type=int, default=8)
    p.add_argument("--residual-threshold", type=float, default=0.05)
    p.add_argument("--min-norm-delta", type=float, default=0.02)
    p.add_argument("--min-gain-stop", type=float, default=1e-4)
    p.add_argument("--metric", choices=["spectral", "clap", "hybrid"], default="hybrid")
    p.add_argument("--hybrid-clap-weight", type=float, default=0.7)
    p.add_argument("--clap-device", default="cuda")
    p.add_argument("--wavetable-lib", type=Path, default=Path("data/wavetable_lib.json"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/iter_policy"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    wavetable_lib: list[dict[str, Any]] = []
    if args.wavetable_lib.exists():
        wavetable_lib = load_wavetable_lib(args.wavetable_lib)
        print(f"Loaded {len(wavetable_lib)} wavetables from {args.wavetable_lib}")

    clap_model, clap_processor = (None, None)
    if args.metric in {"clap", "hybrid"}:
        clap_model, clap_processor = _load_clap(args.clap_device)
        if clap_model is None and args.metric == "clap":
            print("ERROR: --metric clap requested but CLAP model unavailable.", file=sys.stderr)
            sys.exit(1)
        if clap_model is None and args.metric == "hybrid":
            print("Falling back to spectral-only scoring for hybrid metric.", file=sys.stderr)

    synth = _load_vital()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "policy_rows.jsonl"
    summary_path = out_dir / "policy_summary.json"

    rows: list[dict[str, Any]] = []
    start = time.time()
    for i in range(args.generate):
        arch = args.archetypes[i % len(args.archetypes)]
        sid = f"{arch}_{rng.randint(0, 2**31 - 1):08x}"
        row = run_sample(
            sample_id=sid,
            archetype=arch,
            rng=rng,
            synth=synth,
            wavetable_lib=wavetable_lib,
            metric=args.metric,
            hybrid_clap_weight=args.hybrid_clap_weight,
            clap_model=clap_model,
            clap_processor=clap_processor,
            clap_device=args.clap_device,
            max_steps=args.max_steps,
            per_family_budget=args.per_family_budget,
            threshold=args.residual_threshold,
            min_norm_delta=args.min_norm_delta,
            min_gain_stop=args.min_gain_stop,
        )
        rows.append(row)
        if (i + 1) % 5 == 0 or i + 1 == args.generate:
            elapsed = time.time() - start
            print(f"  processed {i + 1}/{args.generate} samples ({elapsed:.1f}s)")

    with open(rows_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    summary = _aggregate_family_stats(rows)
    summary["config"] = {
        "generate": args.generate,
        "archetypes": args.archetypes,
        "max_steps": args.max_steps,
        "per_family_budget": args.per_family_budget,
        "residual_threshold": args.residual_threshold,
        "min_norm_delta": args.min_norm_delta,
        "min_gain_stop": args.min_gain_stop,
        "metric": args.metric,
        "hybrid_clap_weight": args.hybrid_clap_weight,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote rows: {rows_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Recommended family order: {', '.join(summary['recommended_order'])}")


if __name__ == "__main__":
    main()
