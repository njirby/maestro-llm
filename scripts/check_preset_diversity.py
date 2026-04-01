#!/usr/bin/env python3
"""
Acoustic diversity check for generated Vital presets.

Renders each generated .vital preset on a fixed reference MIDI clip,
extracts spectral/temporal features AND CLAP audio embeddings, and reports
pairwise diversity metrics.

CLAP embeddings (laion/larger_clap_general) give a perceptually-grounded
diversity score: high cosine distance in CLAP space means the presets sound
genuinely different to the model, not just spectrally different.

Usage:
    python scripts/check_preset_diversity.py \\
        --presets-dir outputs/gen_presets \\
        --output outputs/diversity_report.json \\
        --n 120 --per-archetype 20

    # Skip CLAP (spectral features only, faster):
    python scripts/check_preset_diversity.py --no-clap --per-archetype 20
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CLAP_MODEL_ID = "laion/larger_clap_general"
CLAP_SAMPLE_RATE = 48000  # CLAP expects 48 kHz

SAMPLE_RATE = 44100

# Fixed 4-bar reference melody (C major scale pattern, 90 BPM feel)
REFERENCE_NOTES = [
    # (pitch, velocity, start_s, end_s)
    (60, 90, 0.00, 0.40), (62, 85, 0.40, 0.80), (64, 88, 0.80, 1.20),
    (65, 82, 1.20, 1.60), (67, 90, 1.60, 2.00), (69, 85, 2.00, 2.40),
    (71, 88, 2.40, 2.80), (72, 92, 2.80, 3.20), (71, 85, 3.20, 3.60),
    (69, 88, 3.60, 4.00), (67, 82, 4.00, 4.50), (65, 85, 4.50, 5.00),
    (64, 88, 5.00, 5.50), (62, 82, 5.50, 6.00), (60, 90, 6.00, 6.80),
]


def _make_pretty_midi_notes():
    import pretty_midi
    notes = []
    for pitch, vel, start, end in REFERENCE_NOTES:
        notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=end))
    return notes


def extract_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract a 1D feature vector from a stereo audio array (2, N).
    Returns a float32 vector of length ~20.
    """
    # Mix to mono
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    n_bands = 6  # must match band_edges below
    n_features = 8 + n_bands  # 14 total
    if mono.size == 0:
        return np.zeros(n_features, dtype=np.float32)

    # RMS
    rms = float(np.sqrt(np.mean(mono ** 2)))

    # Spectral features via FFT
    n_fft = min(4096, mono.size)
    spectrum = np.abs(np.fft.rfft(mono[:n_fft]))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    power = spectrum ** 2
    total_power = power.sum() + 1e-12

    # Spectral centroid
    centroid = float((freqs * power).sum() / total_power)

    # Spectral bandwidth
    bandwidth = float(np.sqrt(((freqs - centroid) ** 2 * power).sum() / total_power))

    # Spectral flatness (Wiener entropy)
    log_s = np.log(spectrum + 1e-12)
    flatness = float(np.exp(log_s.mean()) / (spectrum.mean() + 1e-12))

    # Spectral rolloff (95%)
    cumsum = np.cumsum(power)
    rolloff_idx = np.searchsorted(cumsum, 0.95 * cumsum[-1])
    rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    # Zero crossing rate
    zcr = float(np.mean(np.abs(np.diff(np.sign(mono)))) / 2.0)

    # Energy in frequency bands (sub-bass, bass, mid, high-mid, presence, air)
    band_edges = [0, 80, 250, 2000, 4000, 8000, sr // 2]
    band_energies = []
    for lo, hi in zip(band_edges[:-1], band_edges[1:]):
        mask = (freqs >= lo) & (freqs < hi)
        band_energies.append(float(power[mask].sum() / total_power))

    # Attack time: time to reach 80% peak
    envelope = np.abs(mono)
    peak_val = envelope.max()
    if peak_val > 1e-6:
        attack_samples = np.argmax(envelope > 0.8 * peak_val)
        attack_time = float(attack_samples / sr)
    else:
        attack_time = 0.0

    # Decay steepness: RMS in last 25% vs first 25%
    quarter = max(1, len(mono) // 4)
    rms_start = float(np.sqrt(np.mean(mono[:quarter] ** 2)) + 1e-12)
    rms_end = float(np.sqrt(np.mean(mono[-quarter:] ** 2)) + 1e-12)
    decay_ratio = float(np.log(rms_end / rms_start + 1e-6))

    feat = np.array(
        [rms, centroid, bandwidth, flatness, rolloff, zcr,
         attack_time, decay_ratio] + band_energies,
        dtype=np.float32,
    )
    return feat


def pairwise_cosine_distance(features: np.ndarray) -> float:
    """Mean pairwise cosine distance between feature vectors (0=identical, 1=orthogonal)."""
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-12
    normed = features / norms
    gram = normed @ normed.T
    n = len(features)
    if n < 2:
        return 0.0
    triu = gram[np.triu_indices(n, k=1)]
    return float(1.0 - triu.mean())


def load_clap_model(device: str = "cpu"):
    """Load CLAP model and processor. Returns (model, processor) or (None, None)."""
    try:
        import torch
        from transformers import ClapModel, ClapProcessor
        print(f"Loading CLAP model {CLAP_MODEL_ID} ...")
        processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
        model = ClapModel.from_pretrained(CLAP_MODEL_ID).to(device)
        model.eval()
        print(f"  CLAP loaded on {device}")
        return model, processor
    except Exception as e:
        print(f"  CLAP load failed: {e} — skipping CLAP embeddings")
        return None, None


def embed_audio_clap(
    audio_44k: np.ndarray,
    model,
    processor,
    device: str = "cpu",
) -> np.ndarray:
    """
    Embed a stereo 44.1 kHz audio array with CLAP.

    Resamples to 48 kHz mono, runs through the CLAP audio encoder, and
    returns a float32 embedding vector (512-d for larger_clap_general).
    """
    import torch
    from scipy.signal import resample_poly
    from math import gcd

    # Mix to mono
    mono_44k = audio_44k.mean(axis=0) if audio_44k.ndim == 2 else audio_44k.copy()
    mono_44k = mono_44k.astype(np.float32)

    # Resample 44100 → 48000
    g = gcd(CLAP_SAMPLE_RATE, SAMPLE_RATE)
    up, down = CLAP_SAMPLE_RATE // g, SAMPLE_RATE // g
    mono_48k = resample_poly(mono_44k, up, down).astype(np.float32)

    # Pad or truncate to 10 s (CLAP's expected window)
    target_len = CLAP_SAMPLE_RATE * 10
    if len(mono_48k) < target_len:
        mono_48k = np.pad(mono_48k, (0, target_len - len(mono_48k)))
    else:
        mono_48k = mono_48k[:target_len]

    inputs = processor(
        audio=mono_48k,
        sampling_rate=CLAP_SAMPLE_RATE,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        emb = model.get_audio_features(**inputs)

    return emb.squeeze(0).cpu().numpy().astype(np.float32)


def _render_and_embed(
    synth,
    notes: list,
    clap_model,
    clap_processor,
    clap_device: str,
) -> tuple[float, np.ndarray, np.ndarray | None]:
    """Render notes, return (peak, spec_feat, clap_feat_or_None)."""
    from maestro.render.vital import _render_note_list
    audio = _render_note_list(synth, notes, SAMPLE_RATE, tail_s=1.0)
    peak = float(np.abs(audio).max())
    if peak < 0.001:
        spec_feat = np.zeros(14, dtype=np.float32)
        clap_feat = np.zeros(512, dtype=np.float32) if clap_model else None
    else:
        spec_feat = extract_features(audio, SAMPLE_RATE)
        clap_feat = (
            embed_audio_clap(audio, clap_model, clap_processor, clap_device)
            if clap_model else None
        )
    return peak, spec_feat, clap_feat


def _print_and_save_report(
    label: str,
    total: int,
    all_spectral: list,
    all_clap: list,
    all_peaks: list,
    output: Path,
    group_spectral: dict | None = None,
    group_clap: dict | None = None,
    group_peaks: dict | None = None,
) -> None:
    spec_arr = np.stack(all_spectral)
    overall_spec = pairwise_cosine_distance(spec_arr)
    overall_clap = None
    if all_clap:
        overall_clap = pairwise_cosine_distance(np.stack(all_clap))

    audible_rate = sum(1 for pk in all_peaks if pk > 0.005) / len(all_peaks)
    avg_peak = sum(all_peaks) / len(all_peaks)

    print(f"\nDiversity Report — {label}")
    print(f"  Total evaluated: {total}  audible={audible_rate:.0%}  avg_peak={avg_peak:.4f}")
    print(f"  Spectral overall: {overall_spec:.4f}")
    if overall_clap is not None:
        print(f"  CLAP    overall: {overall_clap:.4f}  ← perceptual score")

    if group_spectral:
        print(f"\n  {'Group':12s}  {'Spectral':>10s}  {'CLAP':>10s}  {'Audible':>8s}")
        for grp, sflist in group_spectral.items():
            gs = pairwise_cosine_distance(np.stack(sflist))
            gc_str = "n/a"
            if group_clap and grp in group_clap and group_clap[grp]:
                gc_str = f"{pairwise_cosine_distance(np.stack(group_clap[grp])):.4f}"
            pkgs = group_peaks.get(grp, []) if group_peaks else []
            ar = sum(1 for pk in pkgs if pk > 0.005) / len(pkgs) if pkgs else 0.0
            print(f"  {grp:12s}  {gs:>10.4f}  {gc_str:>10s}  {ar:>7.0%}")

    report = {
        "label": label,
        "total_presets": total,
        "audible_rate": audible_rate,
        "avg_peak": avg_peak,
        "spectral_diversity": {"overall": overall_spec},
        "clap_diversity": {
            "overall": overall_clap,
            "model": CLAP_MODEL_ID if overall_clap is not None else None,
        },
    }
    if group_spectral:
        report["spectral_diversity"]["per_group"] = {
            grp: pairwise_cosine_distance(np.stack(v))
            for grp, v in group_spectral.items()
        }
    if group_clap:
        report["clap_diversity"]["per_group"] = {
            grp: pairwise_cosine_distance(np.stack(v))
            for grp, v in group_clap.items() if v
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {output}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=Path("outputs/diversity_report.json"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-clap", action="store_true",
                   help="Skip CLAP embeddings (spectral features only, much faster)")
    p.add_argument("--clap-device", default="cuda",
                   help="Device for CLAP inference: cpu or cuda (default: cuda)")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--preset-files", type=Path, default=None, metavar="DIR",
                      help="Evaluate existing .vital files from DIR instead of generating")
    mode.add_argument("--per-archetype", type=int, default=30,
                      help="Generate N presets per archetype (default: 30)")

    p.add_argument("--n", type=int, default=180,
                   help="Max presets to sample when using --preset-files (default: 180)")
    p.add_argument("--wavetable-lib", type=Path, default=Path("data/wavetable_lib.json"),
                   help="Wavetable lib for generated mode (ignored with --preset-files)")
    args = p.parse_args()

    import vita
    from maestro.render.vital import _render_note_list  # noqa: F401 (imported for side effect)

    clap_model, clap_processor = None, None
    if not args.no_clap:
        clap_model, clap_processor = load_clap_model(device=args.clap_device)

    notes = _make_pretty_midi_notes()
    synth = vita.Synth()
    rng = random.Random(args.seed)

    if args.preset_files is not None:
        # ------------------------------------------------------------------ #
        # File mode: load existing .vital presets from disk                   #
        # ------------------------------------------------------------------ #
        preset_dir = args.preset_files.expanduser()
        all_files = sorted(preset_dir.glob("**/*.vital"))
        if not all_files:
            print(f"No .vital files found in {preset_dir}")
            return
        if len(all_files) > args.n:
            all_files = rng.sample(all_files, args.n)
        total = len(all_files)
        print(f"Evaluating {total} .vital files from {preset_dir} ...")

        all_spectral, all_clap, all_peaks = [], [], []
        start = time.time()
        silent = 0
        for i, path in enumerate(all_files):
            try:
                synth.load_preset(str(path))
            except Exception as e:
                print(f"  skip {path.name}: {e}")
                continue
            peak, sf, cf = _render_and_embed(synth, notes, clap_model, clap_processor, args.clap_device)
            all_peaks.append(peak)
            all_spectral.append(sf)
            if cf is not None:
                all_clap.append(cf)
            if peak < 0.001:
                silent += 1
            if (i + 1) % 20 == 0:
                elapsed = time.time() - start
                print(f"  {i+1}/{total}  silent={silent}  {elapsed:.0f}s elapsed")

        elapsed = time.time() - start
        print(f"\nDone in {elapsed:.1f}s  ({total / elapsed:.1f} presets/s)")
        _print_and_save_report(
            label=f"real_presets ({preset_dir.name})",
            total=len(all_spectral),
            all_spectral=all_spectral,
            all_clap=all_clap,
            all_peaks=all_peaks,
            output=args.output,
        )

    else:
        # ------------------------------------------------------------------ #
        # Generate mode: generate presets per archetype                       #
        # ------------------------------------------------------------------ #
        from maestro.synth.preset_gen import ARCHETYPES, generate_preset
        from maestro.synth.wavetable_lib import load_wavetable_lib

        wt_lib = []
        if args.wavetable_lib.exists():
            wt_lib = load_wavetable_lib(args.wavetable_lib)
            print(f"Loaded {len(wt_lib)} wavetables")

        n_per_arch = args.per_archetype
        total = n_per_arch * len(ARCHETYPES)
        print(f"Generating {total} presets ({n_per_arch} per archetype)...")

        all_spectral, all_clap, all_peaks = [], [], []
        group_spectral: dict[str, list] = {a: [] for a in ARCHETYPES}
        group_clap: dict[str, list] = {a: [] for a in ARCHETYPES}
        group_peaks: dict[str, list] = {a: [] for a in ARCHETYPES}
        start = time.time()

        for arch in ARCHETYPES:
            for _ in range(n_per_arch):
                preset = generate_preset(arch, rng, wavetable_lib=wt_lib)
                synth.load_json(json.dumps(preset))
                peak, sf, cf = _render_and_embed(synth, notes, clap_model, clap_processor, args.clap_device)
                all_peaks.append(peak)
                group_peaks[arch].append(peak)
                all_spectral.append(sf)
                group_spectral[arch].append(sf)
                if cf is not None:
                    all_clap.append(cf)
                    group_clap[arch].append(cf)

            audible = sum(1 for pk in group_peaks[arch] if pk > 0.005)
            print(f"  {arch:10s}: audible={audible}/{n_per_arch}  avg_peak={sum(group_peaks[arch])/n_per_arch:.4f}")

        elapsed = time.time() - start
        print(f"\nDone in {elapsed:.1f}s  ({total / elapsed:.1f} presets/s)")
        _print_and_save_report(
            label="generated",
            total=total,
            all_spectral=all_spectral,
            all_clap=all_clap,
            all_peaks=all_peaks,
            output=args.output,
            group_spectral=group_spectral,
            group_clap=group_clap,
            group_peaks=group_peaks,
        )


if __name__ == "__main__":
    main()
