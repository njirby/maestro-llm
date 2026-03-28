#!/usr/bin/env python3
"""
Vibe-check pipeline for Omni->Lua generation.

Pipeline:
1) Sample rows from a JSONL split (expects Omni SFT schema).
2) Run `swift infer` with a base model + optional LoRA adapter checkpoint.
3) Save generated Lua + ground-truth Lua per sample.
4) Render generated Lua via Vita to MP3.
5) Save generated-vs-ground-truth assets and a summary report.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pretty_midi

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


TOKEN_BEATS = {
    "0": 0.0,
    "16t": 1.0 / 6.0,
    "t": 1.0 / 8.0,
    "s": 1.0 / 4.0,
    "8t": 1.0 / 3.0,
    "e": 1.0 / 2.0,
    "qt": 2.0 / 3.0,
    "e.": 3.0 / 4.0,
    "q": 1.0,
    "q.": 1.5,
    "h": 2.0,
    "h.": 3.0,
    "w": 4.0,
}
PITCH_CLASS = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}


@dataclass
class LuaParseResult:
    ok: bool
    notes: list[pretty_midi.Note]
    bpm: float | None
    ts: int
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-jsonl",
        type=Path,
        default=Path("data/prepared/omni_lua_sft_full_90_10/val.jsonl"),
        help="Input JSONL split to sample from (default: data/prepared/omni_lua_sft_full_90_10/val.jsonl).",
    )
    parser.add_argument(
        "--lua-dir",
        type=Path,
        default=Path("data/processed/reaper_tuples_lakh/luas"),
        help="Ground-truth Lua directory (default: data/processed/reaper_tuples_lakh/luas).",
    )
    parser.add_argument("--num-samples", type=int, default=4, help="How many samples to evaluate.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="LoRA checkpoint dir (checkpoint-*). If omitted, latest checkpoint under outputs/qwen25_omni_lora is used.",
    )
    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        default=Path("outputs/qwen25_omni_lora"),
        help="Used only when --checkpoint is omitted.",
    )
    parser.add_argument(
        "--model",
        default=(
            f"{Path.home()}/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-7B/"
            "snapshots/ae9e1690543ffd5c0221dc27f79834d0294cba00"
        ),
        help="Base model path or HF id.",
    )
    parser.add_argument("--model-type", default="qwen2_5_omni", help="MS-Swift model_type.")
    parser.add_argument("--template", default="qwen2_5_omni", help="MS-Swift template.")
    parser.add_argument("--swift-bin", default=".venv/bin/swift", help="Swift CLI path.")
    parser.add_argument("--python-bin", default=".venv/bin/python3", help="Python path for helper checks.")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Generation cap.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling top-p.")
    parser.add_argument(
        "--infer-backend",
        choices=["transformers", "vllm", "sglang", "lmdeploy"],
        default="vllm",
        help="Swift inference backend (default: vllm).",
    )
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        default="auto",
        help='vLLM TP size. Use integer or "auto" (default: auto = all visible GPUs).',
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.8,
        help="vLLM GPU memory utilization target.",
    )
    parser.add_argument("--cuda-visible-devices", default=None, help="Optional CUDA_VISIBLE_DEVICES override.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to outputs/omni_lua_vibecheck_<timestamp>.",
    )
    return parser.parse_args()


def _resolve_swift(swift_bin: str) -> str:
    if os.path.sep in swift_bin:
        p = Path(swift_bin).expanduser()
        if p.exists() and os.access(p, os.X_OK):
            return str(p.resolve())
        raise FileNotFoundError(f"Swift binary not executable: {p}")
    path = shutil.which(swift_bin)
    if path:
        return path
    raise FileNotFoundError(f"Could not find swift binary: {swift_bin}")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{i}: {exc}") from exc
    return rows


def _find_latest_checkpoint(root: Path) -> Path:
    checkpoints = sorted(root.rglob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* dirs found under: {root}")
    return checkpoints[-1].resolve()


def _adapter_tensor_count(checkpoint_dir: Path, python_bin: str) -> int | None:
    adapter = checkpoint_dir / "adapter_model.safetensors"
    if not adapter.is_file():
        return None
    code = (
        "from safetensors import safe_open\n"
        f"p=r'''{adapter}'''\n"
        "with safe_open(p, framework='pt', device='cpu') as f:\n"
        "    print(len(list(f.keys())))\n"
    )
    proc = subprocess.run([python_bin, "-c", code], capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    try:
        return int(proc.stdout.strip())
    except ValueError:
        return None


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_lua(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return raw
    fenced = re.findall(r"```(?:lua)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        ranked = sorted(
            fenced,
            key=lambda s: (
                ("MIDI_InsertNote" in s) or ("local bpm" in s) or ("TrackFX_AddByName" in s),
                len(s),
            ),
            reverse=True,
        )
        return ranked[0].strip()
    start_markers = ("-- wav:", "local bpm,ppb,ts", "local bpm, ppb, ts", "TrackFX_AddByName", "MIDI_InsertNote")
    positions = [raw.find(m) for m in start_markers if raw.find(m) >= 0]
    if positions:
        return raw[min(positions):].strip()
    return raw


def _assistant_from_messages(messages: Any) -> str | None:
    if not isinstance(messages, list):
        return None
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content
    return None


def _token_expr_beats(expr: str) -> float:
    parts = [p.strip() for p in expr.split("+") if p.strip()]
    if not parts:
        return 0.0
    total = 0.0
    for p in parts:
        if p not in TOKEN_BEATS:
            raise ValueError(f"unknown rhythm token: {p}")
        total += TOKEN_BEATS[p]
    return total


def _pitch_to_midi(pitch: str) -> int:
    m = re.fullmatch(r"([A-G]#?)(-?\d+)", pitch.strip())
    if not m:
        raise ValueError(f"invalid pitch: {pitch}")
    name, octave_s = m.groups()
    octave = int(octave_s)
    return PITCH_CLASS[name] + (octave + 1) * 12


def parse_tokenized_lua(lua_text: str) -> LuaParseResult:
    try:
        bpm_match = re.search(
            r"local\s+bpm\s*,\s*ppb\s*,\s*ts\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*[0-9]+(?:\.[0-9]+)?\s*,\s*([0-9]+)",
            lua_text,
        )
        if bpm_match:
            bpm = float(bpm_match.group(1))
            ts = int(bpm_match.group(2))
        else:
            bpm = 120.0
            ts = 4

        n_pattern = re.compile(
            r'n\(\s*"(?P<pitch>[A-G]#?-?\d+)"\s*,\s*m\(\s*(?P<bar>\d+)\s*,\s*(?P<beat>\d+)\s*,\s*"(?P<off>[^"]+)"\s*\)\s*,\s*"(?P<dur>[^"]+)"\s*,\s*(?P<vel>\d+)\s*\)'
        )
        matches = list(n_pattern.finditer(lua_text))
        if not matches:
            return LuaParseResult(ok=False, notes=[], bpm=bpm, ts=ts, error="no tokenized n(...) notes found")

        sec_per_beat = 60.0 / bpm
        notes: list[pretty_midi.Note] = []
        for m in matches:
            pitch = _pitch_to_midi(m.group("pitch"))
            bar = int(m.group("bar"))
            beat = int(m.group("beat"))
            off = _token_expr_beats(m.group("off"))
            dur = _token_expr_beats(m.group("dur"))
            vel = max(1, min(127, int(m.group("vel"))))

            start_beats = ((bar - 1) * ts) + (beat - 1) + off
            end_beats = start_beats + max(dur, 1.0 / 8.0)
            start = start_beats * sec_per_beat
            end = end_beats * sec_per_beat
            notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=end))

        return LuaParseResult(ok=True, notes=notes, bpm=bpm, ts=ts, error=None)
    except Exception as exc:
        return LuaParseResult(ok=False, notes=[], bpm=None, ts=4, error=str(exc))


def _write_mp3_ffmpeg(audio: np.ndarray, sample_rate: int, out_path: Path, bitrate: str = "192k") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio.T, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(audio.shape[0]),
        "-i",
        "pipe:0",
        "-vn",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        str(out_path),
    ]
    proc = subprocess.run(cmd, input=pcm16.tobytes(), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg mp3 encode failed: {proc.stderr.decode('utf-8', errors='ignore')}")


def _decode_audio_mono(path: Path, sample_rate: int = 44100) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed for {path}: {proc.stderr.decode('utf-8', errors='ignore')}")
    data = np.frombuffer(proc.stdout, dtype=np.float32)
    if data.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return data


def _similarity(pred_mp3: Path, gt_mp3: Path, sample_rate: int = 44100) -> dict[str, float | int | None]:
    pred = _decode_audio_mono(pred_mp3, sample_rate=sample_rate)
    gt = _decode_audio_mono(gt_mp3, sample_rate=sample_rate)
    n = min(pred.size, gt.size)
    if n < 512:
        return {
            "pearson_r": None,
            "mse": None,
            "n_samples_compared": int(n),
            "pred_duration_s": float(pred.size / sample_rate),
            "gt_duration_s": float(gt.size / sample_rate),
        }
    x = pred[:n].astype(np.float64)
    y = gt[:n].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    r = float(np.dot(x, y) / denom) if denom > 0 else 0.0
    mse = float(np.mean((pred[:n] - gt[:n]) ** 2))
    return {
        "pearson_r": r,
        "mse": mse,
        "n_samples_compared": int(n),
        "pred_duration_s": float(pred.size / sample_rate),
        "gt_duration_s": float(gt.size / sample_rate),
    }


def _copy_or_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def _run_swift_infer(
    swift_bin: str,
    model: str,
    model_type: str,
    template: str,
    infer_backend: str,
    vllm_tensor_parallel_size: int,
    vllm_gpu_memory_utilization: float,
    adapter_checkpoint: Path | None,
    input_jsonl: Path,
    result_jsonl: Path,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    cuda_visible_devices: str | None,
) -> None:
    cmd = [
        swift_bin,
        "infer",
        "--model",
        str(Path(model).expanduser().resolve()) if Path(model).expanduser().exists() else model,
        "--model_type",
        model_type,
        "--template",
        template,
        "--infer_backend",
        infer_backend,
        "--val_dataset",
        str(input_jsonl.resolve()),
        "--result_path",
        str(result_jsonl.resolve()),
        "--max_new_tokens",
        str(max_new_tokens),
        "--temperature",
        str(temperature),
        "--top_p",
        str(top_p),
        "--stream",
        "false",
        "--max_batch_size",
        "1",
        "--write_batch_size",
        "64",
        "--load_args",
        "false",
        "--dataset_num_proc",
        "1",
    ]
    if infer_backend == "vllm":
        cmd.extend(
            [
                "--vllm_tensor_parallel_size",
                str(vllm_tensor_parallel_size),
                "--vllm_gpu_memory_utilization",
                str(vllm_gpu_memory_utilization),
            ]
        )
    if adapter_checkpoint is not None:
        cmd.extend(["--adapters", str(adapter_checkpoint.resolve())])

    env = os.environ.copy()
    env["ENABLE_AUDIO_OUTPUT"] = "0"
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    print("Running inference command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def _count_all_gpus() -> int:
    proc = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    if proc.returncode != 0:
        return 1
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("GPU ")]
    return max(1, len(lines))


def _resolve_vllm_tp(tp_arg: str, cuda_visible_devices: str | None) -> int:
    raw = str(tp_arg).strip().lower()
    if raw != "auto":
        tp = int(raw)
        if tp <= 0:
            raise ValueError(f"--vllm-tensor-parallel-size must be > 0, got {tp}")
        return tp
    if cuda_visible_devices:
        ids = [x.strip() for x in cuda_visible_devices.split(",") if x.strip()]
        return max(1, len(ids))
    return _count_all_gpus()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be > 0")

    dataset_jsonl = args.dataset_jsonl.expanduser().resolve()
    if not dataset_jsonl.is_file():
        raise FileNotFoundError(f"Dataset split not found: {dataset_jsonl}")

    swift_bin = _resolve_swift(args.swift_bin)

    checkpoint: Path | None
    if args.checkpoint is not None:
        checkpoint = args.checkpoint.expanduser().resolve()
    else:
        checkpoint = _find_latest_checkpoint(args.checkpoints_root.expanduser().resolve())
    if checkpoint is not None and not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    adapter_tensor_count = _adapter_tensor_count(checkpoint, args.python_bin) if checkpoint is not None else None
    if adapter_tensor_count is not None and adapter_tensor_count == 0:
        print(f"WARNING: adapter checkpoint has 0 tensors and will be skipped: {checkpoint}")
        checkpoint = None
    resolved_vllm_tp = _resolve_vllm_tp(args.vllm_tensor_parallel_size, args.cuda_visible_devices)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else Path(f"outputs/omni_lua_vibecheck_{ts}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(dataset_jsonl)
    if not rows:
        raise ValueError(f"No rows in dataset split: {dataset_jsonl}")
    n = min(args.num_samples, len(rows))
    rng = random.Random(args.seed)
    chosen = rng.sample(rows, n)
    input_jsonl = out_dir / "infer_input.jsonl"
    _write_jsonl(input_jsonl, chosen)

    infer_result_jsonl = out_dir / "infer_result.jsonl"
    _run_swift_infer(
        swift_bin=swift_bin,
        model=args.model,
        model_type=args.model_type,
        template=args.template,
        infer_backend=args.infer_backend,
        vllm_tensor_parallel_size=resolved_vllm_tp,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        adapter_checkpoint=checkpoint,
        input_jsonl=input_jsonl,
        result_jsonl=infer_result_jsonl,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        cuda_visible_devices=args.cuda_visible_devices,
    )
    infer_rows = _load_jsonl(infer_result_jsonl)
    if len(infer_rows) != len(chosen):
        print(f"WARNING: infer row count mismatch (input={len(chosen)}, output={len(infer_rows)})")

    from maestro.render.vital import SAMPLE_RATE, _load_vital, _render_note_list

    synth = _load_vital()
    sample_reports: list[dict[str, Any]] = []
    for idx, row in enumerate(infer_rows):
        source_row = chosen[idx] if idx < len(chosen) else row
        sample_id = str(source_row.get("id") or row.get("id") or f"sample_{idx:03d}")
        sample_out = samples_dir / f"{idx:02d}_{sample_id}"
        sample_out.mkdir(parents=True, exist_ok=True)

        audios = source_row.get("audios") or row.get("audios") or []
        gt_audio = Path(audios[0]).expanduser().resolve() if audios else None
        if gt_audio is None or not gt_audio.exists():
            raise FileNotFoundError(f"Missing ground-truth audio for sample {sample_id}: {gt_audio}")

        gt_lua = row.get("labels")
        if not isinstance(gt_lua, str) or not gt_lua.strip():
            gt_lua = _assistant_from_messages(source_row.get("messages"))
        if not isinstance(gt_lua, str) or not gt_lua.strip():
            gt_lua = _assistant_from_messages(row.get("messages"))
        if not isinstance(gt_lua, str) or not gt_lua.strip():
            fallback = args.lua_dir.expanduser().resolve() / f"{sample_id}.lua"
            gt_lua = fallback.read_text(encoding="utf-8") if fallback.is_file() else ""

        pred_raw = str(row.get("response", ""))
        pred_lua = _extract_lua(pred_raw)

        gt_lua_path = sample_out / "ground_truth.lua"
        pred_lua_path = sample_out / "generated.lua"
        gt_lua_path.write_text(gt_lua, encoding="utf-8")
        pred_lua_path.write_text(pred_lua, encoding="utf-8")

        gt_mp3_path = sample_out / "ground_truth.mp3"
        _copy_or_symlink(gt_audio, gt_mp3_path)

        pred_parse = parse_tokenized_lua(pred_lua)
        pred_mp3_path = sample_out / "generated_vita.mp3"
        render_error = None
        if pred_parse.ok and pred_parse.notes:
            try:
                audio = _render_note_list(synth, pred_parse.notes, SAMPLE_RATE, tail_s=0.5)
                _write_mp3_ffmpeg(audio, SAMPLE_RATE, pred_mp3_path, bitrate="192k")
            except Exception as exc:
                render_error = str(exc)
        else:
            render_error = pred_parse.error or "lua_parse_failed"

        sim: dict[str, Any] | None = None
        if pred_mp3_path.exists():
            try:
                sim = _similarity(pred_mp3_path, gt_mp3_path, sample_rate=44100)
            except Exception as exc:
                sim = {"error": str(exc)}

        report_row = {
            "index": idx,
            "id": sample_id,
            "ground_truth_audio": str(gt_mp3_path.resolve()),
            "ground_truth_lua": str(gt_lua_path.resolve()),
            "generated_lua": str(pred_lua_path.resolve()),
            "generated_audio": (str(pred_mp3_path.resolve()) if pred_mp3_path.exists() else None),
            "lua_parse_ok": pred_parse.ok,
            "lua_parse_error": pred_parse.error,
            "pred_bpm": pred_parse.bpm,
            "pred_ts": pred_parse.ts,
            "pred_note_count": len(pred_parse.notes),
            "render_error": render_error,
            "similarity": sim,
        }
        (sample_out / "meta.json").write_text(json.dumps(report_row, indent=2), encoding="utf-8")
        sample_reports.append(report_row)

    summary = {
        "created_at": datetime.now().isoformat(),
        "dataset_jsonl": str(dataset_jsonl),
        "num_requested": args.num_samples,
        "num_inferred": len(infer_rows),
        "seed": args.seed,
        "checkpoint": (str(checkpoint) if checkpoint is not None else None),
        "adapter_tensor_count": adapter_tensor_count,
        "model": args.model,
        "swift_bin": swift_bin,
        "infer_backend": args.infer_backend,
        "vllm_tensor_parallel_size": resolved_vllm_tp,
        "infer_input_jsonl": str(input_jsonl.resolve()),
        "infer_result_jsonl": str(infer_result_jsonl.resolve()),
        "samples": sample_reports,
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    ok_renders = sum(1 for s in sample_reports if s["generated_audio"] is not None)
    print("\nVibe-check complete")
    print(f"  out_dir: {out_dir}")
    print(f"  report:  {report_path}")
    print(f"  inferred: {len(infer_rows)}")
    print(f"  rendered: {ok_renders}/{len(infer_rows)}")


if __name__ == "__main__":
    main()
