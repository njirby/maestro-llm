#!/usr/bin/env python3
"""
Autoregressive inference for audio_caption checkpoints.

Supports:
- Auto-resolving the best checkpoint for a given run/output directory.
- Single audio snippet prediction (optional streaming token output).
- Batched prediction over prepared MusicCaps clips with ground-truth comparison.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer, ClapModel, ClapProcessor

from experiments.audio_caption.config import ExperimentConfig
from experiments.audio_caption.model import AUDIO_TOKEN, AudioLanguageModel
from maestro.audio.processing import CLAP_SAMPLE_RATE, load_and_chunk, load_audio_clip
from maestro.data.musiccaps import build_prepared_dataset

CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")


@dataclass
class EvalPoint:
    step: int
    eval_loss: float
    epoch: float | None
    source: str


@dataclass
class Sample:
    sample_id: str
    audio_path: str
    ground_truth: str | None


def resolve_qwen_hidden_size(qwen_model_id: str) -> int:
    config = AutoConfig.from_pretrained(qwen_model_id, trust_remote_code=True)
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is not None:
        return int(hidden_size)

    text_config = getattr(config, "text_config", None)
    hidden_size = getattr(text_config, "hidden_size", None)
    if hidden_size is not None:
        return int(hidden_size)

    cfg_dict = config.to_dict()
    text_cfg_dict = cfg_dict.get("text_config", {})
    hidden_size = text_cfg_dict.get("hidden_size")
    if hidden_size is not None:
        return int(hidden_size)

    raise ValueError(f"Could not resolve hidden size for qwen_model_id={qwen_model_id}")


def setup_tokenizer(tokenizer_source: str) -> tuple[Any, int]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if AUDIO_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [AUDIO_TOKEN]})
    audio_token_id = tokenizer.convert_tokens_to_ids(AUDIO_TOKEN)
    if audio_token_id is None or audio_token_id < 0:
        raise ValueError("Failed to resolve <audio> token id.")
    return tokenizer, int(audio_token_id)


def parse_step_from_checkpoint_dir(path: Path) -> int | None:
    m = CHECKPOINT_RE.match(path.name)
    if not m:
        return None
    return int(m.group(1))


def list_checkpoint_dirs(output_dir: Path) -> list[Path]:
    out: list[tuple[int, Path]] = []
    for p in output_dir.glob("checkpoint-*"):
        if not p.is_dir():
            continue
        step = parse_step_from_checkpoint_dir(p)
        if step is None:
            continue
        out.append((step, p))
    return [p for _, p in sorted(out, key=lambda x: x[0])]


def find_latest_event_file(run_dir: Path) -> Path:
    files = sorted(run_dir.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No TensorBoard event files found under {run_dir}")
    return files[-1]


def resolve_best_from_run_dir(run_dir: Path) -> EvalPoint | None:
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception:
        return None

    event_file = find_latest_event_file(run_dir)
    acc = event_accumulator.EventAccumulator(str(event_file))
    acc.Reload()

    scalar_tags = acc.Tags().get("scalars", [])
    if "eval/loss" not in scalar_tags:
        return None

    best: EvalPoint | None = None
    for point in acc.Scalars("eval/loss"):
        step = int(point.step)
        loss = float(point.value)
        candidate = EvalPoint(step=step, eval_loss=loss, epoch=None, source=f"events:{event_file.name}")
        if best is None or candidate.eval_loss < best.eval_loss or (
            candidate.eval_loss == best.eval_loss and candidate.step < best.step
        ):
            best = candidate
    return best


def resolve_best_from_trainer_states(output_dir: Path) -> EvalPoint | None:
    best: EvalPoint | None = None
    for ckpt_dir in list_checkpoint_dirs(output_dir):
        state_path = ckpt_dir / "trainer_state.json"
        if not state_path.is_file():
            continue
        data = json.loads(state_path.read_text(encoding="utf-8"))
        for row in data.get("log_history", []):
            step = row.get("step")
            eval_loss = row.get("eval_loss")
            if step is None or eval_loss is None:
                continue
            epoch = row.get("epoch")
            candidate = EvalPoint(
                step=int(step),
                eval_loss=float(eval_loss),
                epoch=float(epoch) if epoch is not None else None,
                source=f"state:{state_path.name}",
            )
            if best is None or candidate.eval_loss < best.eval_loss or (
                candidate.eval_loss == best.eval_loss and candidate.step < best.step
            ):
                best = candidate
    return best


def resolve_epoch_for_step(output_dir: Path, step: int) -> float | None:
    state_path = output_dir / f"checkpoint-{step}" / "trainer_state.json"
    if not state_path.is_file():
        return None
    data = json.loads(state_path.read_text(encoding="utf-8"))
    for row in data.get("log_history", []):
        if row.get("step") == step and row.get("eval_loss") is not None:
            epoch = row.get("epoch")
            return float(epoch) if epoch is not None else None
    return None


def resolve_output_dir(run_dir: Path | None, output_dir: Path | None, cfg: ExperimentConfig) -> tuple[Path, Path | None]:
    if run_dir is not None:
        if not run_dir.exists() and not run_dir.is_absolute():
            prefixed = Path("outputs") / run_dir
            if prefixed.exists():
                run_dir = prefixed
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run dir not found: {run_dir}")
        inferred = run_dir.parent.parent
        return inferred.resolve(), run_dir.resolve()

    if output_dir is not None:
        if not output_dir.exists() and not output_dir.is_absolute():
            prefixed = Path("outputs") / output_dir
            if prefixed.exists():
                output_dir = prefixed
        return output_dir.resolve(), None

    return Path(cfg.training.output_dir).resolve(), None


def resolve_best_checkpoint(output_dir: Path, run_dir: Path | None, explicit_step: int | None, explicit_ckpt: Path | None) -> tuple[Path, EvalPoint | None]:
    if explicit_ckpt is not None:
        if not explicit_ckpt.is_dir():
            raise FileNotFoundError(f"checkpoint dir does not exist: {explicit_ckpt}")
        return explicit_ckpt.resolve(), None

    if explicit_step is not None:
        ckpt = output_dir / f"checkpoint-{explicit_step}"
        if not ckpt.is_dir():
            raise FileNotFoundError(f"checkpoint step does not exist: {ckpt}")
        return ckpt.resolve(), None

    best = None
    if run_dir is not None:
        best = resolve_best_from_run_dir(run_dir)

    if best is None:
        best = resolve_best_from_trainer_states(output_dir)

    if best is None:
        checkpoints = list_checkpoint_dirs(output_dir)
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found under: {output_dir}")
        return checkpoints[-1].resolve(), None

    ckpt = output_dir / f"checkpoint-{best.step}"
    if not ckpt.is_dir():
        checkpoints = list_checkpoint_dirs(output_dir)
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found under: {output_dir}")
        fallback = checkpoints[-1].resolve()
        print(
            f"WARNING: resolved best step={best.step} ({best.eval_loss:.6f}) but no matching checkpoint dir. "
            f"Falling back to latest: {fallback}"
        )
        return fallback, best

    return ckpt.resolve(), best


def resolve_projection_path(
    output_dir: Path,
    checkpoint_dir: Path,
    explicit_projection: Path | None,
    best_eval: EvalPoint | None,
) -> Path | None:
    if explicit_projection is not None:
        if not explicit_projection.is_file():
            raise FileNotFoundError(f"Projection file not found: {explicit_projection}")
        return explicit_projection.resolve()

    if best_eval is not None:
        epoch = best_eval.epoch
        if epoch is None:
            epoch = resolve_epoch_for_step(output_dir, best_eval.step)
        if epoch is not None:
            epoch_idx = int(round(epoch))
            candidate = output_dir / f"projection_epoch{epoch_idx}.pt"
            if candidate.is_file():
                return candidate.resolve()

    step = parse_step_from_checkpoint_dir(checkpoint_dir)
    if step is not None:
        epoch = resolve_epoch_for_step(output_dir, step)
        if epoch is not None:
            candidate = output_dir / f"projection_epoch{int(round(epoch))}.pt"
            if candidate.is_file():
                return candidate.resolve()

    fallback = output_dir / "projection_final.pt"
    if fallback.is_file():
        return fallback.resolve()
    return None


def extract_projection_state(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if k == "projection.weight":
            cleaned["weight"] = v
        elif k == "projection.bias":
            cleaned["bias"] = v
        elif k.startswith("module.projection."):
            cleaned[k.split("module.projection.", 1)[1]] = v
        elif k.startswith("projection."):
            cleaned[k.split("projection.", 1)[1]] = v

    required = {"weight", "bias"}
    missing = required - set(cleaned)
    if missing:
        raise ValueError(f"Projection state is missing keys: {sorted(missing)}")
    return {"weight": cleaned["weight"], "bias": cleaned["bias"]}


def load_projection(model: AudioLanguageModel, projection_path: Path):
    raw = torch.load(projection_path, map_location="cpu")
    if not isinstance(raw, dict):
        raise ValueError(f"Unexpected projection file payload type: {type(raw)!r}")
    proj_state = extract_projection_state(raw)
    proj_state = {
        "weight": proj_state["weight"].to(dtype=model.projection.weight.dtype),
        "bias": proj_state["bias"].to(dtype=model.projection.bias.dtype),
    }
    model.projection.load_state_dict(proj_state, strict=True)


def normalize_for_match(text: str) -> str:
    return " ".join(text.lower().strip().split())


def chunked(items: list[Sample], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def load_samples(args: argparse.Namespace, cfg: ExperimentConfig) -> list[Sample]:
    if args.audio_path is not None:
        audio_path = Path(args.audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"audio_path does not exist: {audio_path}")
        sample_id = audio_path.stem
        return [Sample(sample_id=sample_id, audio_path=str(audio_path), ground_truth=args.ground_truth)]

    prepared_dir = args.prepared_dir or cfg.data.prepared_dir
    dataset = build_prepared_dataset(prepared_dir=prepared_dir, limit=None)
    rows = list(dataset)

    if args.offset > 0:
        rows = rows[args.offset :]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    if not rows:
        raise ValueError("No rows selected for inference.")

    samples: list[Sample] = []
    for idx, row in enumerate(rows):
        sample_id = str(row.get("ytid", f"sample_{args.offset + idx}"))
        samples.append(
            Sample(
                sample_id=sample_id,
                audio_path=str(row["audio_path"]),
                ground_truth=str(row.get("caption", "")),
            )
        )
    return samples


def build_generation_batch(
    batch_samples: list[Sample],
    tokenizer: Any,
    clap_processor: Any,
    audio_token_id: int,
    dynamic_chunking: bool,
    device: torch.device,
) -> dict[str, Any]:
    prefix_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    all_audio: list[Any] = []
    audio_token_counts: list[int] = []
    prompt_ids: list[list[int]] = []
    for sample in batch_samples:
        audio_chunks = load_and_chunk(sample.audio_path) if dynamic_chunking else [load_audio_clip(sample.audio_path)]
        all_audio.extend(audio_chunks)
        n_audio = len(audio_chunks)
        audio_token_counts.append(n_audio)
        prompt_ids.append([audio_token_id] * n_audio + prefix_ids)

    clap_inputs = clap_processor(audio=all_audio, sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt")
    input_features = clap_inputs.get("input_features")
    if input_features is None:
        raise ValueError("CLAP processor output missing 'input_features'.")

    clap_batch: dict[str, torch.Tensor] = {"input_features": input_features.float().cpu()}
    if clap_inputs.get("is_longer") is not None:
        clap_batch["is_longer"] = clap_inputs["is_longer"].cpu()
    if clap_inputs.get("attention_mask") is not None:
        clap_batch["attention_mask"] = clap_inputs["attention_mask"].cpu()

    max_len = max(len(ids) for ids in prompt_ids)
    pad_id = tokenizer.pad_token_id
    padded_ids = []
    padded_mask = []
    for ids in prompt_ids:
        pad_len = max_len - len(ids)
        padded_ids.append(ids + [pad_id] * pad_len)
        padded_mask.append([1] * len(ids) + [0] * pad_len)

    return {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(padded_mask, dtype=torch.long, device=device),
        "clap_input_features": clap_batch["input_features"],
        "clap_is_longer": clap_batch.get("is_longer"),
        "clap_attention_mask": clap_batch.get("attention_mask"),
        "audio_token_counts": torch.tensor(audio_token_counts, dtype=torch.long, device=device),
        "feature_tail_shape": tuple(clap_batch["input_features"].shape[1:]),
    }


def ends_with_suffix(token_ids: list[int], suffix: list[int]) -> bool:
    return len(suffix) > 0 and len(token_ids) >= len(suffix) and token_ids[-len(suffix) :] == suffix


def autoregressive_generate(
    model: AudioLanguageModel,
    tokenizer: Any,
    generation_batch: dict[str, Any],
    max_new_tokens: int,
    stream: bool,
) -> list[str]:
    input_ids = generation_batch["input_ids"]
    attention_mask = generation_batch["attention_mask"]
    batch_size = input_ids.shape[0]

    eos_token_id = tokenizer.eos_token_id
    suffix_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    feature_tail = generation_batch["feature_tail_shape"]
    empty_features = torch.empty((0, *feature_tail), dtype=generation_batch["clap_input_features"].dtype)
    zero_audio_counts = torch.zeros((batch_size,), dtype=torch.long, device=input_ids.device)

    generated_ids: list[list[int]] = [[] for _ in range(batch_size)]
    finished = torch.zeros((batch_size,), dtype=torch.bool, device=input_ids.device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            clap_input_features=generation_batch["clap_input_features"],
            clap_is_longer=generation_batch["clap_is_longer"],
            clap_attention_mask=generation_batch["clap_attention_mask"],
            audio_token_counts=generation_batch["audio_token_counts"],
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1)

        for _ in range(max_new_tokens):
            step_tokens = next_tokens.clone()
            if eos_token_id is not None:
                step_tokens = torch.where(finished, torch.full_like(step_tokens, eos_token_id), step_tokens)

            for i in range(batch_size):
                if finished[i].item():
                    continue
                tid = int(step_tokens[i].item())
                generated_ids[i].append(tid)

                if stream and batch_size == 1:
                    if eos_token_id is None or tid != eos_token_id:
                        piece = tokenizer.decode([tid], skip_special_tokens=False)
                        if piece != "<|im_end|>":
                            print(piece, end="", flush=True)

                hit_eos = eos_token_id is not None and tid == eos_token_id
                hit_suffix = ends_with_suffix(generated_ids[i], suffix_ids)
                if hit_eos or hit_suffix:
                    finished[i] = True

            if bool(finished.all()):
                break

            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device),
                ],
                dim=1,
            )

            outputs = model(
                input_ids=step_tokens.unsqueeze(1),
                attention_mask=attention_mask,
                clap_input_features=empty_features,
                audio_token_counts=zero_audio_counts,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1)

    texts: list[str] = []
    for ids in generated_ids:
        trimmed = list(ids)
        if suffix_ids and ends_with_suffix(trimmed, suffix_ids):
            trimmed = trimmed[: -len(suffix_ids)]
        if eos_token_id is not None and trimmed and trimmed[-1] == eos_token_id:
            trimmed = trimmed[:-1]
        text = tokenizer.decode(trimmed, skip_special_tokens=True).strip()
        texts.append(text)
    return texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autoregressive inference for audio caption checkpoints.")
    parser.add_argument("--config", type=str, default="configs/audio_caption_qwen4b_stable.yaml")

    parser.add_argument("--run-dir", type=str, default=None, help="Run dir (e.g. outputs/.../runs/<name>)")
    parser.add_argument("--output-dir", type=str, default=None, help="Checkpoint root (e.g. outputs/audio_caption_qwen4b)")
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--projection-path", type=str, default=None)

    parser.add_argument("--audio-path", type=str, default=None)
    parser.add_argument("--ground-truth", type=str, default=None)
    parser.add_argument("--prepared-dir", type=str, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=16)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--dynamic-chunking", action="store_true")
    parser.add_argument("--stream", action="store_true")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--clap-device", type=str, choices=["cpu", "cuda"], default=None)

    parser.add_argument("--output-json", type=str, default=None, help="Write results as a JSON array.")
    parser.add_argument("--output-jsonl", type=str, default=None, help="Write results as JSONL (one row per line).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_json is not None and args.output_jsonl is not None:
        raise ValueError("Use only one of --output-json or --output-jsonl.")
    cfg = ExperimentConfig.from_yaml(args.config)

    run_dir = Path(args.run_dir) if args.run_dir is not None else None
    output_dir = Path(args.output_dir) if args.output_dir is not None else None
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir is not None else None
    projection_path = Path(args.projection_path) if args.projection_path is not None else None

    resolved_output_dir, resolved_run_dir = resolve_output_dir(run_dir=run_dir, output_dir=output_dir, cfg=cfg)
    resolved_checkpoint_dir, best_eval = resolve_best_checkpoint(
        output_dir=resolved_output_dir,
        run_dir=resolved_run_dir,
        explicit_step=args.checkpoint_step,
        explicit_ckpt=checkpoint_dir,
    )
    resolved_projection = resolve_projection_path(
        output_dir=resolved_output_dir,
        checkpoint_dir=resolved_checkpoint_dir,
        explicit_projection=projection_path,
        best_eval=best_eval,
    )

    step = parse_step_from_checkpoint_dir(resolved_checkpoint_dir)
    print(f"Output dir: {resolved_output_dir}")
    if resolved_run_dir is not None:
        print(f"Run dir: {resolved_run_dir}")
    print(f"Checkpoint: {resolved_checkpoint_dir}")
    if step is not None:
        print(f"Checkpoint step: {step}")
    if best_eval is not None:
        print(f"Best eval/loss: {best_eval.eval_loss:.6f} at step {best_eval.step} ({best_eval.source})")
    if resolved_projection is not None:
        print(f"Projection: {resolved_projection}")
    else:
        print("Projection: <none>")

    samples = load_samples(args, cfg)
    if args.stream and len(samples) != 1:
        raise ValueError("--stream is only supported for single-sample inference.")
    if args.stream and args.batch_size != 1:
        raise ValueError("--stream requires --batch-size 1.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is not available.")

    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    clap_device = args.clap_device if args.clap_device is not None else cfg.model.clap_device

    tokenizer_source = str(resolved_checkpoint_dir) if (resolved_checkpoint_dir / "tokenizer.json").is_file() else cfg.model.qwen_model_id
    tokenizer, audio_token_id = setup_tokenizer(tokenizer_source)
    clap_processor = ClapProcessor.from_pretrained(cfg.model.clap_model_id)
    clap_model = ClapModel.from_pretrained(cfg.model.clap_model_id)

    resolved_llm_dim = resolve_qwen_hidden_size(cfg.model.qwen_model_id)
    if cfg.model.qwen_hidden_dim is not None and cfg.model.qwen_hidden_dim != resolved_llm_dim:
        raise ValueError(
            "Configured qwen_hidden_dim does not match model config hidden size. "
            f"configured={cfg.model.qwen_hidden_dim}, resolved={resolved_llm_dim}"
        )
    llm_dim = cfg.model.qwen_hidden_dim or resolved_llm_dim

    model = AudioLanguageModel(
        audio_token_id=audio_token_id,
        clap_model=clap_model,
        qwen_model_id=cfg.model.qwen_model_id,
        clap_dim=cfg.model.clap_dim,
        llm_dim=llm_dim,
        train_clap_encoder=cfg.model.train_clap_encoder,
        train_projection=cfg.model.train_projection,
        freeze_llm=cfg.model.freeze_llm,
        clap_device=clap_device,
        torch_dtype=torch_dtype,
    )
    model.resize_token_embeddings(len(tokenizer))

    if resolved_projection is not None:
        load_projection(model, resolved_projection)

    model.llm.to(device)
    model.projection.to(device=device, dtype=torch_dtype)
    model.config.use_cache = True
    model.llm.config.use_cache = True
    if hasattr(model.llm, "generation_config") and model.llm.generation_config is not None:
        model.llm.generation_config.use_cache = True
    model.eval()

    results: list[dict[str, Any]] = []
    total_with_gt = 0
    total_exact = 0

    for batch_samples in chunked(samples, args.batch_size):
        generation_batch = build_generation_batch(
            batch_samples=batch_samples,
            tokenizer=tokenizer,
            clap_processor=clap_processor,
            audio_token_id=audio_token_id,
            dynamic_chunking=args.dynamic_chunking,
            device=device,
        )
        predictions = autoregressive_generate(
            model=model,
            tokenizer=tokenizer,
            generation_batch=generation_batch,
            max_new_tokens=args.max_new_tokens,
            stream=args.stream,
        )
        if args.stream:
            print()

        for sample, pred in zip(batch_samples, predictions):
            gt = sample.ground_truth
            exact_match = None
            if gt is not None:
                total_with_gt += 1
                exact_match = normalize_for_match(pred) == normalize_for_match(gt)
                if exact_match:
                    total_exact += 1

            row = {
                "sample_id": sample.sample_id,
                "audio_path": sample.audio_path,
                "prediction": pred,
                "ground_truth": gt,
                "exact_match": exact_match,
                "checkpoint_dir": str(resolved_checkpoint_dir),
                "checkpoint_step": step,
                "projection_path": str(resolved_projection) if resolved_projection is not None else None,
            }
            results.append(row)

            print(f"[{sample.sample_id}]")
            print(f"  pred: {pred}")
            if gt is not None:
                print(f"  gt:   {gt}")
                print(f"  exact_match: {exact_match}")

    if total_with_gt > 0:
        acc = total_exact / total_with_gt
        print(f"\nSummary: exact_match={total_exact}/{total_with_gt} ({acc:.2%})")

    if args.output_jsonl is not None:
        out_path = Path(args.output_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
        print(f"Wrote results: {out_path}")
    if args.output_json is not None:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=True, indent=2)
            f.write("\n")
        print(f"Wrote results: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
