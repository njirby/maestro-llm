# maestro-llm

Research platform for audio-ML experiments, oriented toward an LLM agent that can **listen to audio and reason about it inside a DAW**. The audio captioning experiment here is one building block of a larger system.

## Vision

The larger goal is an agent that can:
1. **Listen** — embed audio with CLAP, understand what it sounds like
2. **Reason** — use an LLM to interpret and plan actions
3. **Act** — control REAPER via MCP tools (set FX params, load presets, render, compare)
4. **Iterate** — close the loop: render → listen → adjust → repeat

This repo starts with step 1: training a projection layer that bridges CLAP audio embeddings into an LLM's embedding space.

## Structure

```
maestro-llm/
├── maestro/                  # Shared library (pip-installable)
│   ├── audio/processing.py   # Audio loading and chunking
│   └── data/
│       ├── musiccaps.py      # MusicCaps dataset builders (source + prepared)
│       ├── download.py       # YouTube downloader (yt-dlp)
│       └── prepare_musiccaps.py  # Parallel 10s clip preparation + manifest
│
├── experiments/
│   └── audio_caption/        # Experiment: CLAP + projection + frozen Qwen
│       ├── config.py         # Dataclass config (loaded from YAML)
│       ├── model.py          # AudioLanguageModel
│       ├── collator.py       # DataCollator (audio → CLAP features → tokens)
│       └── train.py          # Training entry point
│
├── configs/
│   ├── audio_caption.yaml               # 9B defaults (memory-heavy)
│   └── audio_caption_qwen4b_stable.yaml  # known-good 4B config on 2x24GB GPUs
│
├── scripts/
│   ├── download_data.py      # Download source MusicCaps audio
│   ├── prepare_musiccaps.py  # Build prepared fixed 10s training dataset
│   ├── run_experiment.py     # Unified experiment launcher
│   ├── train_2gpu.sh         # 2 GPU launch via Accelerate
│   ├── train_2gpu_zero3.sh   # Generic 2 GPU ZeRO-3 launcher
│   ├── train_2gpu_zero3_qwen4b.sh  # Known-good 4B ZeRO-3 launcher (smoke/flexible)
│   └── train_2gpu_zero3_qwen4b_full.sh  # Full-dataset 4B ZeRO-3 launcher
│
├── docs/
│   └── audio_caption_prepared_data.md  # Prepared-data runbook
│
└── tests/
    └── test_smoke.py         # Fast smoke tests (no GPU or downloads needed)
```

## Setup

```bash
pip install -e ".[test]"
```

Also needs `ffmpeg` and `yt-dlp` on your PATH for data download.

## Quickstart

### 1. Download data

```bash
# Quick test — grab 10 clips:
python scripts/download_data.py --limit 10

# Full MusicCaps dataset:
python scripts/download_data.py --num_workers 8
```

### 2. Prepare canonical 10s training clips

This is required for stable multi-GPU training:

```bash
# Quick test:
python scripts/prepare_musiccaps.py --limit 16

# Full prepared dataset with explicit parallelism:
python scripts/prepare_musiccaps.py --num_workers 8
```

Artifacts are written to `data/prepared/musiccaps_v1/`:
- `clips/*.wav`
- `manifest.jsonl`
- `prepare_report.json`

### 3. Run the smoke tests

Verifies the pipeline without GPU or model downloads:

```bash
python -m pytest tests/ -v
```

### 4. Train

Single GPU (quick smoke test):
```bash
python scripts/run_experiment.py audio_caption \
    --dataset_limit 16 \
    --num_train_epochs 1
```

Recommended multi-GPU path (2× RTX 3090, known-good):
```bash
bash scripts/train_2gpu_zero3_qwen4b.sh --dataset_limit 16 --num_train_epochs 1
```

Full 4B training (prepared full dataset, default 5 epochs):
```bash
bash scripts/train_2gpu_zero3_qwen4b_full.sh
```

Generic multi-GPU launch:
```bash
bash scripts/train_2gpu_zero3.sh --config configs/audio_caption_qwen4b_stable.yaml
```

### 5. Monitor

```bash
tensorboard --logdir ./outputs/audio_caption_qwen4b/runs
```

## Config

Experiments are configured via YAML. Use:
- `configs/audio_caption_qwen4b_stable.yaml` for the validated 4B path.
- `configs/audio_caption.yaml` for the 9B baseline (likely OOM on 24GB GPUs).

Prepared-data flags:
- `data.prepared_dir` path to prepared manifest + clips.
- `data.use_prepared: true` to use fixed 10s prepared clips.
- `data.require_prepared: true` to fail fast when prepared data is missing.

Override on the CLI:

```bash
python scripts/run_experiment.py audio_caption \
    --learning_rate 1e-3 \
    --per_device_train_batch_size 2
```

## Architecture (audio_caption)

```
Prepared 10s clip (one per row)
  → ClapModel.get_audio_features()         [frozen]    → (1, 512)
  → nn.Linear(512, qwen_hidden_size)       [trainable] → (1, qwen_hidden_size)
  → injected at <audio> token positions
  → Qwen3.5 (4B/9B)                        [frozen]    → text description
```

Only the projection layer is trained. Checkpoints are written to your configured `output_dir` (for 4B stable config: `outputs/audio_caption_qwen4b/`).

## Adding a New Experiment

1. Create `experiments/<name>/` with at minimum `config.py`, `model.py`, `train.py`
2. Create `configs/<name>.yaml`
3. Register the experiment in `scripts/run_experiment.py`
4. Add tests in `tests/`

## Memory

- `Qwen/Qwen3.5-4B` is the known-good path for this repo on 2×24GB GPUs with the stable config.
- `Qwen/Qwen3.5-9B` is currently not viable with ZeRO-3 on 2×24GB GPUs in this stack.
  - Without gradient checkpointing: ZeRO-3 all-gather OOM on first step.
  - With gradient checkpointing: checkpoint metadata mismatch error in backward.
- Training runs in `bf16` (`fp16` disabled).

## Why Prepared Data Is Required

Without preparation, training-time collation can decode full-length source audio and create highly variable per-rank workloads. This causes large step-time skew and temporary GPU utilization imbalance (one rank waiting at synchronization barriers). Prepared fixed 10s clips remove this source of variance.

Legacy dynamic chunking is still available (`--no_use_prepared`) but not recommended for multi-GPU runs.
