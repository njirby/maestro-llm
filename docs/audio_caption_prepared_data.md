# Audio Caption Prepared-Data Runbook

## Why this exists

Multi-GPU ZeRO-3 runs become unstable when training-time collation decodes full-length audio
and dynamically chunks variable-duration tracks. Prepared data fixes this by giving every row
one canonical 10-second clip.

## End-to-end flow

1. Download source audio:

```bash
python scripts/download_data.py --limit 16
```

2. Build prepared dataset (parallelized):

```bash
python scripts/prepare_musiccaps.py --limit 16
```

3. Launch 2-GPU training:

```bash
bash scripts/train_2gpu_zero3_qwen4b.sh --dataset_limit 16 --num_train_epochs 1
```

For full 4B training (no dataset limit, default 5 epochs):

```bash
bash scripts/train_2gpu_zero3_qwen4b_full.sh
```

## 9B status on 2x24GB

- `Qwen/Qwen3.5-9B` is currently unstable/not viable in this repo with ZeRO-3 on 2x24GB:
  - no checkpointing: OOM during ZeRO-3 all-gather at first step
  - checkpointing enabled: PyTorch checkpoint metadata mismatch during backward
- Use 4B for stable end-to-end training and benchmarking on this hardware.

## Parallelism

`prepare_musiccaps.py` defaults to `CPU cores - 1` workers.
Override with:

```bash
python scripts/prepare_musiccaps.py --num_workers 8
```

## Artifacts

Under `data/prepared/musiccaps_v1/`:

- `clips/*.wav` (48k mono canonical 10s clips)
- `manifest.jsonl` (one row per MusicCaps item)
- `prepare_report.json` (summary + failure samples)

## Failure handling

The prepare step fails when failed rows exceed 1%.
Use report details to inspect bad rows:

```bash
cat data/prepared/musiccaps_v1/prepare_report.json
```

## Rebuild and validate

Rebuild existing clips:

```bash
python scripts/prepare_musiccaps.py --overwrite
```

Validate existing prepared dataset only:

```bash
python scripts/prepare_musiccaps.py --validate_only
```
