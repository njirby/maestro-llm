# Transcription LoRA — Arm 1: frozen audio tower (2026-08-11 → 08-12)

**Question:** can a single-task r128 LoRA on Qwen2.5-Omni-7B (audio tower frozen)
learn melody transcription from the repaired legacy data?

**Answer: it learns everything except absolute pitch.**

## Run

- Data: `transcription_aligned.jsonl` → 8,667 train / 200 val (`train.jsonl`/`val.jsonl`)
- Recipe: Megatron TP4, r128/α256, lr 1e-4, packing 32k, `all-linear` (thinker only;
  embed/lm_head/towers frozen), 4 epochs planned (1,046 iters)
- Stopped at iter ~250: GPU2 fell off the PCIe bus (`cudaErrorLaunchFailure`);
  card required a physical power cycle. checkpoint-250 saved but produces
  intermittently degenerate output — treat as suspect; checkpoint-200 is the
  reference. wandb: `maestro-sft/dntg4en4`.

## Eval (checkpoint-200, 30 held-out samples, real REAPER execution)

| metric | value |
|---|---|
| command parse rate (marker-tolerant) | 1.00 |
| code execution rate in container | 1.00 |
| onset-only F1 (pitch-blind, 80 ms) | 0.555 |
| duration abs error (median) | 89 ms |
| pitch F1 (exact) | **0.147** (median 0.02) |
| **pitch F1 (transposition-corrected)** | **0.530** (median 0.43) |

Signature example (`bass_02223c10`): generated melody = GT contour exactly
(interval steps −7,+5 preserved) transposed +7 semitones, rhythm stretched ×1.2.
Best-shift distribution across samples is scattered (+5,+6,+12,−8…; only 4/30
at 0) → **the model hears melodic contour + rhythm (~0.53–0.55) but has no
absolute pitch anchor** — the classic speech-encoder signature. One "perfect"
val score (`keys_0cc6f6d1`) was an exact train-set melody duplicate
(memorization); check melody overlap when reading top scores.

## Infrastructure findings (each cost hours; don't relearn)

1. **Evals must send the trained prefix.** Every record encodes with a
   452-token system+tools prompt (hermes agent template, from the record's
   `tools` field). Prompting with the bare opener sends the model OOD →
   mid-generation token-soup degeneration that mimics a broken model.
   Generate via `swift infer` on opener-truncated records (keep `tools`!),
   score with `scripts/score_swift_transcription_results.py`.
   **The serving harness (maestro-reaper-plugin) also omits system+tools — open fix.**
2. **The recipe corrupts the `<tool_call>` special token (id 151657).**
   Tuned model emits a different random rare token at exactly the marker
   positions (5/5); base model emits the marker correctly (5/5). LoRA-shifted
   hidden states can't hit frozen rare-token lm_head rows. Same recipe trained
   the original sft8k run → retroactively explains its unparseable tool calls
   ("messed up the transcription agent call every time") — it was never data-mix
   starvation. Fix candidates: cooler lr / α=r, `modules_to_save=[embed_tokens,
   lm_head]`, or plain-text markers. Short-term: marker-tolerant parsing.
3. Correction-loop records supervise the deliberately-wrong first attempts
   (by design) → trained-in ceiling on first-attempt F1. The model's own
   verify-listen loop could catch transposition errors at rollout time —
   keep both exact and transposition-corrected metrics.
4. Ops: vLLM kill leaves an orphaned `VLLM::EngineCore` holding GPU memory
   (find via `nvidia-smi --query-compute-apps`). ms-swift resolves repo ids
   via ModelScope (pass local snapshot paths). `--dataset-num-proc 32`.
   3-GPUs-at-100%+1-idle = NCCL spin-wait, not training; diagnose by power
   draw (~160 W spin vs 250 W+ compute) and memory growth.

## Next: Arm 2 — audio tower LoRA

Single change vs arm 1: `--freeze-vit false` (LoRA extends to tower linears).
Same lr/rank/alpha/packing for a clean A/B on the pitch-anchor question.
If pitch F1 (exact) doesn't move materially, the encoder can't be taught an
absolute-pitch reference by LoRA → arm 3 discussion (unfrozen encoder / 30B /
lower-level pitch features).

---

## Arm 2 mid-run reading (checkpoint-200, iteration 200/1,046, 2026-08-12)

Same recipe + `--freeze-vit false --freeze-aligner false` (LoRA on audio tower
+ aligner; 384 tower adapter tensors verified saving/learning/resuming).
Head-to-head vs arm 1 at identical iteration, same 30 val samples:

| metric | arm 1 frozen | arm 2 tower | Δ |
|---|---|---|---|
| pitch F1 exact | 0.147 (med 0.02) | 0.246 (med 0.13) | +67% |
| pitch F1 transp-corrected | 0.530 | 0.623 | +18% |
| onset-only F1 | 0.555 | 0.866 | +56% |
| in-key melodies | 4/30 | 6/29 | — |

Rhythm nearly solved; contour better; absolute-pitch anchor improving but
still the weak link at <1 epoch. Run continuing to 4 epochs (wandb 0u02eucq).
Val loss @250 also better than arm 1 (0.073 vs 0.083). Eval artifacts:
outputs/transcription_lora/eval_v2_ckpt200.json.
