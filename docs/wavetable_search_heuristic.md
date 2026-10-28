# Wavetable search heuristic: exhaustive subset ranking by superposition

Status: proof of concept, 2026-10-28. Code: `maestro/v5/wt_rank.py`, checks in `scripts/wt_rank_check.py`.

## The problem

A searcher subagent auditions a chunk of the wavetable library (~32 tables) and replies with the subsets of its chunk
that, played together on the oscillators, come closest to the target. The builder has to decide that reply ahead of
time. Requirements:

- **Honest**: nothing forced. The reply is what a principled similarity ranking says, not "insert the GT".
- **GT ranks high by construction**, so the search can actually reach the answer.
- **Good non-GT runners-up**, so replies contain real near-alternatives.
- Combinations, not just single tables: up to 3 oscillators.

## What we tried first, and why it failed

Measured on 20 pilot20d targets. The metric is the rank of the GT osc-1 table among its 32-table chunk.

| Approach | Result |
|---|---|
| Raw probe (table on osc 1 of the init patch) vs the real target: what the searcher does today | top-3 1/20, median rank 18 |
| Same, with the target's FX off / FX + filters off | top-3 2/20, median 14 |
| Same, with only osc 1 on | top-3 6/20, median 9 |
| Same, with osc 1's own settings also reset to init (transpose, morph, distortion, unison, frame) | top-3 11/20, median 2. No single setting group mattered on its own; together they did. |
| Probes pre-shaped with the target's amp envelope, transpose and unison ("sketch") | top-3 3/20: no help |
| Precomputed per-table fingerprints (499×499 distance matrix), compared with in-context similarity | Spearman 0.48 median (−0.23..0.96), top-5 overlap 2.4/5. Which tables sound alike depends on the rest of the patch. |

In probe4 and probe5 the current searchers needed the oracle "rescue" in 3 of the 4 chunks that held a GT table.

**The blind test refuted the coarse metric.** A mean |log-mel| distance with a take-to-take noise floor (random phase
on) said that in half the samples 10–31 tables were "indistinguishable" from the GT in context. In a blind 5-way test
(`scripts/wt_blind_page.py`, served on :8774) the user picked the GT table in 10/10. The metric and the noise floor were
too blunt, not the ears.

## The principle: superposition

Vital's oscillator mix is a sum. With random phase off, each oscillator's output is deterministic. With nothing
nonlinear between the oscillators and the output, the rendered patch equals the sum of the oscillators rendered one
at a time. Checked by rendering a combination and the single-oscillator renders of the same stripped patch:

| Case | Residual (combo − Σ singles), relative energy |
|---|---|
| No cross-modulation, filters off | −137 to −139 dB (float rounding: exact) |
| Same with the filter on, drive 0.2–0.3 | −137 dB: the filter is linear here |
| Formant / Quantize / Bend / Pulse / Sync / Squeeze osc distortion | exact (these shape one oscillator only) |
| Osc RM ← another osc, small amount | −52 dB |
| **Osc FM ← another osc** | **−2 to +4 dB: broken** |

So this is a sparse / best-subset problem over a fixed dictionary with one atom per slot, and it can be solved
exhaustively:

1. Strip the GT patch: FX off, mod routes cleared, random phase off. Filters stay, since they are linear.
2. Render each candidate table once per GT oscillator slot: that slot's own GT settings, the other slots off. That is
   32 × (1..3) renders per target.
3. Score every assignment of candidates to slots, with an empty slot allowed, without rendering it:
   - Superseded: the first version prefiltered by waveform L2 (Gram matrix) and re-scored the best 300 by
     log-magnitude. It now uses one exact metric over every assignment; see "The metric" below.
4. The ranked list is the reply material. The GT assignment scores exactly 0. Runners-up are real near-alternatives,
   in order.

This is the k-best-solutions view of the search: the searcher's honest reply is the top-k of an exact objective over
its own chunk.

### Verified (`scripts/wt_rank_check.py`, 8 fresh POC targets, 1–3 oscillators)

- The GT assignment scores 0.00 in 8/8. It ranks first, or ties at 0.00 with a library table that is identical at
  frame 0 (wt004 ≡ wt028, wt417 ≡ wt247, wt022 ≡ wt019).
- Superposition-predicted scores vs actual renders of the top-3 non-GT assignments: difference 0.000 in all 24.
- Time: 5–24 s per target (render-bound; renders cached).

## The metric, checked against facts and ears

One metric scores everything: |dB| difference between the target's and a combination's sustain spectra (4 frames from
the middle of each note of the fixed pattern; a combination's spectrum is the sum of its slots' complex spectra, STFT
being linear), averaged over bins within 80 dB of the target's peak, **each bin weighted by its power^0.5** (the
louder of the two). Every assignment of the chunk's tables (or nothing) to the target's active oscillators is scored
exactly by it. There is no L2 prefilter (the earlier two-stage version kept only 3/10 of the true top 10 on some
targets), no per-table cost, and no slots the target leaves off: those are silent in its patch, so they only produce
ties. Subset sizes, smaller ones included, fall out of the metric alone.

The weighting was chosen by checks, not by taste. In `scripts/wt_metric_check.py` each target drops one oscillator
at a time (23 drops), and in `scripts/wt_drop_page.py` (:8774/drops.html) the user marked each A/B pair heard /
not heard, with 4 identical controls, all answered "same":

| bin weight power^p | Spearman(cost, energy share) | heard vs not-heard separation (AUC) |
|---|---|---|
| p = 0 (unweighted, the first version) | 0.59 | 0.90 |
| p = 0.3 | 0.76 | 0.98 |
| **p = 0.5 (adopted)** | 0.90 | **0.99** |
| p = 1 (energy) | 0.97 | 0.96 |

Unweighted, thousands of quiet bins outvote the harmonics that carry an oscillator: dropping an oscillator holding
nearly all the energy cost 1.87 there. Energy-weighted, quiet but audible layers are nearly free. Audibility is not a
function of energy share alone: masking matters. A −16 dB layer with distinct content was heard; a −7 dB layer under
the dominant one was not. At p = 0.5 every heard drop cost ≳ 2 and every unheard one ≲ 2, so ~2 is this listener's
just-noticeable difference on this scale. That is a measurement, not a threshold used anywhere.

## POC simplifications

These are decisions, not oversights (user, 2026-10-28). The first two are module flags in
`maestro/synth/preset_gen.py`; they take effect from the next manifest build, and existing pilots are unchanged.

- **`FIRST_FRAME_ONLY = True`**: every oscillator plays frame 0, with no frame spread and no LFO → wave-frame routes.
  Before this, only 3 of 43 active pilot20d oscillators were at frame 0 (frames ranged 5.9–248), while searcher probes
  always played frame 0. Tables silent at frame 0 were already never picked (`_SILENT_WAVETABLES`).
- **`NO_CROSS_MOD = True`**: no osc distortion mode that FM/RM-modulates from another oscillator (types 7, 8, 10, 11).
  It was used by 52% of multi-oscillator presets and is the one thing that breaks superposition. The single-oscillator
  distortion modes stay.
- **notes.json holds the MIDI as played, before any oscillator transpose.** With an oscillator transposed (preset_gen
  draws −24..+24 per archetype), the target sounds above or below notes.json, while the transcriber writes the played
  MIDI, not the heard pitch. Kept as is for now. The alternative is pinning the main oscillator's transpose to 0, so
  the heard melody equals notes.json and the other oscillators' transposes become timbre.

The random draws still happen, so every other parameter keeps its seeded value. Wavetable motion, cross-modulation and
heard-pitch transcription can come back later as their own data. Cross-modulation would need a joint-render fallback
for coupled oscillators; frame motion would make frame position an extra search dimension.

## Blind test v2: does the ranking's "close" match the ear?

`scripts/wt_blind_page2.py` (:8774/v2.html), 10 fresh POC targets, each played through its full patch: effects,
modulation and random phase as the patch has them. Five options each: the GT tables (another take), the ranking's two
closest other configurations, and two from deep in the ranking. 7 answered:

- Correct 3/7. Every miss picked one of the ranking's two **closest**; none picked a deep one.
- Where the closest wrong options scored ≥ 0.56, the GT was found (3/3). At ≤ 0.36 they fooled the listener, as did
  one at 2.12.
- One miss (Q10) scored **0.00**. Its tables were identical in that patch, though not on the fixed duplicate list, which
  was built with default oscillator settings. So sound-identity has to be judged in context; see below.
- So the ear's just-noticeable difference for whole-configuration swaps is roughly 0.4–2 on this scale.

## Open questions / next steps

- **Sound-identical in context = the same answer.** A configuration scoring ~0 against the target in its own patch is
  as correct as the GT tables (judge alternatives, the GT gate), not only the fixed `wavetable_frame0_dupes.json` list.
- **What the searcher hears vs. what ranks.** The ranking uses GT knowledge (slot settings, stripped context). The
  searcher's audition renders and listening notes stay what they are. The reply is the heuristic's top-k.
- **Replay memory.** 16 parallel searchers each held ~450 MB of base64 audio (88 clips × ~11 s, float32 stereo); one
  OpenCode process peaked at 25 GB. Fix: dispatch searchers in waves of 4. Audio stays float32 by decision.
