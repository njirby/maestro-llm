# Maestro-LLM

An AI music production agent that collaborates with producers across the full REAPER DAW workflow. It controls the DAW programmatically, perceives and evaluates audio natively, learns the sonic behavior of any plugin, generates audio for gaps in an arrangement, and improves continuously through a combined SFT and RL training pipeline.

---

## Vision

Most AI music tools generate audio in isolation — they produce a stem, a loop, or a preset, and hand it back to the producer to integrate manually. Maestro-LLM is designed differently: it sits inside the production session, perceives the full context of what is already there, and acts directly on the DAW to move the work forward.

The goal is an agent that feels like a skilled collaborator — one who can be told "the verse feels empty" or "that reverb tail is way too long" or "make this sound more like a Fender Rhodes" and actually do something useful in response, not just describe what to do.

---

## System Architecture

The system is built on five pillars:

**Hands — REAPER control**
Full programmatic control of REAPER via reapy and ReaScript (Lua/Python). Every action a human can perform in the DAW is available as a tool call: create and delete tracks, insert and edit MIDI, load and configure plugins, set FX parameters, manage routing and sends, write automation, render regions, import audio. No wrapping layer — direct ReaScript API access at full speed.

**Ears — Qwen3 Omni**
Qwen3 Omni is the agent's perception model. It accepts raw audio input natively and produces text descriptions, comparisons, and judgments. It serves three roles: captioning (what does this sound like?), comparison (how do these two sounds differ?), and judging (does this output match the target?). The listen tool — which renders a region and passes it to Qwen3 Omni — is the agent's first action in almost every trajectory.

**Memory — Skill library, project memory, sample index**
- Skill library: a vector database of (plugin, parameter state, audio embedding, caption) tuples accumulated through plugin play sessions and production use. Enables semantic retrieval of known sounds by text description or audio similarity.
- Project memory: per-session episodic memory of creative decisions, rejected approaches, reference directions, and producer preferences. Persists across sessions.
- Sample index: CLAP embeddings of the producer's full local sample library, enabling text-to-sample and audio-to-audio search.

**Learning — SFT + RL pipeline**
A four-phase training pipeline detailed below. The core idea: SFT on stem-perturbation trajectories teaches the agent what good production looks like; RL on verifiable and aesthetic rewards teaches it to generalize and improve beyond the training distribution.

**Generation — ACE-Step 1.5**
ACE-Step 1.5 (`acestep-v15-sft`) runs locally and handles audio gaps that plugins cannot fill: continuation, inpainting, instrument replacement, and NL-conditioned generation. The agent decides when to reach for ACE-Step vs. a VSTi based on what the skill library can cover.

---

## The Core Loop

```
listen (Qwen3 Omni)
    → reason (LLM + skill library + project memory)
    → act (REAPER tools + ACE-Step)
    → evaluate (CLAP + Audiobox Aesthetics + Qwen3 judge)
    → learn
```

**Listen is always first.** Before any parameter change, plugin load, or MIDI edit, the agent renders a preview and listens. It calls listen again after significant actions to verify direction. A typical trajectory interleaves listen calls throughout:

```
listen()  →  "dense low-mids, reverb tail swamping the transients"
inspect_fx_chain(track="lead")  →  finds reverb with decay=4.2s
set_param(fx=2, param="Decay", val=1.8)
listen()  →  "cleaner, transients back, still slightly bright"
set_param(fx=0, param="HF_shelf", val=-1.5)
listen()  →  "matches target character"
```

---

## Features

### 1. Compose and Add Tracks
The agent reads the current project state (tempo, key, chord progression, existing instruments) and composes new parts that fit. It creates tracks, loads appropriate VSTi instruments using the skill library, writes MIDI, and routes correctly. Key and chord analysis run programmatically (Essentia) before any MIDI is written, constraining composition to the correct harmonic context.

### 2. Plugin Skill Exploration
Every installed plugin gets an unsupervised play session before production use. The agent systematically sweeps parameters, renders audio at each setting, captions with Qwen3 Omni, and embeds with CLAP. The resulting skill library maps the full sonic space of each plugin. During production, the agent retrieves skills by semantic text query ("warm analog pad with slow attack") or audio-to-audio similarity. Skills are versioned by plugin version and refreshed on update.

### 3. EQ and Audio Analysis
Two-track approach:
- **Programmatic (fast, precise)**: librosa and Essentia measure frequency balance, LUFS, dynamic range, spectral correlation, clipping, and phase issues. These run without any LLM.
- **Perceptual (slower, higher-level)**: Qwen3 Omni describes the audio holistically. Reliable for obvious problems and high-level character; less reliable for subtle engineering issues.

Music theory verification tools (key detection, chord analysis, out-of-key note detection, voice leading checks) run before rendering to catch compositional errors early.

### 4. Stylistic Recommendations
The agent renders a section, passes it to Qwen3 Omni, and cross-references the description with its knowledge of arrangement, genre, and production conventions. Recommendations cover arrangement density, instrumentation gaps, mix balance, automation shape, and dynamic variation across the song arc. Can be triggered on demand or run automatically at project save.

### 5. Melody Replication
Pipeline: Demucs source separation → BasicPitch (Spotify) audio-to-MIDI → key/tempo analysis → skill library lookup for closest timbre → MIDI import into REAPER → render → CLAP comparison to original → iterate. Works reliably for simple monophonic melodies. Complex polyphony and dense percussion reduce accuracy. Timbre match quality improves as the skill library grows.

### 6. Song Continuation and Inpainting
ACE-Step 1.5 handles sections that need audio generation: continuation from an existing arrangement, filling a specific gap (a 4-bar guitar solo slot), or replacing an analog instrument with generated audio. The agent renders surrounding context, passes it with a natural language prompt to ACE-Step, and imports the result as a REAPER audio item. Best for sections under ~30 seconds.

### 7. Take Selection
The agent generates N takes (via parameter variation or ACE-Step sampling), embeds all of them with CLAP, and ranks by combined score: similarity to target embedding, Audiobox production quality, and fit with the surrounding mix evaluated in context. Qwen3 Omni handles comparative judgment for aesthetic criteria ("which sounds most energetic?").

### 8. Sample Search
One-time CLAP indexing of the producer's full local sample library into a ChromaDB vector store. At production time: text query ("snappy rimshot, 180ms decay") or audio query (hum, reference clip) returns nearest neighbors by embedding similarity. External fallback: Freesound API (~600k sounds, programmatic semantic search). Final fallback: text web search to Splice, Looperman, or similar.

### 9. Session Organization
The agent sets up project structure from a natural language description: create standard track layouts, configure routing (drum bus, master bus, FX returns, parallel compression), assign colors and names by category, set tempo and key markers, configure render settings. No ML required — pure ReaScript. One of the highest immediate-value features for day-to-day workflow.

### 10. Lyric Co-creation
The agent grounds lyric suggestions in the current project state (tempo, key, mood, track names, arrangement density) rather than generating generically. Covers rhyme scheme analysis, syllable/meter fitting to a detected melody, theme development, genre-specific vocabulary, and line-by-line alternatives. With ACE-Step's lyric conditioning, generated lyrics can feed directly into vocal generation.

---

## Training Pipeline

### Phase 1 — Supervised Fine-Tuning on Stem Perturbations

**The core insight**: completed songs with stems provide free ground truth for musical coherence training. Remove a track, and the correct answer is the original track. The reward is verifiable without human annotation.

**Datasets**: Slakh2100 (2100 songs, MIDI + audio, CC BY 4.0), MUSDB18 (150 songs with stems), MedleyDB (122 professional recordings), Lakh MIDI Dataset (176k MIDI files).

**Perturbation types** (each generates distinct training examples from a single song):

| Perturbation | What the agent learns |
|---|---|
| Remove entire track | Instrumentation recommendation |
| Remove a section | Arrangement continuation |
| Remove automation | Dynamic/expression reasoning |
| Swap instrument timbre | Timbre/role matching |
| Remove FX processing | Mixing chain suggestion |
| Quantize MIDI too hard | Groove and humanization |
| Remove harmony parts | Harmonic voicing |
| Remove a full song section | Song structure and form |

**Trajectory generation**: A teacher model (Claude Opus or GPT-4o) generates ideal agent trajectories for each perturbed arrangement. Every trajectory starts with a listen call. Listen calls appear throughout, interleaved with tool calls. Qwen3 Omni captions and verifies output quality. CLAP similarity to the ground truth stem filters for quality. Only high-scoring trajectories enter the training set.

The synthetic conversation seeding is deliberate: the agent does not know what was removed. It reasons from the audio, exactly as a real producer would. This prevents pattern-matching ("bass is missing, add bass") and trains genuine audio reasoning.

**What SFT produces**: an agent that handles common repair tasks, writes musically coherent MIDI, makes reasonable instrumentation choices, and knows when to reach for ACE-Step vs. a VSTi.

---

### Phase 2 — RLVR on Tool Mastery (REAPER-bench)

SFT teaches the agent what good trajectories look like. RLVR teaches it to produce good trajectories it has never seen.

**REAPER-bench**: a held-out set of production tasks with verifiable outcomes. Examples:
- "Create a track named Bass at index 2" → verify `GetTrackName(GetTrack(0,2)) == "Bass"`
- "Set reverb decay to 2.1s on the master bus reverb return" → verify parameter value
- "Render bars 9–17 to /tmp/preview.wav" → verify file exists and duration matches

Reward: binary (did the expected state change happen?) plus guidance hints when the agent fails — following the Agent-RLVR approach that doubled SWE-bench performance.

**Unknown plugin exploration** is trained here by holding out a set of plugins from the skill library. The agent must figure them out from scratch:
- Enumerate parameters with `GetParamName`
- Hypothesize what each parameter does based on name and range
- Probe one parameter at a time, listen, assess direction
- Commit or revert based on whether it moved toward the target

Reward: CLAP similarity to target + bonus for converging under N steps - penalty for random parameter flailing. This trains a systematic exploration meta-skill that transfers to any new plugin encountered in production.

---

### Phase 3 — Aesthetic RL

**Sound design training**: Given a target sound (audio clip or text description), the agent manipulates plugin parameters and FX chains to match it.

Ground truth is generated two ways:
1. **Synthetic**: sweep own plugins across parameter space → render → Qwen3 Omni captions each sound. Triplet: `(plugin_state, audio, caption)`. Agent is trained to invert this mapping.
2. **Real-world**: Qwen3 Omni captions external audio (from any source). Agent approximates the sound with available plugins. Same Qwen3 instance judges the match.

**Critical design decision — loss in audio space, not parameter space**:

```
WRONG:  L = ||param_agent - param_ground_truth||²
RIGHT:  L = 1 - CLAP_similarity(audio_agent, audio_target)
           + (1 - Qwen3_judge_score(caption_agent, caption_target))
```

Two different parameter configurations that produce perceptually identical sounds should receive equal reward. Constraining to parameter space breaks cross-plugin generalization entirely.

For MIDI: note accuracy, timing error, and velocity MAE are verifiable and used directly.

**Composite reward**:
```
reward = α * CLAP_similarity(output, target)
       + β * Audiobox_production_quality(output)
       + γ * Qwen3_judge_score(caption(output), caption(target))
       + δ * verifiable_tool_reward(api_calls_succeeded)
```

Qwen3 judge reasoning ("the output is too bright, the filter cutoff needs to come down") is fed back as guidance, not just the scalar score. This produces richer learning signal and faster convergence.

---

### Phase 4 — ACE-Step Fine-Tuning

By Phase 4, the system has real conversation-conditioned trajectories from production sessions. ACE-Step is fine-tuned on:
- **Input**: surrounding arrangement audio + producer conversation context + gap duration/position
- **Target**: the ground truth stem (from stem perturbation dataset) or producer-approved generation

The conversation context gives ACE-Step richer conditioning than bare genre tags — it knows the harmonic context, the arrangement role, the stylistic direction, and what the producer specifically asked for.

---

## Skill Library

The skill library is the agent's long-term memory for plugin knowledge — the key component that separates competent from expert behavior.

**Structure per entry**:
```json
{
  "plugin": "Valhalla Shimmer v1.5.1",
  "param_state": {"Size": 0.87, "Feedback": 0.62, "Mix": 0.40},
  "audio_embedding": [...],
  "caption": "Long diffuse shimmer, floats behind the mix, infinite-feeling sustain",
  "tags": ["ambient", "pads", "long-tail", "shimmer"],
  "discovered": "2025-09-12",
  "plugin_version": "1.5.1"
}
```

**Bootstrap**: Before first production use, the agent runs unsupervised play sessions with every installed plugin. Systematic parameter sweeping, one parameter varied at a time, Qwen3 Omni captions each rendered sound, CLAP embeds it. Runs in the background once per plugin install.

**Retrieval**: Semantic text query → CLAP text embedding → nearest neighbor search. Or: audio clip → CLAP audio embedding → audio-to-audio search. ChromaDB handles both.

**Growth**: Every time the agent explores an unknown plugin during RL or production use, discovered mappings are saved automatically. Unknown plugins become known plugins over time.

**Versioning**: Skills are tagged with plugin version. On plugin update, affected skills are flagged for re-verification.

---

## Sample Search

```
Producer: "I need a rimshot that's tighter than what I have"

1. Embed query text via CLAP → search local library → top-5 candidates
   OR: render reference sound → embed audio → audio-to-audio search

2. Agent listens to each candidate (quick preview render in context)

3. Selects best fit → imports into REAPER at correct position and tempo

4. If nothing fits locally → Freesound API (~600k sounds, semantic search)

5. If still nothing → web search to Splice / Looperman
```

One-time indexing pass on the full local library. Incremental re-indexing when new samples are added. No manual tagging required.

---

## ACE-Step Integration

ACE-Step 1.5 (`acestep-v15-sft`) runs locally — no API cost, no internet required after download.

**Use cases**: continuation, inpainting, instrument replacement, NL-conditioned generation.

**Workflow**:
```
Agent identifies gap in REAPER
    → renders surrounding context to WAV
    → passes (context_audio, NL_prompt, duration) to ACE-Step
    → ACE-Step generates fill
    → agent imports result as audio item at correct position
    → renders combined result in context
    → Qwen3 Omni evaluates fit → iterate if needed
```

**Current limitation**: continuation uses repaint rather than mask-aware training. Coherence can drift over sections longer than ~30 seconds.

---

## Implementation Phases

Sequenced to get real producer feedback as early as possible. Each phase produces something useful and generates the infrastructure the next phase needs.

| Phase | Work | Deliverable |
|---|---|---|
| 0 — Foundation | reapy bridge, tool library, Qwen3 listen, session templates, CLAP sample indexer | Usable tool, first producer feedback |
| 1 — Plugin play sessions | Automated parameter sweeping, skill library bootstrap | Agent knows its plugins |
| 2 — SFT + first model | Stem perturbation pipeline, teacher trajectories, fine-tune Qwen2.5-14/32B | First trained agent in producers' hands |
| 3 — RLVR | REAPER-bench tasks, Agent-RLVR loop, unknown plugin exploration RL | Reliable execution, learns any new plugin |
| 4 — Aesthetic RL | CLAP + Audiobox + Qwen3 reward, DPO on preference data | Aesthetic taste, per-producer personalization |
| 5 — ACE-Step fine-tuning | Conversation-conditioned trajectory fine-tuning | Tailored audio generation |

---

## Key Technologies

| Component | Technology | Purpose |
|---|---|---|
| REAPER control | reapy + ReaScript (Lua/Python) | Full DAW manipulation |
| Audio perception | Qwen3 Omni | Listen, caption, compare, judge |
| Audio generation | ACE-Step 1.5 (acestep-v15-sft) | Continuation, inpainting, NL-conditioned generation |
| Source separation | Demucs | Isolate stems from mixed audio |
| Audio-to-MIDI | BasicPitch (Spotify) | Melody transcription |
| Audio quality reward | Meta Audiobox Aesthetics | Production quality reward model |
| Audio-text embedding | LAION CLAP | Similarity search, reward signal, sample indexing |
| Music theory tools | Essentia + music21 | Key/chord detection, note verification, voice leading |
| Sample search | ChromaDB/FAISS + Freesound API | Local + external sample retrieval |
| RL framework | verl / OpenRLHF (GRPO) | Agent RL training |
| SFT training | Hugging Face TRL SFTTrainer | Supervised fine-tuning on trajectories |
| Stem datasets | Slakh2100, MUSDB18, MedleyDB | SFT data source |
| Vector memory | ChromaDB | Skill library, project memory, sample index |

---

## Safety and State Management

- Destructive operations (delete track, bounce in place, clear automation, overwrite recordings) always require explicit confirmation
- Agent snapshots project state before any exploration session or multi-step editing pass
- Agent maintains its own undo stack above REAPER's native undo, scoped to agent-initiated changes
- When skill library coverage is low for a requested task, agent surfaces this rather than acting with false confidence

---

## Open Problems

**Musical coherence at the macro level**: The stem perturbation approach trains coherence within an arrangement. Reasoning about song-level arc — how tension and release should evolve over 3 minutes — requires longer-horizon trajectory data that is harder to collect synthetically.

**Perceptual subtlety**: Qwen3 Omni handles high-level audio description well. Precise mix engineering feedback ("the 400Hz region is masking the kick's fundamental") is less reliable. Programmatic measurement tools partially compensate but the gap between LLM perception and trained-ear perception remains real.

**ACE-Step long-form coherence**: Sections longer than ~30 seconds can drift stylistically. Mask-aware training (not yet released) would address this.

**Timbre matching ceiling**: Melody replication quality is bounded by skill library coverage. Coverage improves over time but starts sparse.

**Copyright boundaries**: Melody replication and sample approximation raise copyright questions the system does not currently reason about. Originality divergence modes and license-awareness for sample retrieval are needed before broad deployment.

---

## Related Work

- [Voyager (Wang et al., 2023)](https://arxiv.org/abs/2305.16291) — skill library accumulation for open-ended embodied agents
- [Agent-RLVR (Scale AI, 2025)](https://arxiv.org/abs/2506.11425) — RLVR with guidance hints for software engineering agents
- [DAWZY (Elkins et al., NeurIPS 2025)](https://arxiv.org/abs/2512.03289) — LLM-based natural language control of REAPER
- [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) — audio generation with continuation and inpainting
- [Meta Audiobox Aesthetics](https://arxiv.org/abs/2502.05139) — production quality reward model
- [LAION CLAP](https://github.com/LAION-AI/CLAP) — contrastive language-audio pretraining
- [BasicPitch (Spotify)](https://basicpitch.spotify.com) — audio-to-MIDI transcription
- [Slakh2100](http://www.slakh.com) — synthesized multi-track dataset for source separation
