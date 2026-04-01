"""
Demo: generate N-step preset paths + render audio for each step.

Produces under outputs/iter_demo/:
  {sample_id}_gt.wav                 — ground truth preset audio
  {sample_id}_step{N}.wav            — cumulative preset audio after step N (all N steps)
  {sample_id}_step{N}.vital          — cumulative preset JSON
  {sample_id}_target.vital           — target preset JSON
  {sample_id}_conversation.json      — conversation in agentic tool-call format
  demo_summary.txt                   — human-readable summary
  demo_train.jsonl                   — combined ms-swift JSONL

Conversation format: the model uses bash tool calls with real reapy Python.
  Search:  enumerate fx.params filtered by keyword → see display names + values
  Set:     fx.params["Display Name"].value = normalized_val
  Listen:  python scripts/reaper_render_probe.py && aplay /tmp/probe.wav
"""

import json
import random
import sys
import os
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from maestro.synth.path_gen import generate_preset_path, compare_preset_path, PARAM_PRIORITY_PREFIXES
from maestro.synth.wavetable_lib import load_wavetable_lib
from maestro.render.vital import (
    make_probe_notes,
    render_notes,
)

_LISTEN_TEXTS = [
    "Let me hear how that changed the sound.",
    "Rendering to check the result.",
    "Let me listen to the current state.",
    "Checking how this sounds now.",
    "Let me hear the current output.",
]

_SEARCH_REASON_TEMPLATES = [
    "Let me find the {keyword} parameters and see their current values.",
    "I'll search for {keyword} params to know what I'm working with.",
    "Checking the current {keyword} settings before making changes.",
    "Let me look up the {keyword} parameters.",
]

# ── config ────────────────────────────────────────────────────────────────────

# Listen/render: the model calls this standalone script
_LISTEN_CMD    = "python scripts/reaper_render_probe.py && aplay /tmp/probe.wav"
_LISTEN_RESULT = "<audio>\nRendered to /tmp/probe.wav"

ARCHETYPES       = ["bass", "lead", "pad"]
SEEDS            = [42, 7, 123]
OUT_DIR          = Path("outputs/iter_demo")
WT_LIB           = Path("data/wavetable_lib.json")
SAMPLE_RATE      = 44100
SEARCH_STEP_PROB = 0.70   # probability of a search turn per iteration

# ── helpers ───────────────────────────────────────────────────────────────────

def render_preset_to_wav(preset: dict, notes: list, out_path: Path,
                         sample_rate: int = SAMPLE_RATE):
    """Render a preset dict to a WAV file using vita bindings (no REAPER)."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".vital", mode="w", delete=False) as tf:
        json.dump(preset, tf)
        tmp = tf.name
    try:
        audio = render_notes(notes, tmp, sample_rate=sample_rate)
    finally:
        os.unlink(tmp)
    import soundfile as sf
    sf.write(str(out_path), audio.T, sample_rate, subtype="PCM_16")
    return audio


def _placeholder_commentary(it: dict, step_num: int, n_total: int,
                             prev_step: dict | None = None) -> str:
    """Placeholder reasoning: what was heard → what to change next."""
    n_params   = len(it["params_changed"])
    n_mods     = len(it["modulations_changed"])
    is_mistake = it["is_mistake_step"]

    # Summarise which param groups are being changed
    prefix_counts: Counter = Counter()
    for name in it["params_changed"]:
        for pfx in PARAM_PRIORITY_PREFIXES:
            if name.startswith(pfx):
                prefix_counts[pfx.rstrip("_")] += 1
                break
    groups = ", ".join(f"{v} {k}" for k, v in prefix_counts.most_common(4))
    if not groups:
        groups = f"{n_params} params"

    # Observation grounded in what was heard
    if step_num == 1:
        observation = "The default preset is far from the target — the timbre and envelope are both off. "
    elif prev_step is not None and prev_step.get("is_mistake_step"):
        observation = "That step overcorrected — the sound moved away from the target. Pulling back. "
    elif step_num == n_total:
        observation = "Getting close. A few final tweaks to nail the remaining difference. "
    else:
        observation = f"Closer, but still need to adjust {groups}. "

    parts = [f"{observation}Adjusting {groups}"]
    if n_mods:
        parts.append(f" + {n_mods} mod route(s)")
    if is_mistake:
        parts.append(" [moving in wrong direction intentionally to test recovery]")
    parts.append(".")
    return "".join(parts)


def build_conversation(path_result: dict, wav_paths: dict, rng: random.Random) -> dict:
    """
    Build the agentic tool-call conversation.

    wav_paths = {"gt": Path, "default": Path|None, "steps": [Path, ...]}

    Structure:
      user: <audio:gt> + task
      assistant: "Let me hear the default." + render tool_call   ← baseline
      tool: <audio:default>
      per iteration:
        (assistant: search for keyword params, tool: name+value list)  ← ~70% prob
        assistant: observation + set tool_call
        tool: Done
        assistant: listen text + render tool_call
        tool: <audio:step_k>
      assistant: conclusion (no tool_call)
    """
    sample_id  = path_result["sample_id"]
    archetype  = path_result["archetype"]
    n_iter     = path_result["n_iterations"]
    iters      = path_result["iterations"]

    messages: list[dict] = []
    audios: list[str]    = [str(wav_paths["gt"])]

    # ── Turn 0: user / GT audio ──────────────────────────────────────────────
    messages.append({
        "role": "user",
        "content": (
            f"<audio>\n"
            f"This is a target synthesizer sound ({archetype}). "
            f"Recreate it in Vital starting from the default preset using reapy."
        ),
    })

    # ── Turn 1: listen to default preset for baseline ────────────────────────
    listen_default_id = "tc_listen_default"
    default_wav = wav_paths.get("default")
    if default_wav:
        audios.append(str(default_wav))
    messages.append({
        "role": "assistant",
        "content": "Let me first hear the default Vital preset as a baseline before making any changes.",
        "tool_calls": [{
            "id": listen_default_id, "type": "function",
            "function": {"name": "bash", "arguments": json.dumps({"command": _LISTEN_CMD})},
        }],
    })
    messages.append({
        "role": "tool",
        "tool_call_id": listen_default_id,
        "content": _LISTEN_RESULT,
    })

    # ── Per-iteration tool-call triples ───────────────────────────────────────
    for i, it in enumerate(iters):
        step_num       = it["step"]
        n_params       = len(it["params_applied"])
        action_snippet = it["action_snippet"]
        search_snippet = it["search_snippet"]
        search_result  = it["search_result"]
        keyword        = it["search_keyword"]
        prev_step      = iters[i - 1] if i > 0 else None

        # 1. Search turn (always on step 1; probabilistic after that)
        if i == 0 or rng.random() < SEARCH_STEP_PROB:
            tc_id = f"tc_search_{step_num}"
            reason = _SEARCH_REASON_TEMPLATES[i % len(_SEARCH_REASON_TEMPLATES)].format(
                keyword=keyword
            )
            messages.append({
                "role": "assistant",
                "content": reason,
                "tool_calls": [{
                    "id": tc_id, "type": "function",
                    "function": {"name": "bash", "arguments": json.dumps({"command": search_snippet})},
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": search_result or f"[no params matching '{keyword}']",
            })

        # 2. Set turn
        set_id = f"tc_set_{step_num}"
        messages.append({
            "role": "assistant",
            "content": _placeholder_commentary(it, step_num, n_iter, prev_step=prev_step),
            "tool_calls": [{
                "id": set_id, "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": action_snippet})},
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": set_id,
            "content": "Done",
        })

        # 3. Listen turn
        listen_id = f"tc_listen_{step_num}"
        step_wav = wav_paths["steps"][i] if i < len(wav_paths["steps"]) else None
        if step_wav:
            audios.append(str(step_wav))
        messages.append({
            "role": "assistant",
            "content": _LISTEN_TEXTS[i % len(_LISTEN_TEXTS)],
            "tool_calls": [{
                "id": listen_id, "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": _LISTEN_CMD})},
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": listen_id,
            "content": _LISTEN_RESULT,
        })

    # ── Final assistant turn: conclusion (no tool_call) ──────────────────────
    messages.append({
        "role": "assistant",
        "content": "[PLACEHOLDER: Recreation complete.]",
    })

    return {
        "id": sample_id,
        "messages": messages,
        "audios": audios,
        "meta": {
            "archetype": archetype,
            "n_iterations": n_iter,
            "n_changed_params": path_result["n_changed_params"],
            "has_mistake_step": path_result["has_mistake_step"],
            "commentary_source": "placeholder",
        },
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading wavetable lib from {WT_LIB} …")
    wt_lib = load_wavetable_lib(str(WT_LIB))

    summary_lines: list[str] = []
    all_conversations: list[dict] = []

    for archetype, seed in zip(ARCHETYPES, SEEDS):
        print(f"\n{'='*60}")
        print(f"Archetype: {archetype}  seed={seed}")
        print(f"{'='*60}")

        rng      = random.Random(seed)
        arch_out = OUT_DIR / archetype
        arch_out.mkdir(exist_ok=True)

        # Generate path
        path_result = generate_preset_path(
            archetype, rng,
            wavetable_lib=wt_lib,
            output_dir=arch_out,
        )
        sid      = path_result["sample_id"]
        n_iter   = path_result["n_iterations"]
        n_changed = path_result["n_changed_params"]

        print(f"  sample_id       : {sid}")
        print(f"  n_iterations    : {n_iter}")
        print(f"  n_changed_params: {n_changed}")
        print(f"  has_mistake_step: {path_result['has_mistake_step']}")
        for it in path_result["iterations"]:
            snippet_preview = (it.get("action_snippet") or "")[:60].replace("\n", " | ")
            print(f"    step {it['step']}: {len(it['params_changed'])} params, "
                  f"{len(it['modulations_changed'])} mods"
                  + (" [MISTAKE]" if it["is_mistake_step"] else "")
                  + f"  snippet: {snippet_preview}…")

        # Save target preset
        target_path = arch_out / f"{sid}_target.vital"
        with open(target_path, "w") as f:
            json.dump(path_result["target_preset"], f, indent=2)
        print(f"  target preset   → {target_path}")

        # Shared probe notes for GT + default + all iteration clips.
        # Using the same note sequence everywhere so all clips are directly
        # comparable — different note lengths would make release sound different
        # even when the preset is identical.
        from maestro.synth.path_gen import _INIT_PRESET
        probe_notes = make_probe_notes(archetype)

        # Render GT audio (same notes as probe so release is comparable)
        gt_wav = arch_out / f"{sid}_gt.wav"
        print(f"  rendering GT    → {gt_wav}")
        try:
            render_preset_to_wav(path_result["target_preset"], probe_notes, gt_wav)
        except Exception as e:
            print(f"    ERROR rendering GT: {e}")

        # Render default (init) preset audio for baseline listen
        default_wav = arch_out / f"{sid}_default.wav"
        print(f"  rendering default → {default_wav}")
        try:
            render_preset_to_wav(_INIT_PRESET, probe_notes, default_wav)
        except Exception as e:
            print(f"    ERROR rendering default: {e}")
            default_wav = None

        # Render ALL N iteration clips (every step gets a vital listen result)
        step_wavs: list[Path | None] = []
        for it in path_result["iterations"]:
            step_num   = it["step"]
            preset_path = it["cumulative_preset_path"]
            if preset_path and Path(preset_path).exists():
                with open(preset_path) as f:
                    cumul_preset = json.load(f)
            else:
                cumul_preset = it.get("cumulative_preset") or path_result["target_preset"]

            step_wav = arch_out / f"{sid}_step{step_num}.wav"
            print(f"  rendering step {step_num} → {step_wav}")
            try:
                render_preset_to_wav(cumul_preset, probe_notes, step_wav)
                step_wavs.append(step_wav)
            except Exception as e:
                print(f"    ERROR rendering step {step_num}: {e}")
                step_wavs.append(None)

        # Build conversation
        conv = build_conversation(
            path_result,
            {"gt": gt_wav, "default": default_wav, "steps": [w for w in step_wavs if w]},
            rng,
        )
        all_conversations.append(conv)

        conv_path = arch_out / f"{sid}_conversation.json"
        with open(conv_path, "w") as f:
            json.dump(conv, f, indent=2)
        print(f"  conversation    → {conv_path}")

        fidelity = compare_preset_path(path_result)
        print(f"  preset fidelity : {fidelity['summary']}")

        summary_lines.append(
            f"{archetype}/{sid}: {n_iter} iters, {n_changed} changed params, "
            f"mistake={path_result['has_mistake_step']} | fidelity: {fidelity['summary']}"
        )

    # Write combined JSONL
    jsonl_path = OUT_DIR / "demo_train.jsonl"
    with open(jsonl_path, "w") as f:
        for conv in all_conversations:
            f.write(json.dumps(conv) + "\n")
    print(f"\n→ JSONL: {jsonl_path}")

    # Write summary
    summary_path = OUT_DIR / "demo_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Demo Iterative Preset Examples\n" + "="*60 + "\n\n")
        for line in summary_lines:
            f.write(line + "\n")
        f.write(f"\nTotal: {len(all_conversations)} samples\n")
        f.write(f"Output: {OUT_DIR.resolve()}\n")
    print(f"→ Summary: {summary_path}")

    # Print one full conversation to stdout
    if all_conversations:
        print("\n" + "="*60)
        print("SAMPLE CONVERSATION (first example):")
        print("="*60)
        conv = all_conversations[0]
        print(f"id: {conv['id']}")
        print(f"audios ({len(conv['audios'])}):")
        for a in conv["audios"]:
            print(f"  {a}")
        print(f"\nmessages ({len(conv['messages'])}):")
        for m in conv["messages"]:
            role = m["role"]
            content = m.get("content", "")
            tc = m.get("tool_calls")
            print(f"\n[{role.upper()}]", end="")
            if tc:
                cmd = json.loads(tc[0]["function"]["arguments"]).get("command", "")
                print(f" [tool_call: bash] {cmd[:80]}{'…' if len(cmd)>80 else ''}")
            else:
                print()
                print((content or "")[:200] + ("…" if len(content or "") > 200 else ""))


if __name__ == "__main__":
    main()
