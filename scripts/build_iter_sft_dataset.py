#!/usr/bin/env python3
"""Assemble multi-turn ms-swift JSONL from a render manifest + Omni commentary."""

import argparse
import asyncio
import base64
import json
import os
import sys

import httpx

# Rough human-readable label for common param name prefixes
_PARAM_LABELS = {
    "osc_": "oscillator",
    "env_1_": "amplitude envelope",
    "env_2_": "modulation envelope",
    "filter_1_": "filter",
    "filter_2_": "secondary filter",
    "unison_": "unison",
    "lfo_": "LFO",
    "reverb_": "reverb",
    "delay_": "delay",
    "chorus_": "chorus",
    "distortion_": "distortion",
}

_SYNTHETIC_PROJECT_STATE = json.dumps({
    "tracks": [
        {"idx": 0, "name": "Reference"},
        {"idx": 1, "name": "Vital Synth", "fx": "Vital"},
    ]
})


def _param_label(name: str) -> str:
    for prefix, label in _PARAM_LABELS.items():
        if name.startswith(prefix):
            return label
    return "synth parameter"


def _format_delta_context(params_delta: list, is_mistake_step: bool) -> str:
    """Format params_delta list into a string for the Omni prompt."""
    if not params_delta:
        return ""

    lines = []
    mistake_params = []
    for d in params_delta:
        name = d["name"]
        from_n = d["from_norm"]
        to_n = d["to_norm"]
        label = _param_label(name)
        magnitude = abs(to_n - from_n)
        if magnitude < 0.01:
            direction = "unchanged"
        elif to_n > from_n:
            direction = "increased"
        else:
            direction = "decreased"
        mag_str = "slightly" if magnitude < 0.15 else ("significantly" if magnitude > 0.35 else "moderately")

        if d.get("mistake"):
            lines.append(f"- {name} ({label}): {direction} {mag_str} [{from_n:.2f}\u2192{to_n:.2f}] \u26a0 moved away from target")
            mistake_params.append(name)
        else:
            lines.append(f"- {name} ({label}): {direction} {mag_str} [{from_n:.2f}\u2192{to_n:.2f}]")

    context = "Parameters being changed in this step:\n" + "\n".join(lines)

    if mistake_params:
        context += (
            f"\n\nNote: {', '.join(mistake_params)} moved in the wrong direction (overcorrection). "
            "The description should acknowledge this mistake and what needs to be fixed next."
        )

    return context


def _build_param_summary(it: dict | None) -> str:
    """Build a short human-readable param-change summary from an iteration dict.

    Uses ``params_delta`` (list of dicts with 'name'/'from_norm'/'to_norm') when
    available, falling back to the simpler ``params_changed`` list.  Returns an
    empty string when no useful data is present.
    """
    if not it:
        return ""

    # Prefer the richer list form: [{"name": ..., "from_norm": ..., "to_norm": ...}]
    delta_list = it.get("params_delta", [])
    if delta_list:
        top = sorted(delta_list, key=lambda d: abs(d.get("to_norm", 0) - d.get("from_norm", 0)), reverse=True)[:6]
        parts = []
        for d in top:
            name = d["name"]
            diff = d.get("to_norm", 0) - d.get("from_norm", 0)
            arrow = "↑" if diff > 0 else "↓"
            parts.append(f"{name.replace('_', ' ')} {arrow} {abs(diff):.2f}")
        return "; ".join(parts)

    # Fallback: simple dict form {name: delta}
    delta_dict = it.get("params_delta_dict", {})
    if delta_dict:
        top = sorted(delta_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:6]
        return "; ".join(
            f"{name.replace('_', ' ')} {'↑' if d > 0 else '↓'} {abs(d):.2f}"
            for name, d in top
        )

    # Last resort: just list the changed param names
    changed = it.get("params_changed", [])
    if changed:
        return "params changed: " + ", ".join(changed[:6])

    return ""


async def get_commentary(
    gt_wav: str,
    iter_wav: str | None,
    step_num: int,
    client: httpx.AsyncClient,
    server: str,
    model: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    params_delta: list | None = None,
    is_mistake_step: bool = False,
    it: dict | None = None,
) -> str:
    delta_context = _format_delta_context(params_delta or [], is_mistake_step)

    # Build grounded description prompt from the iteration dict when available
    param_summary = _build_param_summary(it)
    if param_summary:
        description_prompt = (
            f"Describe this synthesizer audio clip in 2-3 sentences. "
            f"Focus on timbre, texture, and movement. "
            f"Context: the following parameters were just adjusted — {param_summary}. "
            f"Let that context inspire the description without being too literal."
        )
    else:
        description_prompt = (
            "Describe this synthesizer audio clip in 2-3 sentences. "
            "Focus on timbre, texture, and movement."
        )

    with open(gt_wav, "rb") as f:
        gt_b64 = base64.b64encode(f.read()).decode()

    content = [
        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{gt_b64}"}}
    ]

    if iter_wav:
        with open(iter_wav, "rb") as f:
            iter_b64 = base64.b64encode(f.read()).decode()
        content.append(
            {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{iter_b64}"}}
        )
        if is_mistake_step:
            text = (
                "You are a music production AI agent. You heard the target (earlier in conversation) "
                "and now hear your latest iteration (above). Describe what's improved and what still "
                "needs work. Explain your next parameter choices.\n\n"
                + delta_context + "\n\n"
                "Note: this step includes some overcorrections that moved certain parameters too far. "
                "Acknowledge the overcorrection and what it caused perceptually."
            )
        else:
            text = (
                "You are a music production AI agent. You heard the target (earlier in conversation) "
                "and now hear your latest iteration (above). " + description_prompt + "\n\n"
                "Describe what's improved and what still needs work. "
                "Explain your next parameter choices.\n\n"
                + delta_context
            )
    else:
        text = (
            "You are a music production AI agent. You just heard the target sound (above). "
            + description_prompt + "\n\n"
            "Based on what you hear and the parameter changes you're about to make, explain your "
            "reasoning. Be specific about which sonic characteristics you're targeting.\n\n"
            + delta_context
        )

    content.append({"type": "text", "text": text})

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 150,
        "temperature": 0.7,
    }
    resp = await client.post(
        f"{server}/v1/chat/completions", json=payload, timeout=60.0
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


async def process_sample(
    manifest_entry: dict,
    omni_server: str,
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
    model: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
) -> dict | None:
    sample_id = manifest_entry["sample_id"]
    with open(manifest_entry["path_file"]) as f:
        path_data = json.load(f)

    gt_wav = manifest_entry["gt_wav"]
    iter_wavs = manifest_entry["iter_wavs"]  # N entries (all steps rendered)
    n_iterations = manifest_entry["n_iterations"]
    archetype = manifest_entry.get("archetype", "synth")

    messages = []
    audios = [os.path.abspath(gt_wav)]

    # Turn 0: user / GT audio + task
    messages.append({
        "role": "user",
        "content": (
            f"<audio>\n"
            f"This is a target synthesizer sound ({archetype}). "
            f"Recreate it in Vital starting from the default preset. "
            f"Use the terminal: run `vital set key=val ...` to change parameters "
            f"and `vital listen` to hear the current state."
        ),
    })

    for step_idx in range(n_iterations):
        step_data = path_data["iterations"][step_idx]
        action_snippet = (
            step_data.get("action_snippet")
            or step_data.get("python_script")  # backwards compat
            or "vital set  # no action"
        )
        params_delta = step_data.get("params_delta", [])
        is_mistake = step_data.get("is_mistake_step", False)
        params_applied = step_data.get("params_applied", {})
        modulations_changed = step_data.get("modulations_changed", [])
        n_params = len(params_applied)
        n_mods = len(modulations_changed)
        step_num = step_data["step"]

        # iter_wav for Omni's commentary: use previous step's render (None for step 1)
        iter_wav_for_commentary = iter_wavs[step_idx - 1] if step_idx > 0 else None

        # Get Omni commentary (will be used as the text before the vital set tool_call)
        async with sem:
            try:
                commentary = await get_commentary(
                    gt_wav, iter_wav_for_commentary, step_num, client, omni_server,
                    model=model,
                    params_delta=params_delta,
                    is_mistake_step=is_mistake,
                    it=step_data,
                )
            except Exception as e:
                print(f"  WARNING: commentary failed for {sample_id} step {step_num}: {e}",
                      file=sys.stderr)
                commentary = f"[Commentary unavailable: {e}]"

        # vital set tool_call
        set_id = f"tc_set_{step_num}"
        set_result = f"OK: set {n_params} param(s)"
        if n_mods:
            set_result += f", {n_mods} mod route(s)"
        messages.append({
            "role": "assistant",
            "content": commentary,
            "tool_calls": [{
                "id": set_id, "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({"command": action_snippet}),
                },
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": set_id,
            "content": set_result,
        })

        # vital listen tool_call — model explicitly requests to hear result
        listen_id = f"tc_listen_{step_num}"
        iter_wav = iter_wavs[step_idx]
        audios.append(os.path.abspath(iter_wav))
        messages.append({
            "role": "assistant",
            "content": "Let me listen to the current state.",
            "tool_calls": [{
                "id": listen_id, "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({"command": "vital listen"}),
                },
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": listen_id,
            "content": f"<audio>\nRendered to /tmp/iter_{step_num}.wav",
        })

    # Final assistant turn: conclusion (no tool_call)
    messages.append({
        "role": "assistant",
        "content": "Recreation complete.",
    })

    return {"id": sample_id, "messages": messages, "audios": audios}


def validate_record(record: dict) -> None:
    n_audio_tags = sum(
        (m.get("content") or "").count("<audio>") for m in record["messages"]
    )
    assert n_audio_tags == len(record["audios"]), (
        f"{record['id']}: audio tag count ({n_audio_tags}) != audios len ({len(record['audios'])})"
    )
    assert record["messages"][-1]["role"] == "assistant", (
        f"{record['id']}: last message is not assistant"
    )
    assert "tool_calls" not in record["messages"][-1], (
        f"{record['id']}: final assistant message must not have tool_calls"
    )
    assert record["messages"][0]["role"] == "user", (
        f"{record['id']}: first message is not user"
    )


async def main():
    parser = argparse.ArgumentParser(description="Build iterative SFT dataset from render manifest")
    parser.add_argument("--manifest", required=True, help="Path to manifest.jsonl")
    parser.add_argument("--omni-server", default="http://localhost:8000", help="Omni model server URL")
    parser.add_argument("--omni-model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct", help="Model name to pass to the inference API")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent Omni API calls")
    args = parser.parse_args()

    # Load manifest
    with open(args.manifest) as f:
        entries = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(entries)} samples from manifest")

    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient() as client:
        tasks = [
            process_sample(entry, args.omni_server, sem, client, model=args.omni_model)
            for entry in entries
        ]
        records = []
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            try:
                record = await coro
                if record is not None:
                    records.append(record)
            except Exception as e:
                print(f"\n  ERROR processing sample: {e}", file=sys.stderr)
            print(f"  Processed {i + 1}/{len(entries)}", end="\r")
    print()

    # Validate all records
    for record in records:
        validate_record(record)
    print(f"All {len(records)} records valid")

    # Sort by id for deterministic output
    records.sort(key=lambda r: r["id"])

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
