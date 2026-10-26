#!/usr/bin/env python3
"""Eval the search agent LoRA against real Vital-rendered wavetable probes.

For each val sample:
  1. Build a 48-wavetable shard containing the GT wavetable(s)
  2. Copy or render probe audio (reuses pre-rendered candidate_probes/ cache)
  3. Run the model in a multi-turn ChatML loop via /v1/completions
  4. Parse the shortlist from the model's final output
  5. Score: Recall@shortlist of GT wavetables

The model was trained with tool_call / tool_response as ChatML roles:
  <|im_start|>tool_call
  {"name": "Bash", "arguments": {...}}
  <|im_end|>
These special tokens render as garbled UTF-8 when skip_special_tokens=False.
We detect tool calls by finding JSON with "name"+"arguments" keys after the
assistant text boundary.

Usage:
    python scripts/eval_search_agent.py \
        --val-jsonl outputs/sft_32k/prepared/val.jsonl \
        --wavetable-lib data/wavetable_lib.json \
        --audio-dir outputs/sft_32k/audio \
        --probe-dir outputs/eval_search/probes \
        --server http://localhost:8000 \
        --model maestro-sft \
        --n 10
"""
from __future__ import annotations

import argparse
import ast
import base64
import json
import os
import re
import shutil
import sys
import time
import warnings
from pathlib import Path

import requests

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERMES_TOOLS_SYSTEM = """You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "Bash", "description": "Execute shell/Python commands to render and evaluate wavetable candidates.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}
{"type": "function", "function": {"name": "Read", "description": "Read a file. For audio files, returns the audio content for listening.", "parameters": {"type": "object", "properties": {"file_path": {"type": "string", "description": "Absolute path to the file."}}, "required": ["file_path"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""


def load_wavetable_lib(path: str) -> tuple[dict[str, int], dict[int, str]]:
    wt_lib = json.load(open(path))
    name_to_idx: dict[str, int] = {}
    idx_to_name: dict[int, str] = {}
    for i, e in enumerate(wt_lib):
        name = e.get("name")
        if not name:
            groups = e.get("groups", [])
            if isinstance(groups, str):
                groups = ast.literal_eval(groups)
            if groups:
                for g in groups:
                    for c in g.get("components", []):
                        name = c.get("name")
                        if name:
                            break
                    if name:
                        break
        if name:
            name_to_idx[name] = i
            idx_to_name[i] = name
    return name_to_idx, idx_to_name


def build_shard_containing_gt(
    gt_names: list[str],
    name_to_idx: dict[str, int],
    idx_to_name: dict[int, str],
    shard_size: int = 48,
) -> tuple[int, int, list[str]]:
    gt_indices = sorted(name_to_idx[n] for n in gt_names if n in name_to_idx)
    if not gt_indices:
        return 0, shard_size, [idx_to_name[i] for i in range(shard_size) if i in idx_to_name]
    center = gt_indices[len(gt_indices) // 2]
    max_idx = max(idx_to_name.keys())
    start = max(0, center - shard_size // 2)
    end = min(max_idx + 1, start + shard_size)
    if end - start < shard_size:
        start = max(0, end - shard_size)
    shard_names = [idx_to_name[i] for i in range(start, end) if i in idx_to_name]
    return start, end, shard_names


def _slugify(s: str) -> str:
    return (re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_") or "unnamed")[:80]


def _match_probe_name(
    file_path: str,
    name_to_path: dict[str, "Path | None"],
    name_to_idx: dict[str, int],
) -> str | None:
    for name in name_to_path:
        if name_to_path[name] and _slugify(name) in file_path:
            return name
        idx = name_to_idx.get(name)
        if idx is not None and f"wt_{idx:04d}" in file_path:
            return name
    return None


def ensure_probes(
    shard_names: list[str],
    name_to_idx: dict[str, int],
    probe_dir: Path,
    existing_probe_dir: Path,
) -> dict[str, Path | None]:
    probe_dir.mkdir(parents=True, exist_ok=True)
    name_to_path: dict[str, Path | None] = {}
    for name in shard_names:
        idx = name_to_idx.get(name)
        if idx is None:
            continue
        slug = _slugify(name)
        out_path = probe_dir / f"wt_{idx:04d}_{slug}.wav"
        if out_path.exists():
            name_to_path[name] = out_path
            continue
        existing = list(existing_probe_dir.glob(f"{slug}__src{idx:04d}_f*.wav"))
        if not existing:
            existing = list(existing_probe_dir.glob(f"*__src{idx:04d}_f*.wav"))
        if existing:
            shutil.copy2(existing[0], out_path)
            name_to_path[name] = out_path
            continue
        name_to_path[name] = None
    return name_to_path


def render_missing_probes(
    name_to_path: dict[str, Path | None],
    name_to_idx: dict[str, int],
    wt_lib_path: str,
    probe_dir: Path,
) -> dict[str, Path | None]:
    missing = {n: name_to_idx[n] for n in name_to_path
               if name_to_path[n] is None and n in name_to_idx}
    if not missing:
        return name_to_path
    print(f"  Rendering {len(missing)} missing probes via vita...")
    try:
        from maestro.render.vital import make_probe_notes
        from maestro.render.dawdreamer import render_preset_audio
        from agent_sft_common import _load_init_preset, _build_probe_preset
    except ImportError as e:
        print(f"  WARNING: Cannot render probes ({e}). Skipping.")
        return name_to_path
    wt_lib = json.load(open(wt_lib_path))
    init_preset = _load_init_preset()
    notes = make_probe_notes("lead")
    for name, idx in missing.items():
        slug = _slugify(name)
        out_path = probe_dir / f"wt_{idx:04d}_{slug}.wav"
        try:
            wt = wt_lib[idx]
            preset = _build_probe_preset(init_preset, wt, frame_idx=0)
            render_preset_audio(preset, notes, out_path=out_path, tail_s=1.0)
            name_to_path[name] = out_path
        except Exception as e:
            print(f"    Failed '{name}': {e}")
    return name_to_path


def _b64_audio(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ---------------------------------------------------------------------------
# ChatML prompt building and multi-turn loop
# ---------------------------------------------------------------------------

def _extract_tool_calls(text: str) -> list[dict]:
    """Extract tool call JSON objects from model output.

    The model emits <|im_end|><|im_start|>tool_call\\n{...}<|im_end|>
    but these special tokens render as garbled UTF-8 bytes.
    We find JSON objects starting with {"name" and use brace-depth matching.
    """
    results = []
    search_from = 0
    while True:
        idx = text.find('{"name"', search_from)
        if idx < 0:
            break
        depth = 0
        end = -1
        for ci in range(idx, len(text)):
            if text[ci] == '{':
                depth += 1
            elif text[ci] == '}':
                depth -= 1
            if depth == 0:
                end = ci + 1
                break
        if end < 0:
            search_from = idx + 1
            continue
        json_str = text[idx:end]
        try:
            obj = json.loads(json_str)
            if "name" in obj and "arguments" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            pass
        search_from = end
    return results


def _extract_assistant_text(text: str) -> str:
    """Extract the assistant's natural language text from model output.

    Handles both primed outputs (just text) and unprimed outputs
    (garbled <|im_start|> bytes + 'assistant\\n' + text).
    """
    m = re.search(r'\{"name"\s*:', text)
    if m:
        prefix = text[:m.start()]
        cleaned = re.sub(r'[^\x20-\x7e\n]', '', prefix).strip()
    else:
        cleaned = re.sub(r'[^\x20-\x7e\n]', '', text).strip()
    for role_prefix in ('assistant\n', 'assistant\r\n'):
        if cleaned.startswith(role_prefix):
            cleaned = cleaned[len(role_prefix):].strip()
            break
    return cleaned


def run_search_loop(
    server: str,
    model: str,
    target_audio_path: Path,
    shard_start: int,
    shard_end: int,
    shard_names: list[str],
    name_to_idx: dict[str, int],
    idx_to_name: dict[int, str],
    name_to_path: dict[str, Path | None],
    probe_dir: str,
    max_turns: int = 20,
    temperature: float = 0.1,
    verbose: bool = False,
) -> tuple[list[str], list[dict]]:
    """Run multi-turn search via /v1/completions with raw ChatML."""

    target_b64 = _b64_audio(target_audio_path)

    # Build initial prompt
    chatml = (
        f"<|im_start|>system\n{HERMES_TOOLS_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"<|audio_bos|><|AUDIO|><|audio_eos|>\n"
        f"Transcription MIDI: /tmp/eval_notes.json.\n"
        f"Evaluate wavetables at indices {shard_start}-{shard_end - 1}. "
        f"Scan Vital's data directories for .vitaltable and .vital files to get names in your range, "
        f"render all candidates at once, then listen to each batch and assess. "
        f"Return a JSON shortlist of 2-4 wavetable names.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    audio_data = [f"data:audio/wav;base64,{target_b64}"]

    list_response_json = json.dumps({
        "wavetables": [
            {"idx": i, "name": idx_to_name[i]}
            for i in range(shard_start, shard_end)
            if i in idx_to_name
        ],
        "start": shard_start, "end": shard_end,
        "count": sum(1 for i in range(shard_start, shard_end) if i in idx_to_name),
    })

    render_entries = []
    for name in shard_names:
        idx = name_to_idx.get(name)
        if idx is None:
            continue
        render_entries.append({"idx": idx, "name": name, "out": f"{probe_dir}/wt_{idx:04d}_{_slugify(name)}.wav"})
    render_response_json = json.dumps({"status": "ok", "rendered": render_entries})

    conversation_log = []
    all_model_text = ""
    bash_call_count = 0
    is_primed = True  # initial prompt ends with <|im_start|>assistant\n
    consecutive_text_turns = 0
    read_count = 0
    listened_probes = set()  # track which probes have audio in context

    for turn in range(max_turns):
        if verbose:
            prompt_len = len(chatml)
            audio_mb = sum(len(a) for a in audio_data) / 1e6
            print(f"      [pre-req] prompt={prompt_len} chars, audio={len(audio_data)} clips ({audio_mb:.1f} MB b64)", flush=True)
        try:
            t_req = time.time()
            body = {
                "model": model,
                "prompt": chatml,
                "max_tokens": 8192,
                "temperature": temperature,
                "multi_modal_data": {"audio": audio_data},
            }
            if verbose:
                print(f"      [pre-req] body built in {time.time()-t_req:.1f}s, posting...", flush=True)
            resp = requests.post(
                f"{server}/v1/completions",
                json=body,
                timeout=600,
            )
            if verbose:
                print(f"      [post-req] response in {time.time()-t_req:.1f}s", flush=True)
            resp.raise_for_status()
        except Exception as e:
            print(f"    API error turn {turn}: {e}")
            break

        t_parse = time.time()
        resp_json = json.loads(resp.content)
        output = resp_json["choices"][0]["text"]
        finish_reason = resp_json["choices"][0].get("finish_reason", "unknown")
        all_model_text += output + "\n"

        tool_calls = _extract_tool_calls(output)
        assistant_text = _extract_assistant_text(output)
        if verbose:
            print(f"      [parse] {time.time()-t_parse:.1f}s, output={len(output)} chars", flush=True)

        if verbose:
            role = "tool_call" if tool_calls else "assistant"
            tc_names = [tc["name"] for tc in tool_calls]
            print(f"    turn {turn} [{role}]: {len(assistant_text)} chars text, {len(tool_calls)} tc ({tc_names[:5]}), finish={finish_reason}, primed={is_primed}")
            if assistant_text and not tool_calls:
                print(f"      text: {assistant_text[:200]}...")

        conversation_log.append({
            "turn": turn,
            "assistant_text": assistant_text[:200] if not tool_calls else "",
            "n_tool_calls": len(tool_calls),
            "tool_names": [tc["name"] for tc in tool_calls],
        })

        if tool_calls:
            # Only reset stall counter if this turn did something new
            has_new_audio = False

            # Close any primed assistant turn
            if is_primed:
                if assistant_text:
                    chatml += f"{assistant_text}<|im_end|>\n"
                else:
                    chatml += "<|im_end|>\n"

            # Process tool calls — cap at 48 (shard size) to avoid wasting context on repeats
            t_tc = time.time()
            for tc in tool_calls[:48]:
                tool_name = tc["name"]
                tool_args = tc.get("arguments", {})
                tc_json = json.dumps(tc, ensure_ascii=False)

                chatml += f"<|im_start|>tool_call\n{tc_json}<|im_end|>\n"

                if tool_name == "Bash":
                    bash_call_count += 1
                    if bash_call_count == 1:
                        tool_resp = f'{{"stdout": {json.dumps(list_response_json + chr(10))}, "stderr": "", "interrupted": false}}'
                    elif bash_call_count == 2:
                        tool_resp = f'{{"stdout": {json.dumps(render_response_json + chr(10))}, "stderr": "", "interrupted": false}}'
                    else:
                        tool_resp = '{"stdout": "ok\\n", "stderr": "", "interrupted": false}'
                    chatml += f"<|im_start|>tool_response\n{tool_resp}<|im_end|>\n"

                elif tool_name == "Read":
                    read_count += 1
                    file_path = tool_args.get("file_path", "")
                    matched_name = _match_probe_name(file_path, name_to_path, name_to_idx)

                    if matched_name and matched_name in listened_probes:
                        chatml += f"<|im_start|>tool_response\n[Already listened to this probe above]<|im_end|>\n"
                    elif matched_name and name_to_path.get(matched_name):
                        has_new_audio = True
                        listened_probes.add(matched_name)
                        probe_b64 = _b64_audio(name_to_path[matched_name])
                        audio_data.append(f"data:audio/wav;base64,{probe_b64}")
                        chatml += (
                            f"<|im_start|>tool_response\n"
                            f"<|audio_bos|><|AUDIO|><|audio_eos|><|im_end|>\n"
                        )
                    else:
                        chatml += f"<|im_start|>tool_response\nFile not found: {file_path}<|im_end|>\n"
                else:
                    chatml += f"<|im_start|>tool_response\nUnknown tool: {tool_name}<|im_end|>\n"

            if verbose:
                print(f"      [tc-process] {time.time()-t_tc:.1f}s for {min(len(tool_calls),48)} calls", flush=True)

            if has_new_audio or any(tc["name"] == "Bash" for tc in tool_calls[:48]):
                consecutive_text_turns = 0
            else:
                consecutive_text_turns += 1

            # After processing all tool calls, always prime as assistant
            chatml += "<|im_start|>assistant\n"
            is_primed = True

            if verbose and read_count > 0:
                print(f"      reads so far: {read_count}, unique probes: {len(listened_probes)}, audio clips: {len(audio_data)}")

        else:
            consecutive_text_turns += 1

            if is_primed:
                chatml += f"{assistant_text}<|im_end|>\n"
            else:
                chatml += f"<|im_start|>assistant\n{assistant_text}<|im_end|>\n"
            is_primed = False

            shortlist_so_far = parse_shortlist(assistant_text)
            if shortlist_so_far:
                if verbose:
                    print(f"    → shortlist found: {shortlist_so_far}")
                break

            if consecutive_text_turns >= 3:
                if verbose:
                    print(f"    → {consecutive_text_turns} consecutive text turns, stopping")
                break

    shortlist = list(dict.fromkeys(parse_shortlist(all_model_text)))
    return shortlist, conversation_log, all_model_text


def parse_shortlist(text: str) -> list[str]:
    selected = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if "Selected" in line and "Not selected" not in line:
            m = re.match(r"['\"](.+?)['\"]", line)
            if m:
                selected.append(m.group(1))
    if not selected:
        m = re.search(r"Shortlist:\s*\[([^\]]*)\]", text)
        if m:
            selected = re.findall(r"['\"](.+?)['\"]", m.group(1))
    return selected


def main():
    parser = argparse.ArgumentParser(description="Eval search agent on val set")
    parser.add_argument("--val-jsonl", default="outputs/sft_32k/prepared/val.jsonl")
    parser.add_argument("--wavetable-lib", default="data/wavetable_lib.json")
    parser.add_argument("--audio-dir", default="outputs/sft_32k/audio")
    parser.add_argument("--probe-dir", default="outputs/eval_search/probes")
    parser.add_argument("--existing-probe-dir", default="outputs/agent_sft/candidate_probes")
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--model", default="maestro-sft")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--out", default="outputs/eval_search/results.jsonl")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--shard-size", type=int, default=48)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    name_to_idx, idx_to_name = load_wavetable_lib(args.wavetable_lib)
    print(f"Wavetable library: {len(name_to_idx)} named entries")

    val_main_gt = {}
    with open(args.val_jsonl) as f:
        for line in f:
            d = json.loads(line)
            labels = d.get("labels", {})
            if isinstance(labels, str):
                labels = json.loads(labels)
            if d["task_type"] == "main":
                val_main_gt[d["id"]] = labels.get("gt_wavetable_names", [])

    base_ids = [bid for bid, gt in val_main_gt.items() if gt]
    if args.n > 0:
        base_ids = base_ids[: args.n]
    print(f"Evaluating {len(base_ids)} samples\n")

    probe_dir = Path(args.probe_dir)
    existing_probe_dir = Path(args.existing_probe_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    results = []
    total_gt = 0
    total_recalled = 0

    for i, base_id in enumerate(base_ids):
        gt_names = val_main_gt[base_id]
        target_audio = Path(args.audio_dir) / f"{base_id}_gt.wav"
        if not target_audio.exists():
            print(f"[{i+1}/{len(base_ids)}] {base_id}: target audio missing, skip")
            continue

        shard_start, shard_end, shard_names = build_shard_containing_gt(
            gt_names, name_to_idx, idx_to_name, args.shard_size,
        )
        gt_in_shard = [n for n in gt_names if n in shard_names]
        print(f"[{i+1}/{len(base_ids)}] {base_id}: GT={gt_names}, shard=[{shard_start},{shard_end}), GT_in_shard={gt_in_shard}")

        sample_probe_dir = probe_dir / base_id
        name_to_path = ensure_probes(shard_names, name_to_idx, sample_probe_dir, existing_probe_dir)
        missing_count = sum(1 for v in name_to_path.values() if v is None)
        if missing_count:
            print(f"  {missing_count} probes missing (no rendering — using available only)")

        available = sum(1 for v in name_to_path.values() if v is not None)
        print(f"  Probes: {available}/{len(shard_names)} available")

        t0 = time.time()
        shortlist, log, model_text = run_search_loop(
            server=args.server,
            model=args.model,
            target_audio_path=target_audio,
            shard_start=shard_start,
            shard_end=shard_end,
            shard_names=shard_names,
            name_to_idx=name_to_idx,
            idx_to_name=idx_to_name,
            name_to_path=name_to_path,
            probe_dir=str(sample_probe_dir),
            temperature=args.temperature,
            verbose=args.verbose,
        )
        elapsed = time.time() - t0

        recalled = [n for n in gt_in_shard if n in shortlist]
        recall = len(recalled) / len(gt_in_shard) if gt_in_shard else 0.0
        total_gt += len(gt_in_shard)
        total_recalled += len(recalled)

        result = {
            "base_id": base_id,
            "gt_names": gt_names,
            "gt_in_shard": gt_in_shard,
            "shard": [shard_start, shard_end],
            "shortlist": shortlist,
            "recalled": recalled,
            "recall": recall,
            "n_turns": len(log),
            "elapsed_s": round(elapsed, 1),
            "model_text": model_text,
        }
        results.append(result)

        status = "HIT" if recall > 0 else "MISS"
        print(f"  {status}: shortlist={shortlist}, recalled={recalled}, recall={recall:.0%}, {elapsed:.1f}s, {len(log)} turns\n")

        with open(out_path, "a") as f:
            f.write(json.dumps(result) + "\n")

    print("=" * 60)
    if results:
        overall_recall = total_recalled / total_gt if total_gt > 0 else 0.0
        hit_count = sum(1 for r in results if r["recall"] > 0)
        hit_rate = hit_count / len(results)
        avg_shortlist = sum(len(r["shortlist"]) for r in results) / len(results)
        avg_turns = sum(r["n_turns"] for r in results) / len(results)
        print(f"Overall recall: {total_recalled}/{total_gt} = {overall_recall:.1%}")
        print(f"Hit rate: {hit_count}/{len(results)} = {hit_rate:.1%}")
        print(f"Avg shortlist size: {avg_shortlist:.1f}")
        print(f"Avg turns: {avg_turns:.0f}")
    else:
        print("No results.")


if __name__ == "__main__":
    main()
