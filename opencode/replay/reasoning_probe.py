"""Capture what OpenCode 1.19.3 sends back for earlier assistant turns that carried reasoning.

The replay mock streams scripted turns with reasoning (as vLLM's reasoning parser emits it: delta
`reasoning_content`, or delta `reasoning`; or as a `<think>` block in content) through the real binary,
for the main agent and one `task` subagent (transcriber):

  main        0 reasoning + text + tool call   1 reasoning + tool call   2 reasoning + 2 parallel calls
              3 reasoning + task(transcriber)  4 reasoning + final text
  transcriber 0 reasoning + tool call   1 reasoning + 2 parallel calls   2 reasoning + final text

Every request is logged; extract() reduces each to, per earlier assistant turn, the message keys, where
the probe's reasoning text shows up (which field / inside content) and the content itself. Results go to
docs/opencode_capture/goldens/reasoning/ (see its README).

    .venv/bin/python -m opencode.replay.reasoning_probe [--out outputs/replay/reasoning]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from . import golden as G
from . import harness as H
from . import records as R

MODES = ("reasoning_content", "reasoning", "think_tag")
GOLDEN_DIR = G.GOLDENS / "reasoning"
WORKDIR = "/work/rollouts/probe"
PROMPT = "Recreate the sound in target.wav."
TASK_PROMPT = "Target: /work/rollouts/probe/target.wav. Write the melody to /work/rollouts/probe/notes.json."
MARK = "REASONING"  # every probe reasoning string starts with this


def _call(name: str, args: dict[str, Any]) -> dict[str, str]:
    return {"role": "tool_call", "content": json.dumps({"name": name, "arguments": args})}


def _bash(cmd: str, desc: str) -> dict[str, str]:
    return _call("bash", {"command": cmd, "description": desc})


def _resp() -> dict[str, str]:
    return {"role": "tool_response", "content": "(probe: not compared)"}


def _system(agent: str, port: int) -> str:
    text = (G.GOLDENS / "with_plugin" / "system" / f"{agent}.txt").read_text()
    return G._fill(text, {"{WORKDIR}": WORKDIR, "{CONFIG_DIR}": "/tmp/maestro-oc/config",
                          "{DATA_DIR}": "/tmp/maestro-oc/data/opencode", "{PORT}": str(port)})


def probe_records(port: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """(main, transcriber) records; tool responses are placeholders (the probe only reads requests)."""
    main = {"id": "probe-main", "audios": [], "messages": [
        {"role": "system", "content": _system("maestro", port)}, {"role": "user", "content": PROMPT},
        {"role": "assistant", "content": "Looking at the working directory first."},
        _bash("ls", "List files"), _resp(),
        _bash("echo one", "Print one"), _resp(),
        _bash("echo two", "Print two"), _call("read", {"filePath": f"{WORKDIR}/notes.txt"}), _resp(), _resp(),
        _call("task", {"description": "Transcribe target", "prompt": TASK_PROMPT, "subagent_type": "transcriber"}),
        _resp(),
        {"role": "assistant", "content": "Done."}]}
    sub = {"id": "probe-transcriber", "audios": [], "messages": [
        {"role": "system", "content": _system("transcriber", port)}, {"role": "user", "content": TASK_PROMPT},
        _bash("echo sub", "Print sub"), _resp(),
        _bash("echo s1", "Print s1"), _bash("echo s2", "Print s2"), _resp(), _resp(),
        {"role": "assistant", "content": "/work/rollouts/probe/notes.json: 0 notes."}]}
    return main, sub


def reasoning_text(agent: str, step: int) -> str:
    """Multi-line, with the leading/trailing newlines vLLM's qwen3 parser typically leaves in place."""
    return f"\n{MARK} {agent} step {step}: weighing the next move.\nSecond line.\n"


def add_reasoning(scenario: dict[str, Any]) -> None:
    for key, rec in scenario["records"].items():
        agent = "main" if key == "main" else "sub"
        for i, st in enumerate(rec["steps"]):
            st["reasoning"] = reasoning_text(agent, i)


def extract(log: Path) -> list[dict[str, Any]]:
    """Per request: route, step, and for every assistant message its keys, the fields holding probe
    reasoning, whether content holds a <think> block or the reasoning text, and the content."""
    out = []
    for line in log.open():
        e = json.loads(line)
        if e["kind"] == "title":
            continue
        turns = []
        agent = "main" if e["route"] == "main" else "sub"
        for j, m in enumerate(x for x in e["body"]["messages"] if x["role"] == "assistant"):
            c = m.get("content")
            ctext = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
            turns.append({
                "keys": sorted(m),
                "reasoning_fields": sorted(k for k, v in m.items() if k != "content" and MARK in json.dumps(v)),
                "content_has_think_tag": "<think>" in (ctext or ""),
                "content_has_reasoning": MARK in (ctext or ""),
                "reasoning_content_verbatim": m.get("reasoning_content") == reasoning_text(agent, j),
                "content": c,
                "tool_calls": [tc["function"]["name"] for tc in m.get("tool_calls") or []],
            })
        tool_results = [m["content"] for m in e["body"]["messages"] if m["role"] == "tool"]
        out.append({"route": e["route"], "step": e["step"], "assistant_turns": turns,
                    "task_results": [t for t in tool_results if isinstance(t, str) and t.startswith("<task ")],
                    "body_keys": sorted(e["body"])})
    return out


def run(out: Path, port: int = 8911) -> dict[str, Any]:
    init = out / "init"
    shutil.rmtree(init, ignore_errors=True)
    G.make_workdir(init)
    main, sub = probe_records(port)
    results = {}
    for mode in MODES:
        def patch(sc: dict[str, Any], mode: str = mode) -> None:
            add_reasoning(sc)
            sc["reasoning_mode"] = mode
        rep = H.replay(main, [sub], out / mode, init_dir=init, prompt=PROMPT, port=port, scenario_patch=patch)
        cov = {k: (c["requested"], c["steps"]) for k, c in rep["coverage"].items()}
        results[mode] = {"opencode_exit": rep["opencode_exit"], "coverage": cov,
                         "requests": extract(out / mode / "io" / "requests.jsonl")}
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(H.REPO / "outputs/replay/reasoning"))
    ap.add_argument("--port", type=int, default=8911)
    ap.add_argument("--write-goldens", action="store_true", help=f"write {GOLDEN_DIR}/<mode>.json")
    a = ap.parse_args()
    res = run(Path(a.out).resolve(), a.port)
    for mode, r in res.items():
        print(f"== {mode}: exit {r['opencode_exit']} coverage {r['coverage']}")
        for q in r["requests"]:
            print(f"  {q['route']} step {q['step']}: " + "; ".join(
                f"[{i}] fields={t['reasoning_fields']} verbatim={t['reasoning_content_verbatim']} "
                f"think={t['content_has_think_tag']} "
                f"in_content={t['content_has_reasoning']} keys={t['keys']}"
                for i, t in enumerate(q["assistant_turns"])))
    if a.write_goldens:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        for mode, r in res.items():
            (GOLDEN_DIR / f"{mode}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
