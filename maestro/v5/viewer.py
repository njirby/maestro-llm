"""One rollout (main record + its linked subagent records) as a static HTML page with inline audio players.

  python -m maestro.v5.viewer outputs/v5_rollouts/<run> <sample_id> [-o page.html]

Audio sources are paths relative to the page, so the run directory can be served or copied as a whole.
"""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

CSS = """
:root { --bg:#fff; --fg:#1d1d1f; --mut:#6b6b70; --line:#e3e3e6; --sys:#f4f4f6; --ast:#eef4ff; --call:#fff7e6;
        --resp:#f6f6f6; --user:#effaf1; --think:#f5f0ff; }
@media (prefers-color-scheme: dark) { :root { --bg:#151517; --fg:#e8e8ea; --mut:#9a9aa0; --line:#2c2c30;
        --sys:#1d1d20; --ast:#18233a; --call:#2d2614; --resp:#1c1c1f; --user:#15281a; --think:#231a33; } }
body { background:var(--bg); color:var(--fg); font:14px/1.45 system-ui, sans-serif; margin:0 auto; max-width:1000px;
       padding:16px; }
h1 { font-size:20px; } h2 { font-size:17px; margin-top:32px; border-top:1px solid var(--line); padding-top:16px; }
.m { border:1px solid var(--line); border-radius:8px; margin:8px 0; padding:8px 10px; overflow-x:auto; }
.m .r { color:var(--mut); font-size:12px; text-transform:uppercase; letter-spacing:.04em; margin-bottom:4px; }
.system { background:var(--sys); } .assistant { background:var(--ast); } .tool_call { background:var(--call); }
.tool_response { background:var(--resp); } .user { background:var(--user); }
.thinking { background:var(--think); font-style:italic; } .nothink { color:var(--mut); font-size:12px; margin:8px 0 0; }
pre { white-space:pre-wrap; word-break:break-word; margin:0; font:12.5px/1.4 ui-monospace, monospace; }
details summary { cursor:pointer; color:var(--mut); }
audio { display:block; margin:6px 0; width:100%; max-width:520px; }
.meta { color:var(--mut); font-size:12px; }
"""


def _rel(path: str, page: Path) -> str:
    return os.path.relpath(path, page.parent)


def _call_html(content: str, anchors: dict[str, str]) -> str:
    tc = json.loads(content)
    name, args = tc["name"], tc["arguments"]
    if name == "bash":
        body = f"$ {args['command']}"
    elif name == "task":
        body = f"[{args['subagent_type']}] {args['description']}\n\n{args['prompt']}"
        if args["prompt"] in anchors:
            return (f"<b>task</b> <a href='#{anchors[args['prompt']]}'>&rarr; subagent record</a>"
                    f"<pre>{html.escape(body)}</pre>")
    elif name == "write":
        body = f"{args['filePath']}\n\n{args['content']}"
    else:
        body = json.dumps(args, ensure_ascii=False, indent=1)
    return f"<b>{html.escape(name)}</b><pre>{html.escape(body)}</pre>"


def record_html(rec: dict, page: Path, anchors: dict[str, str]) -> str:
    out, ai, prev = [], 0, False
    marks = any("reasoning_content" in m for m in rec["messages"])
    for i, m in enumerate(rec["messages"]):
        role, c = m["role"], m["content"]
        step = role in ("assistant", "tool_call")
        if step and not prev:
            if "reasoning_content" in m:
                out.append(f"<div class='m thinking'><div class='r'>thinking</div>"
                           f"<pre>{html.escape(m['reasoning_content'])}</pre></div>")
            elif marks:
                out.append("<div class='nothink'>(no thinking: every attempt failed the checks)</div>")
        prev = step
        if role == "tool_call":
            body = _call_html(c, anchors)
        elif role == "user" and "<audio>" in c:
            n = c.count("<audio>")
            players = "".join(f"<div class='meta'>{html.escape(Path(a).name)}</div>"
                              f"<audio controls preload='none' src='{html.escape(_rel(a, page))}'></audio>"
                              for a in rec["audios"][ai:ai + n])
            ai += n
            body = f"<pre>{html.escape(c)}</pre>{players}"
        elif role == "system" or (role == "tool_response" and len(c) > 1500):
            body = (f"<details><summary>{len(c):,} chars</summary><pre>{html.escape(c)}</pre></details>")
        else:
            body = f"<pre>{html.escape(c)}</pre>"
        out.append(f"<div class='m {role}'><div class='r'>{i} {role}</div>{body}</div>")
    return "\n".join(out)


def rollout_page(main: dict, subs: list[dict], page: Path, title: str = "") -> str:
    anchors = {s["messages"][1]["content"]: f"rec-{i}" for i, s in enumerate(subs)}
    parts = [f"<h1>{html.escape(title or main['id'])}</h1>",
             f"<p class='meta'>{len(main['messages'])} messages, {len(main['audios'])} audio reads; "
             f"{len(subs)} subagent records</p>", "<h2 id='main'>maestro (main)</h2>", record_html(main, page, anchors)]
    for i, s in enumerate(subs):
        parts += [f"<h2 id='rec-{i}'>{html.escape(s['id'])}</h2>", record_html(s, page, anchors)]
    return (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width, initial-scale=1'><title>{html.escape(main['id'])}</title>"
            f"<style>{CSS}</style></head><body>{''.join(parts)}</body></html>")


def load_rollout(run_dir: Path, sample_id: str) -> tuple[dict, list[dict]]:
    idx = {json.loads(l)["sample_id"]: json.loads(l) for l in (run_dir / "index.jsonl").read_text().splitlines()}
    entry = idx[sample_id]
    recs = {}
    for agent in ("main", "transcriber", "searcher", "judge"):
        f = run_dir / f"{agent}.jsonl"
        if f.exists():
            for line in f.read_text().splitlines():
                r = json.loads(line)
                recs[r["id"]] = r
    return recs[entry["main_id"]], [recs[l["record_id"]] for l in entry["links"]]


def write_page(run_dir: Path, sample_id: str, out: Path | None = None) -> Path:
    out = out or run_dir / "view" / f"{sample_id}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    main, subs = load_rollout(run_dir, sample_id)
    out.write_text(rollout_page(main, subs, out.resolve(), f"v5 rollout {sample_id}"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("sample_id")
    ap.add_argument("-o", "--out", type=Path)
    a = ap.parse_args()
    print(write_page(a.run_dir, a.sample_id, a.out))


if __name__ == "__main__":
    main()
