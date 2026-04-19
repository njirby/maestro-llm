#!/usr/bin/env python3
"""Build a self-contained HTML spot-check viewer for agent rollouts.

Renders each record as a sequence of message cards. Tool calls and
responses are collapsed by default to keep the view scannable; expand
them to see the full bash command or JSON body. Audio is not embedded.

Usage:
    python scripts/build_rollout_spotcheck.py \\
        --main outputs/smoke_v3/main_final8_v15.jsonl \\
        --transcription outputs/smoke_v3/transcription_final8_v15.jsonl \\
        --search outputs/smoke_v3/search_final8_v15.jsonl \\
        --judge outputs/smoke_v3/judge_final8_v15.jsonl \\
        --out outputs/smoke_v3/rollouts_v15.html
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path


def _try_parse_json(text: str) -> dict | list | None:
    try:
        v = json.loads(text)
        return v if isinstance(v, (dict, list)) else None
    except Exception:
        return None


def _pretty_json(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def _extract_heredoc(command: str) -> tuple[str, str] | None:
    """If command is `python - <<'TAG' ... TAG` or similar, return (lang, body)."""
    m = re.search(
        r"<<\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?\s*\n(.*?)\n\1\s*$",
        command,
        re.DOTALL,
    )
    if not m:
        return None
    body = m.group(2)
    # Guess language from the invocation line before the heredoc.
    prefix = command[: m.start()]
    lang = "text"
    if re.search(r"\bpython\b", prefix):
        lang = "python"
    elif re.search(r"\blua\b", prefix):
        lang = "lua"
    elif re.search(r"\bbash\b|\bsh\b", prefix):
        lang = "bash"
    return lang, body.rstrip()


def _message_preview(content: str, role: str) -> str:
    """Short one-line preview for collapsed summary lines."""
    if role == "tool_call":
        parsed = _try_parse_json(content)
        if isinstance(parsed, dict):
            name = parsed.get("name", "?")
            args = parsed.get("arguments", {}) or {}
            if name == "bash":
                cmd = (args.get("command") or "").strip().splitlines()
                first = cmd[0] if cmd else ""
                return f"bash ❯ {first[:140]}"
            if name == "Agent":
                desc = args.get("description", "") or args.get("subagent_type", "")
                return f"Agent ❯ {desc[:140]}"
            if name == "Skill":
                return f"Skill ❯ {args.get('skill', '')}"
            return f"{name}"
        return content[:140]
    if role == "tool_response":
        parsed = _try_parse_json(content)
        if isinstance(parsed, dict):
            status = parsed.get("status", "")
            keys = ",".join(k for k in list(parsed.keys())[:5])
            return f"{status} {{{keys}}}" if status else f"{{{keys}}}"
        return content[:140].replace("\n", " ")
    return content[:140].replace("\n", " ")


def render_bash_body(command: str) -> str:
    """Render a bash command. If it contains a heredoc, show the inner
    script in its own code block."""
    hd = _extract_heredoc(command)
    if hd is None:
        return f'<pre class="code bash">{html.escape(command)}</pre>'
    lang, body = hd
    return (
        f'<pre class="code bash"><span class="comment"># shell</span>\n'
        f'{html.escape(command[: command.find("<<")].rstrip())}</pre>'
        f'<pre class="code {lang}"><span class="comment"># {lang}</span>\n'
        f'{html.escape(body)}</pre>'
    )


def render_tool_call_body(content: str) -> str:
    parsed = _try_parse_json(content)
    if not isinstance(parsed, dict):
        return f'<pre class="code text">{html.escape(content)}</pre>'
    name = parsed.get("name", "?")
    args = parsed.get("arguments", {}) or {}
    if name == "bash" and isinstance(args, dict) and "command" in args:
        return render_bash_body(args["command"])
    return f'<pre class="code json">{html.escape(_pretty_json(parsed))}</pre>'


def render_tool_response_body(content: str) -> str:
    parsed = _try_parse_json(content)
    if parsed is not None:
        return f'<pre class="code json">{html.escape(_pretty_json(parsed))}</pre>'
    return f'<pre class="code text">{html.escape(content)}</pre>'


def render_message(msg: dict) -> str:
    role = msg.get("role", "?")
    content = msg.get("content", "") or ""
    role_cls = f"role-{role.replace('_', '-')}"

    if role == "tool_call":
        body = render_tool_call_body(content)
        preview = _message_preview(content, role)
        return (
            f'<details class="msg {role_cls}">'
            f'<summary><span class="role-tag">tool_call</span> '
            f'<span class="preview">{html.escape(preview)}</span></summary>'
            f'<div class="msg-body">{body}</div>'
            f'</details>'
        )
    if role == "tool_response":
        body = render_tool_response_body(content)
        preview = _message_preview(content, role)
        return (
            f'<details class="msg {role_cls}">'
            f'<summary><span class="role-tag">tool_response</span> '
            f'<span class="preview">{html.escape(preview)}</span></summary>'
            f'<div class="msg-body">{body}</div>'
            f'</details>'
        )

    # user / assistant / other — plain prose, strip <audio> placeholders.
    clean = content.replace("<audio>", "🔊 ")
    safe = html.escape(clean).replace("\n", "<br>")
    return (
        f'<div class="msg {role_cls}">'
        f'<span class="role-tag">{html.escape(role)}</span>'
        f'<div class="msg-body prose">{safe}</div>'
        f'</div>'
    )


def render_record(rec: dict, idx: int, agent: str) -> str:
    sid = rec.get("id") or rec.get("meta", {}).get("sample_id") or f"record-{idx}"
    task_type = rec.get("task_type", "?")
    meta = rec.get("meta") or {}
    archetype = meta.get("archetype", "")
    version = meta.get("pipeline_version", "")

    messages = rec.get("messages", []) or []
    msgs_html = "\n".join(render_message(m) for m in messages)

    first_user = next(
        (m.get("content", "") for m in messages if m.get("role") == "user"),
        "",
    )
    preview = first_user.replace("<audio>", "🔊").replace("\n", " ")[:140]

    chips = []
    if archetype:
        chips.append(f'<span class="chip">{html.escape(archetype)}</span>')
    if task_type:
        chips.append(f'<span class="chip">{html.escape(task_type)}</span>')
    if version:
        chips.append(f'<span class="chip">{html.escape(version)}</span>')
    chips.append(f'<span class="chip">{len(messages)} msgs</span>')

    return f'''
<details class="record" data-agent="{html.escape(agent)}">
  <summary>
    <span class="sid">{html.escape(str(sid))}</span>
    <span class="chips">{''.join(chips)}</span>
    <span class="rec-preview">{html.escape(preview)}</span>
  </summary>
  <div class="record-body">{msgs_html}</div>
</details>
'''


def render_agent_section(agent_name: str, records: list[dict]) -> str:
    if not records:
        return ""
    records_html = "\n".join(
        render_record(r, i, agent_name) for i, r in enumerate(records)
    )
    return (
        f'<section class="agent" id="agent-{html.escape(agent_name)}">'
        f'<h2>{html.escape(agent_name)} <span class="count">({len(records)} records)</span></h2>'
        f'{records_html}</section>'
    )


def load_jsonl(path: Path) -> list[dict]:
    recs: list[dict] = []
    if not path.exists():
        return recs
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs


_CSS = """
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  margin: 0; padding: 24px;
  background: #0f1115; color: #e8e8e8;
  line-height: 1.55;
  max-width: 1200px; margin-left: auto; margin-right: auto;
}
h1 { margin: 0 0 16px; font-size: 22px; }
h2 { margin: 32px 0 12px; font-size: 18px; border-bottom: 1px solid #2a2f3a; padding-bottom: 6px; }
h2 .count { color: #8a97ab; font-weight: normal; font-size: 14px; margin-left: 4px; }
nav.toc {
  position: sticky; top: 0;
  background: rgba(15, 17, 21, 0.96); backdrop-filter: blur(4px);
  padding: 10px 0; border-bottom: 1px solid #2a2f3a; margin-bottom: 16px; z-index: 10;
}
nav.toc a { color: #7ec4ff; margin-right: 16px; text-decoration: none; }
nav.toc a:hover { text-decoration: underline; }

.controls { margin: 12px 0; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
button { background: #2a2f3a; color: #e8e8e8; border: 1px solid #3a4250; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 12px; }
button:hover { background: #353b47; }

.record { background: #1a1d24; border: 1px solid #2a2f3a; border-radius: 6px; margin-bottom: 10px; overflow: hidden; }
.record > summary { padding: 10px 14px; cursor: pointer; list-style: none; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
.record > summary::-webkit-details-marker { display: none; }
.record > summary::before { content: "▶"; font-size: 10px; color: #888; transition: transform 0.15s; }
.record[open] > summary::before { transform: rotate(90deg); }
.record > summary:hover { background: #20242e; }
.record-body { padding: 10px 14px 16px; border-top: 1px solid #2a2f3a; }

.sid { font-weight: 600; font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 13px; }
.rec-preview { color: #9aa; font-size: 12px; flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.chips { display: flex; gap: 6px; flex-wrap: wrap; }
.chip { background: #2a2f3a; padding: 2px 8px; border-radius: 10px; font-size: 11px; color: #b8c2d1; }

.msg { margin: 6px 0; border-left: 3px solid #2a2f3a; background: #13161d; border-radius: 0 4px 4px 0; }
.msg:not(details) { padding: 6px 10px 8px; }
details.msg > summary { padding: 6px 10px; cursor: pointer; list-style: none; display: flex; gap: 8px; align-items: baseline; }
details.msg > summary::-webkit-details-marker { display: none; }
details.msg > summary::before { content: "▸"; color: #666; font-size: 10px; }
details.msg[open] > summary::before { content: "▾"; }
details.msg > summary:hover { background: #1a1e28; }
.msg-body { padding: 2px 10px 10px; }
.msg-body.prose { padding-top: 0; }

.role-tag { font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; color: #8a97ab; font-weight: 700; padding: 1px 6px; border-radius: 3px; background: #1f232c; white-space: nowrap; }
.msg.role-user { border-left-color: #4a9eff; }
.msg.role-user .role-tag { color: #7ec4ff; }
.msg.role-assistant { border-left-color: #58d68d; }
.msg.role-assistant .role-tag { color: #8fe4b0; }
.msg.role-tool-call { border-left-color: #e67e22; background: #1a1810; }
.msg.role-tool-call .role-tag { color: #ffae70; }
.msg.role-tool-response { border-left-color: #bb8fce; background: #15131a; }
.msg.role-tool-response .role-tag { color: #d1b0e0; }

.preview { color: #b8c2d1; font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; min-width: 0; }

pre.code {
  background: #0b0d12; padding: 10px 12px; border-radius: 4px;
  overflow-x: auto; font-size: 12px; color: #d0d7e0;
  font-family: "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
  white-space: pre-wrap; word-break: break-word;
  margin: 6px 0 0 0;
  max-height: 500px; overflow-y: auto;
  border: 1px solid #1c2028;
  line-height: 1.45;
}
pre.code.json { border-left: 2px solid #4a9eff; }
pre.code.bash { border-left: 2px solid #e67e22; }
pre.code.python { border-left: 2px solid #58d68d; }
pre.code.lua { border-left: 2px solid #bb8fce; }
pre.code .comment { color: #5a6475; font-style: italic; }
code { background: #0b0d12; padding: 2px 5px; border-radius: 3px; font-size: 12px; font-family: "SF Mono", Menlo, Consolas, monospace; }

.prose { color: #e0e6ed; }
"""

_JS = """
function toggleAll(agent, open) {
  const sel = agent
    ? `section#agent-${agent} details.record`
    : `details.record`;
  document.querySelectorAll(sel).forEach(d => d.open = open);
}
function toggleAllMsgs(open) {
  document.querySelectorAll('details.msg').forEach(d => d.open = open);
}
"""


def build_html(by_agent: dict[str, list[dict]], title: str) -> str:
    toc_links = []
    sections = []
    for agent, records in by_agent.items():
        if not records:
            continue
        toc_links.append(
            f'<a href="#agent-{html.escape(agent)}">{html.escape(agent)} '
            f'({len(records)})</a>'
        )
        sections.append(render_agent_section(agent, records))

    toc = f'<nav class="toc">{" ".join(toc_links)}</nav>' if toc_links else ""
    body = "\n".join(sections) if sections else "<p>No records to display.</p>"

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
{toc}
<div class="controls">
  <button onclick="toggleAll(null, true)">Expand records</button>
  <button onclick="toggleAll(null, false)">Collapse records</button>
  <button onclick="toggleAllMsgs(true)">Expand tool messages</button>
  <button onclick="toggleAllMsgs(false)">Collapse tool messages</button>
</div>
{body}
<script>{_JS}</script>
</body>
</html>
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", type=Path, default=None)
    ap.add_argument("--transcription", type=Path, default=None)
    ap.add_argument("--search", type=Path, default=None)
    ap.add_argument("--judge", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="Agent rollout spot-check")
    args = ap.parse_args()

    by_agent: dict[str, list[dict]] = {}
    for agent_name, path in [
        ("main", args.main),
        ("transcription", args.transcription),
        ("search", args.search),
        ("judge", args.judge),
    ]:
        if path is None:
            continue
        by_agent[agent_name] = load_jsonl(path)
        print(
            f"{agent_name}: {len(by_agent[agent_name])} records from {path}",
            file=sys.stderr,
        )

    if not any(by_agent.values()):
        print("No records loaded — nothing to render.", file=sys.stderr)
        sys.exit(1)

    html_text = build_html(by_agent, title=args.title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_text)
    size_kb = args.out.stat().st_size / 1024
    print(f"Wrote {args.out} ({size_kb:.1f} KB)", file=sys.stderr)


if __name__ == "__main__":
    main()
