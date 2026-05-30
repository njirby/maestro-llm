#!/usr/bin/env python3
"""Generate an HTML spot-check viewer for SFT rollout JSONL files.

Usage:
    python scripts/build_sft_viewer.py \
        --main outputs/sft_smoke_16k/rollouts/main_final_smoke.jsonl \
        --search outputs/sft_smoke_16k/rollouts/search_final_smoke.jsonl \
        --judge outputs/sft_smoke_16k/rollouts/judge_final_smoke.jsonl \
        --transcription outputs/sft_smoke_16k/rollouts/transcription_final_smoke.jsonl \
        --output outputs/sft_smoke_16k/rollouts/viewer.html
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _render_message(msg: dict, idx: int) -> str:
    role = msg.get("role", "unknown")
    content = msg.get("content", "")

    role_class = role.replace("_", "-")
    role_label = role.replace("_", " ").title()

    if role == "tool_call":
        try:
            tc = json.loads(content)
            name = tc.get("name", "?")
            args = tc.get("arguments", {})
            args_str = json.dumps(args, indent=2)
            if len(args_str) > 600:
                args_str = args_str[:600] + "\n..."
            body = f'<span class="tool-name">{html.escape(name)}</span>\n<pre class="tool-args">{html.escape(args_str)}</pre>'
        except json.JSONDecodeError:
            body = f"<pre>{html.escape(content[:800])}</pre>"
    elif role == "tool_response":
        try:
            tr = json.loads(content)
            stdout = tr.get("stdout", "")
            stderr = tr.get("stderr", "")
            display = stdout
            if stderr:
                display += f"\n[stderr] {stderr}"
            if len(display) > 800:
                display = display[:800] + "\n..."
            body = f"<pre>{html.escape(display)}</pre>"
        except json.JSONDecodeError:
            display = content if len(content) <= 800 else content[:800] + "..."
            body = f"<pre>{html.escape(display)}</pre>"
    elif role == "assistant":
        escaped = html.escape(content)
        escaped = re.sub(r'`([^`]+)`', r'<code>\1</code>', escaped)
        if "<audio>" in content or "🔊" in content:
            body = f'<div class="has-audio">{escaped}</div>'
        else:
            body = f"<div>{escaped}</div>"
    elif role == "user":
        escaped = html.escape(content)
        escaped = escaped.replace("&lt;audio&gt;", '<span class="audio-tag">[AUDIO]</span>')
        body = f"<div>{escaped}</div>"
    else:
        body = f"<div>{html.escape(content[:500])}</div>"

    return f'<div class="message {role_class}" data-idx="{idx}"><span class="role-badge">{role_label}</span>{body}</div>'


def _render_meta(meta: dict) -> str:
    rows = []
    highlight_keys = {
        "sample_id", "archetype", "n_turns", "n_tool_calls", "n_batches_applied",
        "n_correction_turns", "n_search_rounds", "wall_time_s", "path_complete",
        "injected_mistakes", "mistake_caught", "gt_wavetable_names",
        "applied_wavetable_names",
    }
    for k in highlight_keys:
        if k in meta:
            v = meta[k]
            if isinstance(v, float):
                v = f"{v:.1f}"
            elif isinstance(v, (list, dict)):
                v = json.dumps(v)
            rows.append(f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>")
    return f'<table class="meta-table">{"".join(rows)}</table>'


def _render_record(rec: dict, agent_type: str, rec_idx: int) -> str:
    meta = rec.get("meta", {})
    messages = rec.get("messages", [])
    rec_id = rec.get("id", f"{agent_type}_{rec_idx}")
    sample_id = meta.get("sample_id", rec_id)
    archetype = meta.get("archetype", "")

    meta_html = _render_meta(meta)
    msgs_html = "\n".join(_render_message(m, i) for i, m in enumerate(messages))

    return f"""
    <div class="record" data-agent="{agent_type}" data-archetype="{archetype}" data-sample="{sample_id}">
      <div class="record-header" onclick="this.parentElement.classList.toggle('expanded')">
        <span class="agent-badge {agent_type}">{agent_type}</span>
        <span class="sample-id">{html.escape(str(sample_id))}</span>
        <span class="archetype-badge">{html.escape(archetype)}</span>
        <span class="msg-count">{len(messages)} msgs</span>
        <span class="expand-icon">▶</span>
      </div>
      <div class="record-body">
        {meta_html}
        <div class="messages">{msgs_html}</div>
      </div>
    </div>"""


def build_html(main: list[dict], search: list[dict], judge: list[dict],
               transcription: list[dict]) -> str:
    all_records = []
    for i, r in enumerate(main):
        all_records.append((r, "main", i))
    for i, r in enumerate(search):
        all_records.append((r, "search", i))
    for i, r in enumerate(judge):
        all_records.append((r, "judge", i))
    for i, r in enumerate(transcription):
        all_records.append((r, "transcription", i))

    archetypes = sorted(set(
        r.get("meta", {}).get("archetype", "unknown")
        for r, _, _ in all_records
    ))
    agent_types = ["main", "search", "judge", "transcription"]

    records_html = "\n".join(_render_record(r, at, idx) for r, at, idx in all_records)

    archetype_options = "".join(
        f'<option value="{a}">{a}</option>' for a in archetypes
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SFT Rollout Viewer</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'SF Mono', 'Cascadia Code', 'Consolas', monospace; background: #0d1117; color: #c9d1d9; padding: 20px; }}
h1 {{ color: #58a6ff; margin-bottom: 8px; font-size: 1.3em; }}
.stats {{ color: #8b949e; margin-bottom: 16px; font-size: 0.85em; }}
.filters {{ display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; align-items: center; }}
.filters select, .filters input {{ background: #161b22; border: 1px solid #30363d; color: #c9d1d9; padding: 6px 10px; border-radius: 6px; font-family: inherit; font-size: 0.85em; }}
.filters label {{ color: #8b949e; font-size: 0.85em; }}

.record {{ border: 1px solid #21262d; border-radius: 8px; margin-bottom: 8px; overflow: hidden; }}
.record-header {{ display: flex; align-items: center; gap: 10px; padding: 10px 14px; cursor: pointer; background: #161b22; }}
.record-header:hover {{ background: #1c2128; }}
.record-body {{ display: none; padding: 14px; border-top: 1px solid #21262d; background: #0d1117; }}
.record.expanded .record-body {{ display: block; }}
.record.expanded .expand-icon {{ transform: rotate(90deg); }}
.expand-icon {{ color: #484f58; transition: transform 0.15s; font-size: 0.8em; margin-left: auto; }}

.agent-badge {{ padding: 2px 8px; border-radius: 4px; font-size: 0.75em; font-weight: 600; text-transform: uppercase; }}
.agent-badge.main {{ background: #1f6feb33; color: #58a6ff; }}
.agent-badge.search {{ background: #23863533; color: #3fb950; }}
.agent-badge.judge {{ background: #9e6a0333; color: #d29922; }}
.agent-badge.transcription {{ background: #6e40c933; color: #bc8cff; }}

.sample-id {{ color: #c9d1d9; font-size: 0.85em; }}
.archetype-badge {{ background: #30363d; color: #8b949e; padding: 2px 8px; border-radius: 4px; font-size: 0.75em; }}
.msg-count {{ color: #484f58; font-size: 0.8em; }}

.meta-table {{ width: 100%; border-collapse: collapse; margin-bottom: 16px; font-size: 0.8em; }}
.meta-table td {{ padding: 4px 8px; border-bottom: 1px solid #21262d; }}
.meta-table td:first-child {{ color: #8b949e; width: 200px; }}

.messages {{ display: flex; flex-direction: column; gap: 6px; }}
.message {{ padding: 8px 12px; border-radius: 6px; font-size: 0.82em; line-height: 1.5; }}
.message.assistant {{ background: #1f6feb15; border-left: 3px solid #1f6feb; }}
.message.user {{ background: #23863515; border-left: 3px solid #238636; }}
.message.tool-call {{ background: #9e6a0310; border-left: 3px solid #9e6a03; }}
.message.tool-response {{ background: #161b22; border-left: 3px solid #30363d; }}
.role-badge {{ display: inline-block; font-size: 0.7em; font-weight: 600; text-transform: uppercase; color: #8b949e; margin-right: 8px; min-width: 90px; }}
.tool-name {{ color: #d2a8ff; font-weight: 600; }}
.tool-args {{ margin-top: 4px; font-size: 0.9em; color: #8b949e; max-height: 200px; overflow-y: auto; white-space: pre-wrap; word-break: break-all; }}
.message pre {{ white-space: pre-wrap; word-break: break-all; max-height: 300px; overflow-y: auto; }}
.audio-tag {{ background: #238636; color: #fff; padding: 1px 6px; border-radius: 3px; font-size: 0.85em; }}
.has-audio {{ color: #c9d1d9; }}
code {{ background: #30363d; padding: 1px 4px; border-radius: 3px; font-size: 0.9em; }}

.hidden {{ display: none !important; }}
</style>
</head>
<body>
<h1>SFT Rollout Viewer</h1>
<div class="stats">{len(main)} main &middot; {len(search)} search &middot; {len(judge)} judge &middot; {len(transcription)} transcription</div>

<div class="filters">
  <label>Agent:</label>
  <select id="filter-agent">
    <option value="all">All</option>
    <option value="main">Main</option>
    <option value="search">Search</option>
    <option value="judge">Judge</option>
    <option value="transcription">Transcription</option>
  </select>
  <label>Archetype:</label>
  <select id="filter-archetype">
    <option value="all">All</option>
    {archetype_options}
  </select>
  <label>Sample ID:</label>
  <input id="filter-sample" type="text" placeholder="filter..." />
</div>

<div id="records">
{records_html}
</div>

<script>
const agentSel = document.getElementById('filter-agent');
const archSel = document.getElementById('filter-archetype');
const sampleInput = document.getElementById('filter-sample');
const records = document.querySelectorAll('.record');

function applyFilters() {{
  const agent = agentSel.value;
  const arch = archSel.value;
  const sample = sampleInput.value.toLowerCase();
  records.forEach(r => {{
    const matchAgent = agent === 'all' || r.dataset.agent === agent;
    const matchArch = arch === 'all' || r.dataset.archetype === arch;
    const matchSample = !sample || r.dataset.sample.toLowerCase().includes(sample);
    r.classList.toggle('hidden', !(matchAgent && matchArch && matchSample));
  }});
}}

agentSel.addEventListener('change', applyFilters);
archSel.addEventListener('change', applyFilters);
sampleInput.addEventListener('input', applyFilters);
</script>
</body>
</html>"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--main", type=Path, required=True)
    p.add_argument("--search", type=Path, default=None)
    p.add_argument("--judge", type=Path, default=None)
    p.add_argument("--transcription", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    main_recs = _load_jsonl(args.main)
    search_recs = _load_jsonl(args.search) if args.search else []
    judge_recs = _load_jsonl(args.judge) if args.judge else []
    trans_recs = _load_jsonl(args.transcription) if args.transcription else []

    print(f"Loaded: {len(main_recs)} main, {len(search_recs)} search, "
          f"{len(judge_recs)} judge, {len(trans_recs)} transcription")

    content = build_html(main_recs, search_recs, judge_recs, trans_recs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content)
    print(f"Written to {args.output} ({args.output.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
