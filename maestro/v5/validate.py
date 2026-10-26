"""The v5 record contract (replaces agent_sft_common.validate_oc_record for OpenCode-format records).

Single record (validate_record):
  - keys {id, messages, audios, tools}; tools == json.dumps(oc.tools_for(agent))
  - messages[0] system == oc.system_message(agent, cwd, date) for the record's own cwd and the fixed date;
    messages[1] user, plain text; roles only system/user/assistant/tool_call/tool_response
  - then steps: [assistant text] tool_call+ tool_response (same count, call order) [attachment user]; ends with
    assistant text; no two assistant messages in a row; no empty assistant message
  - tool_call = {"name", "arguments"} JSON; the tool is offered to the agent; arguments have the schema's required keys
    and no unknown keys
  - an audio read's response is AUDIO_READ_TEXT; a step with k audio reads is followed by exactly one user message
    MEDIA_ATTACHED_TEXT + "<audio>" * k (parallel reads allowed: "<audio><audio>"); a step without audio reads is not
    followed by a user message
  - "<audio>" appears only in those attachment messages; their total == len(audios); every audio file exists
  - task responses are <task id="ses_..." state="completed"><task_result>...</task_result></task>
Linked rollout (validate_rollout): every main task call has exactly one subagent record of that subagent_type whose
first user message is the task prompt verbatim, and the task result carries that record's final assistant text and the
linked session id; every subagent record is used once.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from scripts import opencode_contract as oc

from .record import DATE

ROLES = {"system", "user", "assistant", "tool_call", "tool_response"}
AUDIO_EXT = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus")
TASK_RE = re.compile(r'<task id="(ses_[0-9a-f]{12}[0-9A-Za-z]{14})" state="completed">\n<task_result>\n(.*)\n'
                     r'</task_result>\n</task>', re.S)
_CWD_RE = re.compile(r"^  Working directory: (.*)$", re.M)


def agent_of(record: dict) -> str | None:
    """The agent whose prompt opens the system message."""
    sys_msg = record["messages"][0]["content"]
    for a in oc.AGENTS:
        if sys_msg.startswith(oc.agent_prompt(a) + "\n"):
            return a
    return None


def _schema(agent: str) -> dict[str, dict]:
    return {t["function"]["name"]: t["function"]["parameters"] for t in oc.tools_for(agent)}


def validate_record(rec: dict, agent: str | None = None, date: str = DATE, check_files: bool = True) -> list[str]:
    err: list[str] = []
    if set(rec) != {"id", "messages", "audios", "tools"}:
        return [f"keys {sorted(rec)}"]
    msgs = rec["messages"]
    if len(msgs) < 3 or msgs[0]["role"] != "system" or msgs[1]["role"] != "user":
        return ["must start system, user and have a reply"]
    agent = agent or agent_of(rec)
    if agent is None:
        return ["system message matches no agent prompt"]
    m = _CWD_RE.search(msgs[0]["content"])
    if not m or msgs[0]["content"] != oc.system_message(agent, m.group(1), date=date):
        err.append("system message != oc.system_message(agent, cwd, date)")
    if rec["tools"] != json.dumps(oc.tools_for(agent), ensure_ascii=False):
        err.append("tools != oc.tools_for(agent)")
    for i, mm in enumerate(msgs):
        if mm.get("role") not in ROLES or not isinstance(mm.get("content"), str) or set(mm) != {"role", "content"}:
            err.append(f"[{i}] bad message {str(mm)[:80]}")
    if err:
        return err
    if "<audio>" in msgs[1]["content"]:
        err.append("[1] first user message holds audio")
    schema = _schema(agent)
    n_placeholders = 0
    i, n = 2, len(msgs)
    while i < n:
        if msgs[i]["role"] == "assistant":
            if not msgs[i]["content"].strip():
                err.append(f"[{i}] empty assistant message")
            if "<audio>" in msgs[i]["content"]:
                err.append(f"[{i}] <audio> in assistant text")
            if i + 1 < n and msgs[i + 1]["role"] == "assistant":
                err.append(f"[{i}] two assistant messages in a row")
            i += 1
            if i == n:
                break
        if msgs[i]["role"] != "tool_call":
            err.append(f"[{i}] expected tool_call, got {msgs[i]['role']}")
            i += 1
            continue
        calls = []
        while i < n and msgs[i]["role"] == "tool_call":
            calls.append((i, msgs[i]["content"]))
            i += 1
        resps = []
        while i < n and msgs[i]["role"] == "tool_response":
            resps.append((i, msgs[i]["content"]))
            i += 1
        if len(resps) != len(calls):
            err.append(f"[{calls[0][0]}] {len(calls)} tool calls, {len(resps)} responses")
            continue
        n_audio = 0
        for (ci, c), (ri, r) in zip(calls, resps):
            try:
                tc = json.loads(c)
                name, args = tc["name"], tc["arguments"]
                assert set(tc) == {"name", "arguments"} and isinstance(args, dict)
            except Exception:
                err.append(f"[{ci}] tool_call is not {{name, arguments}} JSON")
                continue
            if name not in schema:
                err.append(f"[{ci}] tool {name!r} not offered to {agent}")
                continue
            props, req = schema[name].get("properties", {}), schema[name].get("required", [])
            if missing := [k for k in req if k not in args]:
                err.append(f"[{ci}] {name}: missing {missing}")
            if unknown := [k for k in args if k not in props]:
                err.append(f"[{ci}] {name}: unknown args {unknown}")
            if name == "read" and str(args.get("filePath", "")).lower().endswith(AUDIO_EXT):
                n_audio += 1
                if r != oc.AUDIO_READ_TEXT:
                    err.append(f"[{ri}] audio read response != AUDIO_READ_TEXT")
            if name == "task":
                if not TASK_RE.fullmatch(r):
                    err.append(f"[{ri}] task response is not a completed <task> block")
                if args.get("subagent_type") not in [a for a in oc.AGENTS if oc.agent_mode(a) == "subagent"]:
                    err.append(f"[{ci}] unknown subagent_type {args.get('subagent_type')!r}")
            if name == "bash" and not r:
                err.append(f"[{ri}] empty bash response")
        if n_audio:
            want = oc.MEDIA_ATTACHED_TEXT + oc.AUDIO_PLACEHOLDER * n_audio
            if i >= n or msgs[i]["role"] != "user" or msgs[i]["content"] != want:
                err.append(f"[{i}] expected attachment user message with {n_audio} <audio>")
            else:
                n_placeholders += n_audio
                i += 1
        elif i < n and msgs[i]["role"] == "user":
            err.append(f"[{i}] user message after a step without audio reads")
            i += 1
    if msgs[-1]["role"] != "assistant":
        err.append("record does not end with assistant text")
    total = sum(mm["content"].count("<audio>") for mm in msgs)
    if total != n_placeholders:
        err.append(f"<audio> outside attachment messages ({total} vs {n_placeholders})")
    if n_placeholders != len(rec["audios"]):
        err.append(f"{n_placeholders} <audio> placeholders, {len(rec['audios'])} audios")
    if check_files:
        err += [f"missing audio {a}" for a in rec["audios"] if not Path(a).is_file()]
    return err


def task_calls(rec: dict) -> list[tuple[dict, str]]:
    """[(task arguments, task response)] of a record, in order."""
    msgs, out, i = rec["messages"], [], 0
    while i < len(msgs):
        if msgs[i]["role"] != "tool_call":
            i += 1
            continue
        calls = []
        while i < len(msgs) and msgs[i]["role"] == "tool_call":
            calls.append(json.loads(msgs[i]["content"]))
            i += 1
        resps = []
        while i < len(msgs) and msgs[i]["role"] == "tool_response":
            resps.append(msgs[i]["content"])
            i += 1
        out += [(c["arguments"], r) for c, r in zip(calls, resps) if c["name"] == "task"]
    return out


def validate_rollout(main: dict, subs: list[dict], links: list[dict] | None = None,
                     check_files: bool = True) -> list[str]:
    err = [f"{main['id']}: {e}" for e in validate_record(main, "maestro", check_files=check_files)]
    for s in subs:
        err += [f"{s['id']}: {e}" for e in validate_record(s, check_files=check_files)]
    used: set[str] = set()
    by_sid = {l["session_id"]: l for l in links or []}
    for args, resp in task_calls(main):
        cands = [s for s in subs if s["id"] not in used and s["messages"][1]["content"] == args["prompt"]
                 and agent_of(s) == args["subagent_type"]]
        if len(cands) != 1:
            err.append(f"task {args.get('description')!r}: {len(cands)} matching subagent records")
            continue
        s = cands[0]
        used.add(s["id"])
        m = TASK_RE.fullmatch(resp)
        if not m:
            continue
        if m.group(2) != s["messages"][-1]["content"]:
            err.append(f"task {args['description']!r}: result != {s['id']} final text")
        if links is not None:
            link = by_sid.get(m.group(1))
            if not link or link["record_id"] != s["id"]:
                err.append(f"task {args['description']!r}: session {m.group(1)} not linked to {s['id']}")
    for s in subs:
        if s["id"] not in used:
            err.append(f"{s['id']}: no task call in {main['id']} dispatches it")
    return err
