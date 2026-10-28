"""Record <-> OpenCode wire conversion, normalization and diffing for the conformance replay.

Stdlib only: the replay mock server imports this inside the daw-farm container.

Record form is the ms-swift schema the builders write ({"id", "messages", "audios", "tools"}; roles
system/user/assistant/tool_call/tool_response). Wire form is the OpenAI chat request OpenCode 1.19.3
sends. One assistant *step* on the wire is one assistant message (text and/or tool_calls); in a record
it is a run of [assistant][tool_call...] messages, followed by its tool_responses and, if the step read
media, one "Attached media from tool result:<audio>..." user message (scripts/opencode_contract.py).

The replay compares in record form: every captured request is converted with wire_to_record() and
compared to the record prefix that precedes the step the request asks for (expected_prefix()).
Media is compared by sha256 of the decoded bytes against the record's `audios` files.
"""
from __future__ import annotations

import base64
import difflib
import hashlib
import json
import re
from typing import Any, Callable, Iterable

MEDIA_ATTACHED_TEXT = "Attached media from tool result:"
AUDIO_PLACEHOLDER = "<audio>"
IMAGE_PLACEHOLDER = "<image>"
TITLE_SYSTEM_PREFIX = "You are a title generator."
ROLLOUT_ROLES = ("assistant", "tool_call")


# ---------------------------------------------------------------------------
# media
# ---------------------------------------------------------------------------

def media_digest(b64: str) -> dict[str, Any]:
    """What the replay logs instead of base64 media: sha256 + size of the decoded bytes."""
    raw = base64.b64decode(b64)
    return {"sha256": hashlib.sha256(raw).hexdigest(), "nbytes": len(raw)}


def file_digest(path: str) -> dict[str, Any]:
    raw = open(path, "rb").read()
    return {"sha256": hashlib.sha256(raw).hexdigest(), "nbytes": len(raw)}


def strip_media(body: Any) -> Any:
    """Copy of a request body with every base64 media payload replaced by media_digest()."""
    if isinstance(body, dict):
        if body.get("type") == "input_audio" and isinstance(body.get("input_audio"), dict):
            ia = body["input_audio"]
            data = ia.get("data")
            return {**body, "input_audio": {**ia, "data": media_digest(data) if isinstance(data, str) else data}}
        if body.get("type") == "image_url" and isinstance(body.get("image_url"), dict):
            url = body["image_url"].get("url", "")
            if isinstance(url, str) and url.startswith("data:") and "," in url:
                head, data = url.split(",", 1)
                return {**body, "image_url": {**body["image_url"], "url": {"head": head, **media_digest(data)}}}
        return {k: strip_media(v) for k, v in body.items()}
    if isinstance(body, list):
        return [strip_media(v) for v in body]
    return body


def media_payloads(body: Any) -> list[tuple[str, bytes, str]]:
    """(sha256, decoded bytes, extension) of every base64 media part of a request body, in order (the
    mock saves them so re-rendered audio can be compared by content, not only by hash)."""
    out: list[tuple[str, bytes, str]] = []
    if isinstance(body, dict):
        if body.get("type") == "input_audio" and isinstance(body.get("input_audio"), dict) \
                and isinstance(body["input_audio"].get("data"), str):
            raw = base64.b64decode(body["input_audio"]["data"])
            return [(hashlib.sha256(raw).hexdigest(), raw, body["input_audio"].get("format") or "bin")]
        if body.get("type") == "image_url" and isinstance(body.get("image_url"), dict):
            url = body["image_url"].get("url", "")
            if isinstance(url, str) and url.startswith("data:") and "," in url:
                head, data = url.split(",", 1)
                raw = base64.b64decode(data)
                ext = head[5:].split(";", 1)[0].rsplit("/", 1)[-1] or "bin"
                return [(hashlib.sha256(raw).hexdigest(), raw, ext)]
        for v in body.values():
            out += media_payloads(v)
    elif isinstance(body, list):
        for v in body:
            out += media_payloads(v)
    return out


# ---------------------------------------------------------------------------
# wire -> record
# ---------------------------------------------------------------------------

def _canon_call(name: str, arguments: Any) -> str:
    """Canonical tool_call content: {"name", "arguments"} with argument key order kept and
    whitespace normalized. Applied to both sides, so only names, values and key order count."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            pass  # kept as the raw string; differs from any parsed record call
    return json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)


def canon_tool_call_content(content: str) -> str:
    d = json.loads(content)
    return _canon_call(d["name"], d.get("arguments", {}))


def wire_to_record(messages: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[Any]]:
    """Wire messages -> (record messages, media payloads in placeholder order).

    List content is joined part by part with no separator (input_audio -> <audio>, image_url ->
    <image>), as scripts/opencode_contract.py renders the media message. A media payload is whatever
    the wire carried (base64 str, or the digest dict of a logged body)."""
    out: list[dict[str, str]] = []
    media: list[Any] = []
    for m in messages:
        role = m["role"]
        c = m.get("content")
        if isinstance(c, list):
            parts = []
            for p in c:
                t = p.get("type")
                if t == "text":
                    parts.append(p["text"])
                elif t == "input_audio":
                    parts.append(AUDIO_PLACEHOLDER)
                    media.append(p["input_audio"]["data"])
                elif t == "image_url":
                    parts.append(IMAGE_PLACEHOLDER)
                    url = p["image_url"]["url"]
                    media.append(url.split(",", 1)[1] if isinstance(url, str) else url)
                else:
                    parts.append("<unsupported part %s>" % json.dumps(p, ensure_ascii=False)[:200])
            c = "".join(parts)
        if role == "assistant":
            if c:
                out.append({"role": "assistant", "content": c})
            for tc in m.get("tool_calls") or []:
                out.append({"role": "tool_call",
                            "content": _canon_call(tc["function"]["name"], tc["function"].get("arguments", ""))})
            if not c and not m.get("tool_calls"):
                out.append({"role": "assistant", "content": ""})
        elif role == "tool":
            out.append({"role": "tool_response", "content": c if isinstance(c, str) else json.dumps(c)})
        else:
            out.append({"role": role, "content": c if c is not None else ""})
    return out, media


# ---------------------------------------------------------------------------
# record structure
# ---------------------------------------------------------------------------

def record_steps(messages: list[dict[str, str]]) -> list[tuple[int, int]]:
    """(start, end) index ranges of the assistant steps (runs of assistant/tool_call messages)."""
    steps, i = [], 0
    while i < len(messages):
        if messages[i]["role"] in ROLLOUT_ROLES:
            j = i
            while j < len(messages) and messages[j]["role"] in ROLLOUT_ROLES:
                j += 1
            steps.append((i, j))
            i = j
        else:
            i += 1
    return steps


def step_reply(messages: list[dict[str, str]], start: int, end: int) -> dict[str, Any]:
    """What the mock answers for a step: {"text": str|None, "calls": [[name, arguments_json], ...]}.

    A record call of OpenCode's hidden `invalid` tool ({"tool": X, "error": ...}) is what OpenCode
    writes into history after the model called an unavailable tool X; the mock re-emits a call of X."""
    text, calls = None, []
    for m in messages[start:end]:
        if m["role"] == "assistant":
            text = m["content"] if text is None else text + m["content"]
        else:
            d = json.loads(m["content"])
            name, args = d["name"], d.get("arguments", {})
            if name == "invalid" and isinstance(args, dict) and "tool" in args:
                name, args = args["tool"], {}
            calls.append([name, args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)])
    return {"text": text, "calls": calls}


def expected_prefix(record: dict[str, Any], step: int) -> list[dict[str, str]]:
    """Record messages the model sees when asked for assistant step `step` (0-based)."""
    msgs = record["messages"]
    steps = record_steps(msgs)
    return msgs[: steps[step][0]] if step < len(steps) else msgs


def count_media(messages: Iterable[dict[str, str]]) -> int:
    return sum(m["content"].count(AUDIO_PLACEHOLDER) + m["content"].count(IMAGE_PLACEHOLDER)
               for m in messages if m["role"] == "user" and isinstance(m.get("content"), str))


def record_to_wire(messages: list[dict[str, str]], media: list[Any], agent: str = "maestro",
                   media_formats: list[str] | None = None) -> list[dict[str, Any]]:
    """Record messages -> the wire messages OpenCode would send (inverse of wire_to_record()).

    Tool call ids follow the mock's scheme call_<agent>_<step>_<i>; user messages with placeholders
    become text/input_audio/image_url part lists, `media` supplying payloads in placeholder order."""
    out: list[dict[str, Any]] = []
    mi = 0
    step = -1
    pending: list[str] = []
    for m in messages:
        role, c = m["role"], m["content"]
        if role in ROLLOUT_ROLES:
            if not out or out[-1]["role"] != "assistant" or out[-1].get("_closed"):
                step += 1
                out.append({"role": "assistant", "content": ""})
            a = out[-1]
            if role == "assistant":
                a["content"] += c
            else:
                d = json.loads(c)
                tcs = a.setdefault("tool_calls", [])
                cid = f"call_{agent}_{step}_{len(tcs)}"
                tcs.append({"id": cid, "type": "function",
                            "function": {"name": d["name"],
                                         "arguments": json.dumps(d.get("arguments", {}), ensure_ascii=False,
                                                                 separators=(",", ":"))}})
                pending.append(cid)
            continue
        if out and out[-1]["role"] == "assistant":
            out[-1]["_closed"] = True
        if role == "tool_response":
            out.append({"role": "tool", "tool_call_id": pending.pop(0) if pending else "", "content": c})
        elif role == "user" and (AUDIO_PLACEHOLDER in c or IMAGE_PLACEHOLDER in c):
            parts: list[dict[str, Any]] = []
            for tok in re.split(r"(<audio>|<image>)", c):
                if tok == AUDIO_PLACEHOLDER:
                    fmt = media_formats[mi] if media_formats else "wav"
                    parts.append({"type": "input_audio", "input_audio": {"data": media[mi], "format": fmt}})
                    mi += 1
                elif tok == IMAGE_PLACEHOLDER:
                    parts.append({"type": "image_url", "image_url": {"url": media[mi]}})
                    mi += 1
                elif tok:
                    parts.append({"type": "text", "text": tok})
            out.append({"role": "user", "content": parts})
        else:
            out.append({"role": role, "content": c})
    for m in out:
        m.pop("_closed", None)
    return out


# ---------------------------------------------------------------------------
# normalization (the whitelist of allowed run-to-run differences)
# ---------------------------------------------------------------------------

_SUBS = [
    (re.compile(r"Today's date: \w{3} \w{3} \d{2} \d{4}"), "Today's date: {DATE}"),
    (re.compile(r"ses_[0-9a-f]{12}[0-9A-Za-z]{14}"), "{SES_ID}"),
    (re.compile(r"tool_[0-9a-f]{12}[0-9A-Za-z]{14}"), "{TOOL_OUTPUT_ID}"),
    (re.compile(r"/tmp/tmp[A-Za-z0-9_]{6,}"), "{TMPDIR}"),
]
_SKILL_FILES = re.compile(r"(Base directory for this skill: )(\S+)(\n(?:.*\n)*?<skill_files>\n)(.*?)(\n</skill_files>)", re.S)
_SHELL_META = re.compile(r"\n*<shell_metadata>.*?</shell_metadata>", re.S)

WHITELIST = {
    "date": "`Today's date: <Date.toDateString()>` in the <env> block",
    "ses_id": "ses_<26> session ids (task results)",
    "tool_output_id": "tool_<26> bash spill-file ids",
    "tmpdir": "/tmp/tmp* paths (python tempfile)",
    "skill_files": "the <skill_files> sample (ripgrep order, max 10): only the count and that every "
                   "entry lies under the skill's base directory are compared",
    "glob_order": "glob result lines (mtime order): compared as a sorted list",
    "stream_interleave": "bash output whose only difference is where stderr-shaped lines (error:/warning:) "
                         "land among stdout lines (OpenCode reads the two pipes separately: a race); "
                         "same lines, everything else in order",
    "media_render": "audio produced during the rollout (not in the initial workdir) whose sha256 differs: "
                    "equivalent by state (the producing tool calls and final non-audio workdir are "
                    "byte-identical); fails only on gross breakage: length > one block off, silence "
                    "mismatch (peak < -60 dBFS on one side only), mean |log-mel diff| > max_mel_db "
                    "(default 10 dB); mel_db is informational (audio_eq)",
    "shell_metadata": "the <shell_metadata> block OpenCode appends when a bash call exceeds its 120 s shell "
                      "timeout (replay contention: the parallel subagents of one task step share a single "
                      "container); the command output itself must still match",
}


def _norm_skill_files(text: str) -> str:
    def sub(m: re.Match) -> str:
        base, files = m.group(2), m.group(4)
        lines = files.split("\n") if files else []
        ok = all(re.fullmatch(r"<file>" + re.escape(base) + r"/.+</file>", ln) for ln in lines)
        body = "\n".join("<file>{SKILL_FILE}</file>" for _ in lines) if ok else files
        return m.group(1) + base + m.group(3) + body + m.group(5)
    return _SKILL_FILES.sub(sub, text)


def norm_text(s: str) -> str:
    for rx, rep in _SUBS:
        s = rx.sub(rep, s)
    s = _SHELL_META.sub("", s)
    if "<skill_files>" in s:
        s = _norm_skill_files(s)
    return s.rstrip("\n")  # trailing blank lines in tool output are not stable across runs


def _norm_glob(s: str) -> str:
    lines = s.split("\n")
    body = [ln for ln in lines if ln and not ln.startswith("(Results are truncated")]
    tail = [ln for ln in lines if ln.startswith("(Results are truncated")]
    if s == "No files found":
        return s
    return "\n".join(sorted(body) + ([""] + tail if tail else []))


def normalize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Apply the whitelist: volatile strings, skill-file sample, glob order. Tool_call contents are
    canonicalized (whitespace-insensitive JSON, key order kept)."""
    out = []
    names: list[str] = []
    prev = None
    for m in messages:
        role, c = m["role"], m.get("content", "")
        if role == "tool_call":
            if prev != "tool_call":
                names = []
            try:
                c = canon_tool_call_content(c)
                names.append(json.loads(c)["name"])
            except (json.JSONDecodeError, KeyError):
                names.append("?")
        elif role == "tool_response":
            name = names.pop(0) if names else "?"
            if name == "glob":
                c = _norm_glob(c)
        prev = role
        out.append({"role": role, "content": norm_text(c) if isinstance(c, str) else c})
    return out


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

def _render(messages: list[dict[str, str]]) -> list[str]:
    lines = []
    for i, m in enumerate(messages):
        lines.append(f"### [{i}] {m['role']}")
        lines.extend(str(m.get("content", "")).split("\n"))
    return lines


def unified(a: list[str], b: list[str], fa: str, fb: str, context: int = 2, max_lines: int = 80) -> str:
    d = list(difflib.unified_diff(a, b, fa, fb, n=context, lineterm=""))
    if len(d) > max_lines:
        d = d[:max_lines] + [f"... ({len(d) - max_lines} more diff lines)"]
    return "\n".join(d)


_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def mask_numbers(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{**m, "content": _NUM.sub("{N}", m["content"]) if isinstance(m.get("content"), str) else m.get("content")}
            for m in messages]


STDERR_LINE = re.compile(r"(?i)^(error|warning)\b")


def stream_canon(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Tool responses with their stderr-shaped lines ("error: ...", "warning: ...") moved to the end in
    sorted order. OpenCode's bash tool reads stdout and stderr through separate listeners, so where a
    stderr line lands among stdout lines is a race; two outputs equal under this canon hold the same
    lines with only stderr-shaped lines moved (whitelist: stream_interleave)."""
    out = []
    for m in messages:
        c = m.get("content")
        if m["role"] == "tool_response" and isinstance(c, str):
            lines = c.split("\n")
            err = [ln for ln in lines if STDERR_LINE.match(ln)]
            if err:
                c = "\n".join([ln for ln in lines if not STDERR_LINE.match(ln)] + ["--stderr--"] + sorted(err))
        out.append({**m, "content": c})
    return out


MediaEq = Callable[[int, str], "dict[str, Any] | None"]


def compare_request(body: dict[str, Any], record: dict[str, Any], step: int,
                    media_digests: list[dict[str, Any]], expected_params: dict[str, Any] | None = None,
                    media_eq: MediaEq | None = None) -> dict[str, Any]:
    """Diff one captured request (media already stripped to digests) against the record.

    Message diffs that vanish when every number is masked are kind "numeric" (timing / randomness /
    re-rendered audio leaking into tool output) rather than "messages". A media item whose sha256
    differs is passed to media_eq(index into the record's audios, got sha256); if that returns
    {"equivalent": True, ...} it is whitelisted (listed under "whitelisted"), else a "media" diff.

    Returns {"status": "match"|"diff", "diffs": [{"kind", "detail"}...], "whitelisted": [...]}."""
    diffs: list[dict[str, Any]] = []
    whitelisted: list[dict[str, Any]] = []
    got, got_media = wire_to_record(body.get("messages", []))
    exp = expected_prefix(record, step)
    g, e = normalize_messages(got), normalize_messages(exp)
    if g != e:
        # system separately (it is long and identical across steps)
        if g[:1] != e[:1]:
            diffs.append({"kind": "system", "detail": unified(_render(e[:1]), _render(g[:1]), "record", "opencode")})
        if g[1:] != e[1:]:
            detail = unified(_render(e[1:]), _render(g[1:]), "record", "opencode")
            if stream_canon(g[1:]) == stream_canon(e[1:]):
                whitelisted.append({"kind": "stream_interleave", "detail": detail})
            else:
                kind = "numeric" if mask_numbers(g[1:]) == mask_numbers(e[1:]) else "messages"
                diffs.append({"kind": kind, "detail": detail})
    n = count_media(exp)
    exp_media = media_digests[:n]
    got_sha = [x.get("sha256") if isinstance(x, dict) else media_digest(x)["sha256"] for x in got_media]
    exp_sha = [x["sha256"] for x in exp_media]
    if got_sha != exp_sha:
        bad = len(got_sha) != len(exp_sha)
        items = []
        for i, (gs, es) in enumerate(zip(got_sha, exp_sha)):
            if gs == es:
                continue
            eq = media_eq(i, gs) if media_eq else None
            items.append({"index": i, "record": es[:16], "opencode": gs[:16], **(eq or {})})
            if not (eq and eq.get("equivalent")):
                bad = True
        if bad:
            diffs.append({"kind": "media", "detail": {"record": [x[:16] for x in exp_sha],
                                                      "opencode": [x[:16] for x in got_sha], "items": items}})
        else:
            whitelisted += [{"kind": "media_render", **it} for it in items]
    tools = record.get("tools")
    if isinstance(tools, str):
        tools = json.loads(tools)
    if tools is not None and body.get("tools") != tools:
        gn = [t["function"]["name"] for t in body.get("tools") or []]
        en = [t["function"]["name"] for t in tools]
        detail = {"record_names": en, "opencode_names": gn}
        if gn == en:
            detail["changed"] = [a["function"]["name"] for a, b in zip(tools, body["tools"]) if a != b]
        diffs.append({"kind": "tools", "detail": detail})
    if expected_params is not None:
        got_p = {k: v for k, v in body.items() if k not in ("messages", "tools")}
        if got_p != expected_params:
            diffs.append({"kind": "params", "detail": {"record": expected_params, "opencode": got_p}})
    return {"status": "diff" if diffs else "match", "diffs": diffs, "whitelisted": whitelisted}


# ---------------------------------------------------------------------------
# routing (shared by the mock and the offline report)
# ---------------------------------------------------------------------------

def first_user_text(messages: list[dict[str, Any]]) -> str | None:
    for m in messages:
        if m["role"] == "user":
            return wire_to_record([m])[0][0]["content"] if isinstance(m.get("content"), list) else m["content"]
    return None


def route_key(system: str, first_user: str | None) -> tuple[str, str]:
    """(first line of the system prompt, normalized first user message): the agent prompt's first
    line names the agent, the first user message (the task prompt, verbatim) names the session."""
    return (system.split("\n", 1)[0], norm_text(first_user or ""))


def build_scenario(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The mock's input: per record its route key and the replies for every step."""
    out = {}
    for key, rec in records.items():
        msgs = rec["messages"]
        assert msgs[0]["role"] == "system", f"{key}: first message must be system"
        sys_head, fu = route_key(msgs[0]["content"], first_user_text(msgs))
        out[key] = {"system_head": sys_head, "first_user": fu,
                    "steps": [step_reply(msgs, a, b) for a, b in record_steps(msgs)]}
    return {"records": out}


def _served(body: dict[str, Any]) -> list[dict[str, Any]]:
    """The assistant steps already in a request, as step_reply()-shaped dicts (canonical calls)."""
    out = []
    for m in body.get("messages") or []:
        if m["role"] != "assistant":
            continue
        c = m.get("content")
        out.append({"text": c or None,
                    "calls": [_canon_call(tc["function"]["name"], tc["function"].get("arguments", ""))
                              for tc in m.get("tool_calls") or []]})
    return out


def _step_matches(served: dict[str, Any], reply: dict[str, Any]) -> bool:
    calls = [_canon_call(n, a) for n, a in reply["calls"]]
    return served["calls"] == calls and (served["text"] or None) == (reply["text"] or None)


def route(scenario: dict[str, Any], body: dict[str, Any], claimed: set[str] | None = None
          ) -> tuple[str | None, str]:
    """(record key or None, kind) for a request: kind is "record", "record_fallback", "title" or "unrouted".

    Records with the same system head and first user message (e.g. two transcriber dispatches with the
    same prompt) are told apart by the assistant steps already in the request (what the mock served
    that session so far); a step-0 request takes the first such record not yet `claimed` (the mock adds
    every record it served a step 0 for)."""
    msgs = body.get("messages") or []
    system = "".join(m["content"] for m in msgs if m["role"] == "system" and isinstance(m["content"], str))
    if system.startswith(TITLE_SYSTEM_PREFIX):
        return None, "title"
    head, fu = route_key(system, first_user_text(msgs))
    cands = [k for k, r in scenario["records"].items() if r["system_head"] == head]
    exact = [k for k in cands if scenario["records"][k]["first_user"] == fu]
    if len(exact) > 1:
        served = _served(body)
        hist = [k for k in exact if len(scenario["records"][k]["steps"]) >= len(served) and
                all(_step_matches(s, r) for s, r in zip(served, scenario["records"][k]["steps"]))]
        if not served and claimed is not None:
            hist = [k for k in hist if k not in claimed] or hist
        exact = hist or exact
    if exact:
        return exact[0], "record"
    if len(cands) == 1:
        return cands[0], "record_fallback"
    return None, "unrouted"


def step_of(body: dict[str, Any]) -> int:
    return sum(1 for m in body.get("messages") or [] if m["role"] == "assistant")


LoadMedia = Callable[[str], dict[str, Any]]
