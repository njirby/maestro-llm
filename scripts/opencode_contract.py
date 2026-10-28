#!/usr/bin/env python
"""The OpenCode tool contract — single source of truth, byte-faithful to v1.19.3.

Every rollout builder AND the serving harness express tool calls, tool
results, errors, the system message and subagent returns through these
helpers, so synthetic training rollouts match what the real OpenCode harness
puts on the wire.

Ground truth is the real binary, not this file: docs/opencode_capture/goldens/
holds requests captured from `opencode run` against a scripted mock server
(capture/capture.sh + capture/extract.py), and tests/test_opencode_contract.py
checks every helper here against them byte for byte. When OpenCode is
upgraded, re-capture and fix whatever the tests flag.

What OpenCode does (verified by capture):
  - tool schemas: the stock set, loaded verbatim from goldens/tools.json
    (primary agents) and goldens/tools_subagent.json (subagents get no task
    or todowrite)
  - bash returns stdout+stderr as one stream; the exit code is metadata and
    never reaches the model. Output past 2000 lines / 51200 bytes keeps the
    tail behind a "...output truncated..." header naming the spill file; a
    timeout appends a <shell_metadata> footer
  - read wraps text in <path>/<type>/<content> with `N: line` numbering and an
    end-of-file / continue footer; directories list <entries>; errors are the
    bare error message
  - skill returns a <skill_content> block: body, base directory, sampled files
  - task returns <task id="ses_..." state="completed"><task_result>...; the
    subagent's first user message is the task prompt verbatim
  - system message = agent prompt, model line, <env> block, instruction files,
    skills block, joined by "\\n"
  - audio is not readable by stock OpenCode (`Cannot read binary file`); the
    maestro read plugin makes it so: the tool result is AUDIO_READ_TEXT and the
    audio rides in a user message after the step's tool results, exactly like
    stock image reads (see media_attachment_message / audio_read_step)
In ms-swift records audio is the <audio> placeholder inside that user message
(positionally aligned with the `audios` list), never inside a tool_response.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO = Path(__file__).resolve().parent.parent
GOLDENS_DIR = REPO / "docs" / "opencode_capture" / "goldens"
OPENCODE_DIR = REPO / "opencode"          # repo-owned config, staged per opencode/README.md
OPENCODE_VERSION = "1.19.3"

# Provider/model as configured in opencode/opencode.json ("maestro/<served name>").
DEFAULT_PROVIDER = "maestro"
DEFAULT_MODEL = "maestro"
# Model-visible paths fixed by the launcher recipe (opencode/README.md,
# MAESTRO_OC_ROOT=/tmp/maestro-oc): CONFIG_DIR is the staged OPENCODE_CONFIG_DIR
# (skill <location> and base directory), DATA_DIR is $XDG_DATA_HOME/opencode
# (bash spill files in DATA_DIR/tool-output).
CONFIG_DIR = "/tmp/maestro-oc/config"
DATA_DIR = "/tmp/maestro-oc/data/opencode"

# Limits (OpenCode defaults; `tool_output` config is unset).
BASH_MAX_LINES = 2000
BASH_MAX_BYTES = 51200
BASH_DEFAULT_TIMEOUT_MS = 120000
READ_DEFAULT_LIMIT = 2000
READ_MAX_LINE_CHARS = 2000
READ_MAX_BYTES = 51200
SKILL_FILES_LIMIT = 10

# ---------------------------------------------------------------------------
# Tool schemas (exactly what OpenCode sends in `tools=`; also the records'
# `tools` field so the training template renders the same descriptions)
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = json.loads((GOLDENS_DIR / "tools.json").read_text())
SUBAGENT_TOOLS: list[dict[str, Any]] = json.loads((GOLDENS_DIR / "tools_subagent.json").read_text())
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}


def tools_for(agent: str) -> list[dict[str, Any]]:
    """Tool list OpenCode advertises to `agent` (by its frontmatter mode)."""
    return SUBAGENT_TOOLS if agent_mode(agent) == "subagent" else TOOLS


# ---------------------------------------------------------------------------
# Ids
# ---------------------------------------------------------------------------

_BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def opencode_id(prefix: str, descending: bool = False, time_ms: int | None = None,
                seed: str | None = None, counter: int = 1) -> str:
    """OpenCode's Identifier: prefix_ + 6 time bytes (hex) + 14 base62 chars.

    Session ids are descending (`ses_`), tool-output ids ascending (`tool_`).
    `seed` makes the random tail (and, without `time_ms`, the time) deterministic.
    """
    if time_ms is None:
        time_ms = (int(hashlib.sha1(seed.encode()).hexdigest()[:8], 16) + 1_700_000_000_000
                   if seed is not None else int(time.time() * 1000))
    now = time_ms * 0x1000 + counter
    if descending:
        now = ~now
    head = bytes((now >> (40 - 8 * i)) & 0xFF for i in range(6)).hex()
    if seed is None:
        tail = "".join(secrets.choice(_BASE62) for _ in range(14))
    else:
        digest = hashlib.sha256(seed.encode()).digest()
        tail = "".join(_BASE62[b % 62] for b in digest[:14])
    return f"{prefix}_{head}{tail}"


def session_id(seed: str | None = None, time_ms: int | None = None) -> str:
    return opencode_id("ses", descending=True, time_ms=time_ms, seed=seed)


def tool_output_path(seed: str | None = None, data_dir: str = DATA_DIR) -> str:
    return f"{data_dir}/tool-output/{opencode_id('tool', time_ms=None, seed=seed)}"


# ---------------------------------------------------------------------------
# Tool calls (ms-swift role=tool_call content)
# ---------------------------------------------------------------------------

def tool_call(name: str, arguments: dict[str, Any]) -> str:
    """Content string for a role=tool_call message."""
    assert name in TOOL_NAMES or name == "invalid", f"unknown tool {name!r}"
    return json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)


def _drop_none(**kw: Any) -> dict[str, Any]:
    return {k: v for k, v in kw.items() if v is not None}


def bash_call(command: str, timeout_ms: int | None = None,
              workdir: str | None = None) -> str:
    args: dict[str, Any] = {"command": command}
    if timeout_ms is not None:
        args["timeout"] = int(timeout_ms)
    if workdir is not None:
        args["workdir"] = workdir
    return tool_call("bash", args)


def read_call(file_path: str, offset: int | None = None, limit: int | None = None) -> str:
    return tool_call("read", _drop_none(filePath=file_path, offset=offset, limit=limit))


def skill_call(name: str) -> str:
    return tool_call("skill", {"name": name})


def task_call(description: str, prompt: str, subagent_type: str) -> str:
    return tool_call("task", {"description": description, "prompt": prompt,
                              "subagent_type": subagent_type})


def write_call(file_path: str, content: str) -> str:
    return tool_call("write", {"filePath": file_path, "content": content})


def edit_call(file_path: str, old: str, new: str, replace_all: bool | None = None) -> str:
    return tool_call("edit", _drop_none(filePath=file_path, oldString=old, newString=new,
                                        replaceAll=replace_all))


def glob_call(pattern: str, path: str | None = None) -> str:
    return tool_call("glob", _drop_none(pattern=pattern, path=path))


def grep_call(pattern: str, path: str | None = None, include: str | None = None) -> str:
    return tool_call("grep", _drop_none(pattern=pattern, path=path, include=include))


def todowrite_call(todos: list[dict[str, str]]) -> str:
    return tool_call("todowrite", {"todos": todos})


# ---------------------------------------------------------------------------
# bash
# ---------------------------------------------------------------------------

def _tail_truncate(text: str, max_lines: int, max_bytes: int) -> tuple[str, bool]:
    """OpenCode's bash truncation: keep the last lines that fit both limits."""
    lines = text.split("\n")
    if len(lines) <= max_lines and len(text.encode()) <= max_bytes:
        return text, False
    kept: list[str] = []
    size = 0
    for line in reversed(lines):
        if len(kept) >= max_lines:
            break
        n = len(line.encode()) + (1 if kept else 0)
        if size + n > max_bytes:
            if not kept:  # a single over-long last line: keep its tail bytes
                raw = line.encode()
                start = max(len(raw) - max_bytes, 0)
                while start < len(raw) and (raw[start] & 0xC0) == 0x80:
                    start += 1
                kept.insert(0, raw[start:].decode())
            break
        kept.insert(0, line)
        size += n
    return "\n".join(kept), True


def bash_output(stdout: str = "", exit_code: int = 0, stderr: str = "", *,
                timeout_ms: int | None = None, aborted: bool = False,
                output_path: str | None = None,
                max_lines: int = BASH_MAX_LINES, max_bytes: int = BASH_MAX_BYTES) -> str:
    """bash tool result as the model sees it.

    OpenCode captures stdout and stderr as one stream (pass the interleaved
    text as `stdout` when you have it; otherwise stderr is appended). The exit
    code is metadata only and is accepted here for call compatibility.
    `timeout_ms` marks a command killed at that timeout. `output_path` is the
    spill file named in the truncation header (generated when needed).
    """
    del exit_code
    raw = stdout + stderr
    text, cut = _tail_truncate(raw, max_lines, max_bytes)
    out = text or "(no output)"
    if cut:
        out = (f"...output truncated...\n\nFull output saved to: "
               f"{output_path or tool_output_path()}\n\n{out}")
    notes = []
    if timeout_ms is not None:
        notes.append(f"shell tool terminated command after exceeding timeout {timeout_ms} ms. "
                     "If this command is expected to take longer and is not waiting for "
                     "interactive input, retry with a larger timeout value in milliseconds.")
    if aborted:
        notes.append("User aborted the command")
    if notes:
        out += "\n\n<shell_metadata>\n" + "\n".join(notes) + "\n</shell_metadata>"
    return out


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------

def _split_lines(content: str) -> list[str]:
    """Effect Stream.splitLines: \\n, \\r\\n or \\r; no empty line after a final break."""
    if content == "":
        return []
    lines = re.split(r"\r\n|\r|\n", content)
    if lines[-1] == "":
        lines.pop()
    return lines


def _utf16_len(s: str) -> int:
    return len(s.encode("utf-16-le")) // 2


def _utf16_prefix(s: str, units: int) -> str:
    return s.encode("utf-16-le")[: 2 * units].decode("utf-16-le", errors="ignore")


def read_output_text(file_path: str, content: str, offset: int | None = None,
                     limit: int | None = None) -> str:
    """read result for a text file (or the out-of-range error string)."""
    start = offset or 1
    limit = READ_DEFAULT_LIMIT if limit is None else limit
    kept: list[str] = []
    size = count = 0
    more = cut = False
    for line in _split_lines(content):
        count += 1
        if count < start:
            continue
        if len(kept) >= limit:
            more = True
            continue
        if _utf16_len(line) > READ_MAX_LINE_CHARS:
            line = (_utf16_prefix(line, READ_MAX_LINE_CHARS)
                    + f"... (line truncated to {READ_MAX_LINE_CHARS} chars)")
        n = len(line.encode()) + (1 if kept else 0)
        if size + n > READ_MAX_BYTES:
            cut = more = True
            break
        kept.append(line)
        size += n
    if count < start and not (count == 0 and start == 1):
        return read_offset_error(start, count)
    last = start + len(kept) - 1
    out = f"<path>{file_path}</path>\n<type>file</type>\n<content>\n"
    out += "\n".join(f"{i + start}: {w}" for i, w in enumerate(kept))
    if cut:
        out += (f"\n\n(Output capped at {READ_MAX_BYTES // 1024} KB. Showing lines "
                f"{start}-{last}. Use offset={last + 1} to continue.)")
    elif more:
        out += f"\n\n(Showing lines {start}-{last} of {count}. Use offset={last + 1} to continue.)"
    else:
        out += f"\n\n(End of file - total {count} lines)"
    return out + "\n</content>"


def _locale_key(name: str) -> tuple:
    """Approximates JS String.localeCompare (ICU root): punctuation < digits <
    letters, case-insensitive first, lowercase before uppercase on ties."""
    def cls(c: str) -> int:
        return 2 if c.isalpha() else 1 if c.isdigit() else 0
    return (tuple((cls(c), c.casefold()) for c in name), name.swapcase())


def read_output_directory(dir_path: str, entries: Iterable[str], offset: int | None = None,
                          limit: int | None = None) -> str:
    """read result for a directory. `entries` are names, subdirectories with a
    trailing "/"; they are sorted the way OpenCode sorts them."""
    names = sorted(entries, key=_locale_key)
    start = offset or 1
    limit = READ_DEFAULT_LIMIT if limit is None else limit
    page = names[start - 1:start - 1 + limit]
    if start - 1 + len(page) < len(names):
        footer = (f"\n(Showing {len(page)} of {len(names)} entries. Use 'offset' parameter "
                  f"to read beyond entry {start + len(page)})")
    else:
        footer = f"\n({len(names)} entries)"
    return "\n".join([f"<path>{dir_path}</path>", "<type>directory</type>", "<entries>",
                      "\n".join(page), footer, "</entries>"])


def read_not_found(file_path: str, suggestions: Iterable[str] = ()) -> str:
    """Missing path. OpenCode suggests up to 3 siblings whose name contains (or
    is contained in) the requested basename, case-insensitively."""
    sugg = list(suggestions)[:3]
    if sugg:
        return f"File not found: {file_path}\n\nDid you mean one of these?\n" + "\n".join(sugg)
    return f"File not found: {file_path}"


def read_binary_error(file_path: str) -> str:
    return f"Cannot read binary file: {file_path}"


def read_offset_error(offset: int, count: int) -> str:
    return f"Offset {offset} is out of range for this file ({count} lines)"


IMAGE_READ_TEXT = "Image read successfully"
PDF_READ_TEXT = "PDF read successfully"
# Listening = the maestro read plugin (opencode/plugin/audio-read.ts): a read
# of an audio file returns this text, like the stock image read.
AUDIO_READ_TEXT = "Audio read successfully"

# Media from read results never sits in the tool message. After ALL tool
# messages of the step, OpenCode appends one user message holding the media
# of every media read of that step, in call order (images: stock; audio: the
# plugin, same shape). Captured wire form (goldens/with_plugin/media_messages.json):
#   {"role": "user", "content": [{"type": "text", "text": MEDIA_ATTACHED_TEXT},
#                                {"type": "input_audio", ...}, ...]}
# In ms-swift records that message's content is MEDIA_ATTACHED_TEXT followed
# directly by one placeholder per file, no separator ("...result:<audio><audio>").
# Reason: the Qwen2.5-Omni chat template renders list content part by part
# with nothing in between (text, then <|audio_bos|><|AUDIO|><|audio_eos|>),
# and ms-swift expands a placeholder in place, so both give the same tokens.
# This holds when vLLM passes list content to the template (its "openai"
# content format, auto-detected for the Omni templates); vLLM's "string"
# format would instead put the placeholders first, joined by "\n".
# The record's `audios` list must follow the same call order.
MEDIA_ATTACHED_TEXT = "Attached media from tool result:"
AUDIO_PLACEHOLDER = "<audio>"
IMAGE_PLACEHOLDER = "<image>"
_PLACEHOLDER = {"audio": AUDIO_PLACEHOLDER, "image": IMAGE_PLACEHOLDER}


def read_output_audio(file_path: str, *_legacy: Any) -> str:
    """tool_response content of an audio read (extra legacy args are ignored).
    The audio itself goes in media_attachment_message(), never in here."""
    return AUDIO_READ_TEXT


def media_attachment_message(kinds: Iterable[str]) -> dict[str, str]:
    """The user message that follows a step's tool responses when the step
    read media. `kinds` lists "audio"/"image" per media read, in call order.
    Steps mixing audio and image reads are not supported (the plugin and
    OpenCode hoist them into separate messages; never captured)."""
    kinds = list(kinds)
    assert kinds and len(set(kinds)) == 1, "one media kind per step"
    return {"role": "user",
            "content": MEDIA_ATTACHED_TEXT + "".join(_PLACEHOLDER[k] for k in kinds)}


def audio_read_step(paths: Iterable[str]) -> tuple[list[dict[str, str]], dict[str, str]]:
    """One assistant step of N parallel audio reads: the N tool_response
    messages (one per call, in call order) and the user message that follows
    them. Append the clips to the record's `audios` in the same order."""
    paths = list(paths)
    responses = [{"role": "tool_response", "content": read_output_audio(p)} for p in paths]
    return responses, media_attachment_message(["audio"] * len(paths))


# ---------------------------------------------------------------------------
# skill
# ---------------------------------------------------------------------------

def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    m = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n?(.*)$", text, re.S)
    if not m:
        return {}, text
    return (yaml.safe_load(m.group(1)) or {}), m.group(2)


def sample_skill_files(base_dir: str, limit: int = SKILL_FILES_LIMIT) -> list[str]:
    """Up to `limit` files under the skill dir, SKILL.md excluded. OpenCode's
    sample is in ripgrep traversal order (not stable); this one is sorted."""
    out: list[str] = []
    for root, dirs, files in os.walk(base_dir):
        dirs.sort()
        for fn in sorted(files):
            if fn != "SKILL.md":
                out.append(os.path.join(root, fn))
    return out[:limit]


def skill_output(name: str, path: str, content: str, description: str = "",
                 files: Iterable[str] | None = None) -> str:
    """skill tool result. `path` is the skill's SKILL.md (or its directory) as
    the serving process sees it; `content` is the SKILL.md text (frontmatter
    is dropped). `files` defaults to sampling the directory if it exists
    locally. `description` is unused (kept for call compatibility)."""
    del description
    base = os.path.dirname(path) if os.path.basename(path) == "SKILL.md" else path
    body = _split_frontmatter(content)[1].strip()
    if files is None:
        files = sample_skill_files(base) if os.path.isdir(base) else []
    return "\n".join([f'<skill_content name="{name}">', f"# Skill: {name}", "", body, "",
                      f"Base directory for this skill: {base}",
                      "Relative paths in this skill (e.g., scripts/, reference/) are relative "
                      "to this base directory.",
                      "Note: file list is sampled.", "", "<skill_files>",
                      "\n".join(f"<file>{f}</file>" for f in files),
                      "</skill_files>", "</skill_content>"])


def skill_not_found(name: str, available: Iterable[str]) -> str:
    """Unknown skill. `available` is every loaded skill, permission-denied ones
    included (OpenCode lists the built-in customize-opencode here)."""
    return f'Skill "{name}" not found. Available skills: {", ".join(sorted(available))}'


def discover_skills(skills_dir: str | Path = OPENCODE_DIR / "skills",
                    location_root: str | None = None) -> list[dict[str, str]]:
    """Skills as OpenCode lists them: name, description (YAML-parsed) and the
    SKILL.md location, rebased onto `location_root` (the serving process's
    view of `skills_dir`) when given."""
    skills_dir = Path(skills_dir)
    out = []
    for md in sorted(skills_dir.glob("**/SKILL.md")):
        meta, _ = _split_frontmatter(md.read_text())
        if not meta.get("name") or meta.get("description") is None:
            continue
        loc = str(md) if location_root is None else f"{location_root}/{md.relative_to(skills_dir)}"
        out.append({"name": str(meta["name"]), "description": str(meta["description"]),
                    "location": loc})
    return out


def skills_block(skills: list[dict[str, str]]) -> str:
    head = ("Skills provide specialized instructions and workflows for specific tasks.\n"
            "Use the skill tool to load a skill when a task matches its description.\n")
    if not skills:
        return head + "No skills are currently available."
    rows = []
    for s in sorted(skills, key=lambda s: _locale_key(s["name"])):
        loc = s["location"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        rows += ["  <skill>", f"    <name>{s['name']}</name>",
                 f"    <description>{s['description']}</description>",
                 f"    <location>{loc}</location>", "  </skill>"]
    return head + "\n".join(["<available_skills>", *rows, "</available_skills>"])


# ---------------------------------------------------------------------------
# task
# ---------------------------------------------------------------------------

def task_result(session_id: str, text: str, state: str = "completed",
                summary: str | None = None) -> str:
    """Inline subagent return: the tool_response content of a task call.
    `session_id` is the subagent's `ses_...` id (see session_id())."""
    tag = "task_error" if state == "error" else "task_result"
    lines = [f'<task id="{session_id}" state="{state}">']
    if summary:
        lines.append(f"<summary>{summary}</summary>")
    lines += [f"<{tag}>", text, f"</{tag}>", "</task>"]
    return "\n".join(lines)


def task_unknown_agent(subagent_type: str) -> str:
    return f"Unknown agent type: {subagent_type} is not a valid agent type"


# ---------------------------------------------------------------------------
# write / edit / glob / grep / todowrite / errors
# ---------------------------------------------------------------------------

WRITE_OK = "Wrote file successfully."
EDIT_OK = "Edit applied successfully."
EDIT_NOT_FOUND = ("Could not find oldString in the file. It must match exactly, including "
                  "whitespace, indentation, and line endings.")
EDIT_MULTIPLE = ("Found multiple matches for oldString. Provide more surrounding context to "
                 "make the match unique.")
NO_FILES_FOUND = "No files found"


def glob_output(paths: list[str], truncated: bool = False) -> str:
    """Absolute paths in OpenCode's order (newest first); at most 100."""
    if not paths:
        return NO_FILES_FOUND
    lines = list(paths)
    if truncated:
        lines += ["", "(Results are truncated: showing first 100 results. Consider using a "
                      "more specific path or pattern.)"]
    return "\n".join(lines)


def grep_output(matches: list[tuple[str, int, str]], truncated: bool = False) -> str:
    """matches: (absolute path, line number, line text). The text carries its
    line break as ripgrep reports it; one is added when missing."""
    if not matches:
        return NO_FILES_FOUND
    out = [f"Found {len(matches)} matches{' (more matches available)' if truncated else ''}"]
    current = ""
    for path, line, text in matches:
        if path != current:
            if current:
                out.append("")
            current = path
            out.append(f"{path}:")
        out.append(f"  Line {line}: {text if text.endswith(chr(10)) else text + chr(10)}")
    if truncated:
        out += ["", "(Results truncated. Consider using a more specific path or pattern.)"]
    return "\n".join(out)


def todowrite_output(todos: list[dict[str, str]]) -> str:
    return json.dumps(todos, indent=2, ensure_ascii=False)


def invalid_arguments_error(tool: str, detail: str) -> str:
    """Schema-invalid arguments. `detail` is the Effect schema message, e.g.
    missing_key_detail("filePath")."""
    return (f"The {tool} tool was called with invalid arguments: {detail}.\n"
            "Please rewrite the input so it satisfies the expected schema.")


def missing_key_detail(key: str) -> str:
    return f'SchemaError(Missing key\n  at ["{key}"])'


def unavailable_tool(name: str, tools: Iterable[str] | None = None) -> tuple[str, str]:
    """A call to a tool that isn't offered. OpenCode rewrites the assistant's
    call in history into a call of the hidden `invalid` tool; returns the
    (tool_call content, tool_response content) pair."""
    avail = ", ".join(sorted(set(tools or TOOL_NAMES) | {"invalid"}))
    error = f"Model tried to call unavailable tool '{name}'. Available tools: {avail}."
    return (tool_call("invalid", {"tool": name, "error": error}),
            f"The arguments provided to the tool are invalid: {error}")


# ---------------------------------------------------------------------------
# Agents and the system message
# ---------------------------------------------------------------------------

AGENT_DIR = OPENCODE_DIR / "agent"
# The main agent is OpenCode's default primary agent `build`, unchanged: its prompt is OpenCode's stock prompt
# (captured verbatim into goldens/stock_prompt.txt). Our workflow reaches it as instructions (INSTRUCTIONS_FILE).
# "maestro" stays the main agent's name in this module and in the records' bookkeeping.
MAIN_AGENT = "maestro"
STOCK_PROMPT = (GOLDENS_DIR / "stock_prompt.txt").read_text()
# Config-dir AGENTS.md: OpenCode appends it to EVERY agent's system prompt (main and subagents) as
# "Instructions from: <OPENCODE_CONFIG_DIR>/AGENTS.md\n<text>".
INSTRUCTIONS_FILE = OPENCODE_DIR / "AGENTS.md"
# Skills OpenCode ships built in (listed for every agent, location "<built-in>").
BUILTIN_SKILLS = [{
    "name": "customize-opencode",
    "description": ("Use ONLY when the user is editing or creating opencode's own configuration: opencode.json, "
                    "opencode.jsonc, files under .opencode/, or files under ~/.config/opencode/. Also use when "
                    "creating or fixing opencode agents, subagents, skills, plugins, MCP servers, or permission "
                    "rules. Do not use for the user's own application code, or for any project that is not "
                    "configuring opencode itself."),
    "location": "<built-in>"}]
# Pre-OpenCode role names used by the v3/v4 builders.
LEGACY_AGENT_NAMES = {"main": "maestro", "melody_transcription": "transcriber",
                      "wavetable_search": "searcher", "wavetable_judge": "judge", "build": "maestro"}


def _agent_file(agent: str) -> tuple[dict[str, Any], str]:
    if agent == MAIN_AGENT:
        return {"mode": "primary"}, STOCK_PROMPT
    meta, body = _split_frontmatter((AGENT_DIR / f"{agent}.md").read_text())
    return meta, body.strip()


def agent_mode(agent: str) -> str:
    agent = LEGACY_AGENT_NAMES.get(agent, agent)
    return _agent_file(agent)[0].get("mode", "all")


def agent_prompt(agent: str) -> str:
    """The agent's prompt as OpenCode loads it (markdown body, trimmed)."""
    return _agent_file(LEGACY_AGENT_NAMES.get(agent, agent))[1]


AGENTS = sorted(p.stem for p in AGENT_DIR.glob("*.md"))      # our agent files (the subagents)
AGENT_PROMPTS = {a: agent_prompt(a) for a in AGENTS + [MAIN_AGENT]}
AGENT_PROMPTS.update({old: AGENT_PROMPTS[new] for old, new in LEGACY_AGENT_NAMES.items()
                      if new in AGENT_PROMPTS})


def date_string(date: _dt.date | str | None = None) -> str:
    """JS Date.toDateString(), e.g. 'Sun Oct 25 2026'."""
    if isinstance(date, str):
        return date
    d = date or _dt.date.today()
    return d.strftime("%a %b %d %Y")


def env_block(cwd: str, date: _dt.date | str | None = None, model: str = DEFAULT_MODEL,
              provider: str = DEFAULT_PROVIDER, worktree: str = "/", git: bool = False) -> str:
    return "\n".join([
        f"You are powered by the model named {model}. The exact model ID is {provider}/{model}",
        "Here is some useful information about the environment you are running in:",
        "<env>",
        f"  Working directory: {cwd}",
        f"  Workspace root folder: {worktree}",
        f"  Is directory a git repo: {'yes' if git else 'no'}",
        "  Platform: linux",
        f"  Today's date: {date_string(date)}",
        "</env>",
    ])


def system_message(agent: str, cwd: str, date: _dt.date | str | None = None,
                   model: str = DEFAULT_MODEL, provider: str = DEFAULT_PROVIDER,
                   skills: list[dict[str, str]] | None | bool = None, *,
                   config_dir: str | None = None, worktree: str = "/", git: bool = False,
                   instructions: Iterable[tuple[str, str]] | None = None, user_system: str | None = None,
                   include_tools: bool = False) -> str:
    """The system message OpenCode sends for `agent`.

    `agent` is an agent name (opencode/agent/<name>.md, legacy role names
    accepted) or literal prompt text. Parts, joined by "\\n": prompt, model
    line + <env> block, `instructions` ((path, text) pairs -> "Instructions
    from: ..."), skills block, per-message `user_system`.

    `skills=None` lists opencode/skills with locations under
    `config_dir`/skills (config_dir = the serving process's
    OPENCODE_CONFIG_DIR, default CONFIG_DIR); pass a list from
    discover_skills() to override, or False to omit the block (skill tool
    denied). OpenCode's built-in skills (BUILTIN_SKILLS) are always listed
    with them. `instructions=None` = the config-dir AGENTS.md (every agent
    gets it); pass () for none. `date` defaults to today, so pass it for
    reproducible records.

    `include_tools=True` appends the legacy hermes tools block (tools_block()).
    """
    if agent in AGENTS or agent in LEGACY_AGENT_NAMES or agent == MAIN_AGENT:
        prompt = agent_prompt(agent)
    else:
        prompt = agent
    root = str(config_dir or CONFIG_DIR)
    if skills is None:
        skills = discover_skills(OPENCODE_DIR / "skills", location_root=f"{root}/skills")
    if skills is not False:
        skills = list(skills) + [s for s in BUILTIN_SKILLS if s["name"] not in {x["name"] for x in skills}]
    if instructions is None:
        instructions = ([(f"{root}/AGENTS.md", INSTRUCTIONS_FILE.read_text())]
                        if INSTRUCTIONS_FILE.exists() else [])
    parts = [prompt, env_block(cwd, date, model, provider, worktree, git)]
    parts += [f"Instructions from: {p}\n{text}" for p, text in instructions]
    if skills is not False:
        parts.append(skills_block(skills))
    if user_system:
        parts.append(user_system)
    out = "\n".join(p for p in parts if p)
    return f"{out}\n\n{tools_block()}" if include_tools else out


def tools_block(tools: list[dict[str, Any]] | None = None) -> str:
    """The hermes tool-schema block ms-swift's hermes agent template appends
    to the system message at TRAINING time (legacy serving path: inline it so
    the prompt matches training regardless of the engine's chat template)."""
    ts = TOOLS if tools is None else tools
    sigs = "\n".join(json.dumps(t, ensure_ascii=False) for t in ts)
    return (
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        f"<tools>\n{sigs}\n</tools>\n\n"
        "For each function call, return a json object with function name and "
        "arguments within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
        "</tool_call>"
    )


# ---------------------------------------------------------------------------
# Subagent dispatch prompts — the SAME string is the parent's task prompt and
# the subagent's first user message (OpenCode passes it through verbatim).
# Divergence here is the D1 contract bug; never compose these ad hoc.
# ---------------------------------------------------------------------------

def transcription_dispatch_prompt(target_audio_path, notes_path) -> str:
    """Parent -> transcriber. The agent prompt (agent/transcriber.md) carries the method: write notes.json in the
    vital skill's format, render through the default patch, listen next to the target, fix and re-render."""
    return ("Transcribe the melody in the target clip — I'll use the notes to audition wavetables, so the pitches "
            "and the rhythm have to be right. Write them to notes.json in the format the `vital` skill documents.\n\n"
            f"Target: {target_audio_path}\n"
            f"Write the melody to {notes_path}.")


def wavetable_slice_label(start: int, end: int) -> str:
    """Library slice [start, end) (0-based) as `wavetable.py list` ids, e.g. 0, 166 -> "wt001-wt166"."""
    return f"wt{start + 1:03d}-wt{end:03d}"


def search_dispatch_prompt(target_audio_path, notes_path, start: int, end: int) -> str:
    """Parent -> searcher: the target, the session notes and a chunk [start, end) of the wavetable library.
    The searcher audits EVERY table in the chunk and replies with the best 2-3 configurations (tables in
    oscillator order) for the target in that chunk."""
    return ("Audition wavetables for the target clip — I'm recreating its sound and need the oscillator wavetables. "
            "Audition every table in your chunk: render each one with the session notes and listen to them in "
            "batches of ~8, saying what each batch is or isn't close to. Then combine the closest ones into pairs "
            "— both oscillator orders, since osc 1 leads — render and listen to those in batches too. Reply with "
            "your best 2-3 configurations for the target in this chunk: tables in oscillator order, each with a "
            "few words on what it explains.\n\n"
            f"Target: {target_audio_path}\n"
            f"Notes: {notes_path}.\n"
            f"Chunk: wavetables {wavetable_slice_label(start, end)}.")


def judge_dispatch_prompt(target_audio_path, pool: list[list[str]], locked_section: str = "",
                          notes_path=None) -> str:
    """Parent -> judge: the target, the session notes and the pooled configurations (each a list of table names
    in oscillator order). The reply format ({"tuple": [...], "reasoning": "..."}) is in agent/judge.md.
    `locked_section` is appended verbatim."""
    pool_str = ", ".join(json.dumps(cfg, ensure_ascii=False) for cfg in pool)
    notes = f"Notes: {notes_path}.\n" if notes_path else ""
    return ("Pick the oscillator wavetables for the target clip. The pool is the best configurations each search "
            "chunk found — a configuration is a list of table names in oscillator order. Render the strongest "
            "configurations, including recombos (one table from one configuration, another from another), and "
            "listen against the target; reply with JSON: "
            "{\"tuple\": [names in oscillator order], \"reasoning\": \"one or two sentences\"}.\n\n"
            f"Target: {target_audio_path}\n{notes}Pool: [{pool_str}].{locked_section}")
