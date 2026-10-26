"""scripts/opencode_contract.py vs the goldens captured from real OpenCode.

Goldens: docs/opencode_capture/goldens/ (re-capture: capture/capture.sh, then
capture/extract.py). Placeholders such as {WORKDIR} stand for volatile values;
the helpers are fed the same placeholders, so comparisons are exact.
"""

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scripts import opencode_contract as oc  # noqa: E402

G = REPO / "docs" / "opencode_capture" / "goldens"
CASES = {c["case"]: c for c in json.loads((G / "tool_results.json").read_text())}
CONV = json.loads((G / "conversation_maestro.json").read_text())
SUBAGENTS = ("transcriber", "searcher", "judge")


def case(name):
    c = CASES[name]
    return c, json.loads(c["wire_call"]["arguments"])


def skills_snapshot():
    return oc.discover_skills(G / "skills_snapshot", location_root="{CONFIG_DIR}/skills")


# -- schemas, request shape --------------------------------------------------

def test_tools_are_the_captured_stock_set():
    assert oc.TOOLS == json.loads((G / "tools.json").read_text())
    names = [t["function"]["name"] for t in oc.TOOLS]
    assert names == ["bash", "edit", "glob", "grep", "read", "skill", "task", "todowrite",
                     "webfetch", "write"]
    assert [t["function"]["name"] for t in oc.tools_for("searcher")] == [
        n for n in names if n not in ("task", "todowrite")]
    assert oc.tools_for("maestro") is oc.TOOLS


def test_request_params():
    assert json.loads((G / "request_params.json").read_text()) == {
        "model": oc.DEFAULT_MODEL, "max_tokens": 4096, "tool_choice": "auto", "stream": True,
        "stream_options": {"include_usage": True}}


# -- system messages ---------------------------------------------------------

def test_agent_files_match_capture_snapshot():
    """Agent prompts changed since the goldens were captured -> re-capture."""
    for md in (G / "agent_snapshot").glob("*.md"):
        assert (oc.AGENT_DIR / md.name).read_text() == md.read_text(), md.name
    assert sorted(p.stem for p in (G / "agent_snapshot").glob("*.md")) == oc.AGENTS


def test_skill_files_match_capture_snapshot():
    """opencode/skills changed since the goldens were captured -> re-capture."""
    snap = sorted(p.relative_to(G / "skills_snapshot") for p in (G / "skills_snapshot").glob("**/SKILL.md"))
    live = sorted(p.relative_to(oc.OPENCODE_DIR / "skills")
                  for p in (oc.OPENCODE_DIR / "skills").glob("**/SKILL.md"))
    assert snap == live
    for rel in snap:
        assert (oc.OPENCODE_DIR / "skills" / rel).read_text() == (G / "skills_snapshot" / rel).read_text()


@pytest.mark.parametrize("agent", ("maestro",) + SUBAGENTS)
def test_system_message(agent):
    got = oc.system_message(agent, "{WORKDIR}", date="{DATE}", skills=skills_snapshot())
    assert got == (G / "system" / f"{agent}.txt").read_text()


def test_system_message_legacy_role_names_and_literal_prompt():
    kw = dict(cwd="/w", date="Sun Oct 25 2026", skills=[])
    assert oc.system_message("wavetable_search", **kw) == oc.system_message("searcher", **kw)
    assert oc.AGENT_PROMPTS["main"] == oc.agent_prompt("maestro")
    lit = oc.system_message("Custom prompt.", **kw)
    assert lit.startswith("Custom prompt.\nYou are powered by the model named maestro.")
    assert lit.endswith("No skills are currently available.")
    assert oc.date_string(__import__("datetime").date(2026, 10, 25)) == "Sun Oct 25 2026"


def test_conversation_opens_with_system_then_raw_user_text():
    assert CONV[0] == {"role": "user", "content": "Recreate the sound in target.wav."}


# -- bash -----------------------------------------------------------------------

def test_bash_plain_and_failing():
    assert oc.bash_output("hello\n") == case("bash_ok")[0]["output"]
    # exit code 3 never reaches the model; stdout and stderr are one stream
    assert oc.bash_output("out\n", exit_code=3, stderr="err\n") == case("bash_fail_stderr")[0]["output"]
    assert oc.bash_output("") == case("bash_no_output")[0]["output"]
    assert oc.bash_output("{WORKDIR}/sub\n") == case("bash_workdir")[0]["output"]


def test_bash_timeout_footer():
    c, args = case("bash_timeout")
    assert oc.bash_output("start\n", timeout_ms=args["timeout"]) == c["output"]


SAVED = "{DATA_DIR}/tool-output/{TOOL_OUTPUT_ID}"


def test_bash_truncation_keeps_tail_lines():
    c, _ = case("bash_truncated_lines")
    raw = "".join(f"{i}\n" for i in range(1, 3001))
    assert len(raw) == c["inputs"]["saved_output_bytes"]
    assert oc.bash_output(raw, output_path=SAVED) == c["output"]


def test_bash_truncation_drops_overlong_line():
    c, _ = case("bash_truncated_bytes")
    raw = "x" * 60000 + "\n"
    assert len(raw) == c["inputs"]["saved_output_bytes"]
    assert oc.bash_output(raw, output_path=SAVED) == c["output"]


def test_bash_call_schema():
    assert json.loads(oc.bash_call("ls", 1000, "/w")) == {
        "name": "bash", "arguments": {"command": "ls", "timeout": 1000, "workdir": "/w"}}


# -- read -----------------------------------------------------------------------

@pytest.mark.parametrize("name", ["read_offset_limit", "read_full", "read_short",
                                  "read_default_limit", "read_long_line", "read_empty",
                                  "read_offset_out_of_range", "read_byte_cap"])
def test_read_text(name):
    c, args = case(name)
    got = oc.read_output_text(args["filePath"], c["inputs"]["content"],
                              offset=args.get("offset"), limit=args.get("limit"))
    assert got == c["output"]


def test_read_directory():
    c, args = case("read_directory")
    assert oc.read_output_directory(args["filePath"], reversed(c["inputs"]["entries"])) == c["output"]


def test_read_errors():
    c, args = case("read_missing")
    assert oc.read_not_found(args["filePath"]) == c["output"]
    c, args = case("read_missing_suggest")
    assert oc.read_not_found(args["filePath"], ["{WORKDIR}/notes.txt"]) == c["output"]
    c, args = case("read_binary")
    assert oc.read_binary_error(args["filePath"]) == c["output"]
    assert case("read_image")[0]["output"] == oc.IMAGE_READ_TEXT
    c, _ = case("read_invalid_args")
    assert oc.invalid_arguments_error("read", oc.missing_key_detail("filePath")) == c["output"]


def test_image_attachment_is_hoisted_into_a_user_message():
    """Media attachments leave the tool message: a synthetic user message
    follows the step's tool results (unsupported modality -> ERROR text)."""
    i = next(i for i, m in enumerate(CONV) if m.get("content") == oc.IMAGE_READ_TEXT)
    assert CONV[i + 1]["role"] == "user"
    assert CONV[i + 1]["content"][0] == {"type": "text", "text": "Attached media from tool result:"}


def test_audio_read_text():
    assert oc.read_output_audio("/w/a.wav") == oc.AUDIO_READ_TEXT
    assert oc.read_output_audio("a.wav", 1.5, 44100) == oc.AUDIO_READ_TEXT  # legacy call shape
    assert not hasattr(oc, "audio_read_response")  # no <audio> inside a tool response


# -- media reads with the audio-read plugin (goldens/with_plugin/) -----------------

GP = G / "with_plugin"
MEDIA = {m["case"]: m for m in json.loads((GP / "media_messages.json").read_text())}
MEDIA_IMG = {m["case"]: m for m in json.loads(
    (G / "with_plugin_image_modality" / "media_messages.json").read_text())}


def qwen_render(parts):
    """How the Qwen2.5-Omni chat template renders list content, with ms-swift
    placeholders for the media tokens: parts back to back, no separator."""
    ph = {"input_audio": "<audio>", "image_url": "<image>"}
    return "".join(p["text"] if p["type"] == "text" else ph[p["type"]] for p in parts)


@pytest.mark.parametrize("name,files", [("read_audio", ["target.wav"]),
                                        ("read_audio_parallel", ["target.wav", "second.wav"])])
def test_audio_read_step(name, files):
    m = MEDIA[name]
    calls = m["assistant"]["tool_calls"]
    paths = [json.loads(c["function"]["arguments"])["filePath"] for c in calls]
    assert paths == [f"{{WORKDIR}}/{f}" for f in files]
    responses, user = oc.audio_read_step(paths)
    # one tool message per call, in call order, carrying only the text
    assert [t["tool_call_id"] for t in m["tool_messages"]] == [c["id"] for c in calls]
    assert [t["content"] for t in m["tool_messages"]] == [r["content"] for r in responses]
    assert all(r["role"] == "tool_response" for r in responses)
    # then ONE user message with every clip, in call order
    parts = m["user"]["content"]
    assert [p["input_audio"] for p in parts[1:]] == [
        {"data": "{B64:" + f + "}", "format": "wav"} for f in files]
    assert user == {"role": "user", "content": qwen_render(parts)}
    assert user["content"] == "Attached media from tool result:" + "<audio>" * len(files)


def test_image_reads_use_the_same_message():
    for name, n in (("read_image", 1), ("read_image_parallel", 2)):
        m = MEDIA_IMG[name]
        assert [t["content"] for t in m["tool_messages"]] == [oc.IMAGE_READ_TEXT] * n
        assert [p["type"] for p in m["user"]["content"]] == ["text"] + ["image_url"] * n
        assert oc.media_attachment_message(["image"] * n) == {
            "role": "user", "content": qwen_render(m["user"]["content"])}
    # audio placement doesn't depend on whether the model takes images
    assert MEDIA_IMG["read_audio_parallel"]["user"] == MEDIA["read_audio_parallel"]["user"]


def test_plugin_changes_only_audio_reads():
    for f in ["tools.json", "tools_subagent.json", "request_params.json", "task_roundtrip.json",
              *(f"system/{a}.txt" for a in ("maestro",) + SUBAGENTS)]:
        assert (GP / f).read_text() == (G / f).read_text(), f
    plugin = {c["case"]: c for c in json.loads((GP / "tool_results.json").read_text())}
    assert plugin.keys() == CASES.keys()
    def strip(s):  # order-dependent parts: skill file sample, glob (mtime) order
        return sorted(re.sub(r"<skill_files>.*</skill_files>", "", s, flags=re.S).split("\n"))
    changed = {k for k in CASES if strip(CASES[k]["output"]) != strip(plugin[k]["output"])}
    assert changed == {"read_audio", "read_audio_parallel_0", "read_audio_parallel_1"}
    for k in changed:
        assert CASES[k]["output"] == oc.read_binary_error(
            json.loads(CASES[k]["wire_call"]["arguments"])["filePath"])
        assert plugin[k]["output"] == oc.AUDIO_READ_TEXT


# -- skill ------------------------------------------------------------------------

def test_skill_output():
    c, _ = case("skill_vital")
    files = re.findall(r"<file>(.*?)</file>", c["output"])
    got = oc.skill_output("vital", "{CONFIG_DIR}/skills/vital/SKILL.md",
                          c["inputs"]["skill_md"], files=files)
    assert got == c["output"]


def test_skill_not_found():
    c, args = case("skill_missing")
    names = [s["name"] for s in skills_snapshot()] + ["customize-opencode"]
    assert oc.skill_not_found(args["name"], names) == c["output"]


# -- task -------------------------------------------------------------------------

def test_task_roundtrip():
    rt = json.loads((G / "task_roundtrip.json").read_text())
    assert [r["subagent_type"] for r in rt] == list(SUBAGENTS)
    for r in rt:
        # the subagent's first user message is the dispatch prompt verbatim
        assert r["subagent_first_user"] == {"role": "user", "content": r["prompt"]}
        assert oc.task_result("{SES_ID}", r["subagent_final_text"]) == r["parent_result"]
    c, args = case("task_unknown_agent")
    assert oc.task_unknown_agent(args["subagent_type"]) == c["output"]


def test_parallel_task_calls_return_in_call_order():
    i = next(i for i, m in enumerate(CONV) if len(m.get("tool_calls") or []) == 3)
    ids = [tc["id"] for tc in CONV[i]["tool_calls"]]
    assert [m["tool_call_id"] for m in CONV[i + 1:i + 4]] == ids


def test_session_ids():
    sid = oc.session_id("seed")
    assert re.fullmatch(r"ses_[0-9a-f]{12}[0-9A-Za-z]{14}", sid)
    assert sid == oc.session_id("seed") != oc.session_id("other")
    assert re.fullmatch(r"ses_[0-9a-f]{12}[0-9A-Za-z]{14}", oc.session_id())
    # descending ids sort newest first
    assert oc.session_id(time_ms=2_000) < oc.session_id(time_ms=1_000)


# -- the rest of the stock tools ---------------------------------------------------

def test_write_edit():
    assert case("write_new")[0]["output"] == oc.WRITE_OK
    assert case("write_existing")[0]["output"] == oc.WRITE_OK
    assert case("edit_ok")[0]["output"] == oc.EDIT_OK
    assert case("edit_multiple")[0]["output"] == oc.EDIT_MULTIPLE
    assert case("edit_not_found")[0]["output"] == oc.EDIT_NOT_FOUND


def test_glob_grep():
    c, _ = case("glob_ok")
    assert oc.glob_output(c["output"].split("\n")) == c["output"]
    assert oc.glob_output([]) == case("glob_none")[0]["output"]
    c, _ = case("grep_ok")
    matches = [("{WORKDIR}/notes.txt", n, f"line {n}") for n in [1] + list(range(10, 20))]
    assert oc.grep_output(matches) == c["output"]
    assert oc.grep_output([]) == case("grep_none")[0]["output"]


def test_todowrite():
    c, args = case("todowrite")
    assert oc.todowrite_output(args["todos"]) == c["output"]


def test_unavailable_tool_is_rewritten_to_invalid():
    c, _ = case("unavailable_tool")
    call, out = oc.unavailable_tool("nosuchtool")
    assert out == c["output"]
    assert json.loads(call) == {"name": "invalid", "arguments": json.loads(c["wire_call"]["arguments"])}
