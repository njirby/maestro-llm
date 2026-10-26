"""v5 record assembly against the OpenCode goldens (docs/opencode_capture/goldens/with_plugin)."""
import json
import re

import pytest

from maestro.v5.agents.main import skill_response
from maestro.v5.record import DATE, Record, final_text
from scripts import opencode_contract as oc

G = oc.GOLDENS_DIR / "with_plugin"
AGENTS = ("maestro", "transcriber", "searcher", "judge")


def golden_conv():
    return json.loads((G / "conversation_maestro.json").read_text())


def wire_steps():
    """[(tool calls [(name, args)], tool contents, following user message or None)] of the golden conversation."""
    msgs, out, i = golden_conv(), [], 0
    while i < len(msgs):
        m = msgs[i]
        i += 1
        if m["role"] != "assistant" or not m.get("tool_calls"):
            continue
        calls = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in m["tool_calls"]]
        tools = []
        while i < len(msgs) and msgs[i]["role"] == "tool":
            tools.append(msgs[i]["content"])
            i += 1
        user = msgs[i] if i < len(msgs) and msgs[i]["role"] == "user" else None
        out.append((calls, tools, user))
    return out


@pytest.mark.parametrize("agent", AGENTS)
def test_system_message_matches_golden(agent):
    rec = Record(agent, "x", "{WORKDIR}", "hi", date="{DATE}")
    want = (G / "system" / f"{agent}.txt").read_text().replace("{CONFIG_DIR}", oc.CONFIG_DIR)
    assert rec.messages[0] == {"role": "system", "content": want}
    assert rec.to_dict()["tools"] == json.dumps(oc.tools_for(agent), ensure_ascii=False)


def test_fixed_date():
    assert DATE == oc.date_string(__import__("datetime").date(2026, 10, 25))
    assert f"Today's date: {DATE}" in Record("maestro", "x", "/w", "hi").messages[0]["content"]


def test_tool_call_formats_match_wire():
    """Our tool_call JSON carries exactly the arguments OpenCode put on the wire (the capture's mock added a bash
    `description`, which is not in the v1.19.3 bash schema)."""
    makers = {
        "bash": lambda a: oc.bash_call(a["command"], timeout_ms=a.get("timeout"), workdir=a.get("workdir")),
        "read": lambda a: oc.read_call(a["filePath"], a.get("offset"), a.get("limit")),
        "skill": lambda a: oc.skill_call(a["name"]),
        "task": lambda a: oc.task_call(a["description"], a["prompt"], a["subagent_type"]),
        "write": lambda a: oc.write_call(a["filePath"], a["content"]),
    }
    seen = set()
    for calls, _, _ in wire_steps():
        for name, args in calls:
            if name not in makers or not args:
                continue
            args = {k: v for k, v in args.items() if not (name == "bash" and k == "description")}
            assert json.loads(makers[name](args)) == {"name": name, "arguments": args}
            seen.add(name)
    assert seen == set(makers)


def test_audio_steps_match_wire():
    """Single and parallel audio reads: N tool responses, then ONE user attachment with N audios in call order."""
    steps = [(c, t, u) for c, t, u in wire_steps()
             if c and all(n == "read" and a.get("filePath", "").endswith(".wav") for n, a in c)]
    assert [len(c) for c, _, _ in steps] == [1, 2]
    for calls, tools, user in steps:
        paths = [a["filePath"] for _, a in calls]
        rec = Record("maestro", "x", "/w", "hi")
        rec.audio_step(paths, [f"/host/{i}.wav" for i in range(len(paths))])
        tail = rec.messages[2:]
        assert [m["role"] for m in tail] == ["tool_call"] * len(paths) + ["tool_response"] * len(paths) + ["user"]
        assert [m["content"] for m in tail[len(paths):2 * len(paths)]] == tools
        parts = user["content"]
        assert parts[0] == {"type": "text", "text": oc.MEDIA_ATTACHED_TEXT}
        assert [p["type"] for p in parts[1:]] == ["input_audio"] * len(paths)
        assert tail[-1]["content"] == oc.MEDIA_ATTACHED_TEXT + "<audio>" * len(paths)
        assert rec.audios == [f"/host/{i}.wav" for i in range(len(paths))]


def test_step_text_and_finish():
    rec = Record("searcher", "x", "/w", "prompt")
    rec.say("first")
    rec.step([oc.bash_call("true")], ["(no output)"])
    rec.say("a")
    rec.say("b")
    d = rec.finish("done")
    assert [m["role"] for m in d["messages"]] == ["system", "user", "assistant", "tool_call", "tool_response",
                                                 "assistant"]
    assert d["messages"][2]["content"] == "first"
    assert final_text(d) == "a\n\nb\n\ndone"


def test_task_roundtrip_matches_golden():
    rt = json.loads((G / "task_roundtrip.json").read_text())
    by = {c["subagent_type"]: c for c in rt}
    wd = "{WORKDIR}"
    assert oc.transcription_dispatch_prompt(f"{wd}/target.wav", f"{wd}/notes.json") == by["transcriber"]["prompt"]
    assert oc.judge_dispatch_prompt(f"{wd}/target.wav", ["Basic Shapes", "Harmonic Series"],
                                    notes_path=f"{wd}/notes.json") == by["judge"]["prompt"]
    assert oc.search_dispatch_prompt(f"{wd}/target.wav", f"{wd}/notes.json", 0, 48) == \
        f"Target: {wd}/target.wav.\nNotes: {wd}/notes.json.\nSlice: wavetables wt001-wt048."
    for c in rt:
        sub = Record(c["subagent_type"], "s", wd, c["prompt"]).finish(c["subagent_final_text"])
        assert sub["messages"][1] == c["subagent_first_user"]
        assert oc.task_result("{SES_ID}", final_text(sub)) == c["parent_result"]


def test_skill_response_matches_golden():
    case = next(c for c in json.loads((G / "tool_results.json").read_text()) if c["case"] == "skill_vital")
    assert skill_response() == case["output"].replace("{CONFIG_DIR}", oc.CONFIG_DIR)
    assert re.findall(r"Base directory for this skill: (.*)", skill_response()) == [f"{oc.CONFIG_DIR}/skills/vital"]
