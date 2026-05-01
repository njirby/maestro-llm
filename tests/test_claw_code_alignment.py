"""Validate generated SFT conversations against claw-code's actual tool interface.

Layer 1: Schema validation — _V3_TOOL_SPECS and generated tool_calls vs claw-code's real schemas.
Layer 2: Conversation flow — lifecycle patterns (Agent dispatch→cat, parallel batching, role order).

Runs against generated JSONL files. Skips gracefully if output files don't exist.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.build_main_agent_sft_v3 import _V3_TOOL_SPECS

# ---------------------------------------------------------------------------
# Claw-code reference schemas (source of truth from Claude Code system prompt)
# ---------------------------------------------------------------------------

CLAW_CODE_PARAMS = {
    "Bash": {
        "required": {"command"},
        "optional": {"description", "timeout", "run_in_background", "dangerouslyDisableSandbox"},
    },
    "Agent": {
        "required": {"description", "prompt"},
        "optional": {"subagent_type", "name", "run_in_background", "model", "isolation", "mode", "team_name"},
    },
    "Read": {
        "required": {"file_path"},
        "optional": {"offset", "limit", "pages"},
    },
    "Skill": {
        "required": {"skill"},
        "optional": {"args"},
    },
}

AGENT_RESPONSE_ALLOWED_FIELDS = {"status", "outputFile"}
AGENT_RESPONSE_FORBIDDEN_FIELDS = {"agentId", "manifestFile", "createdAt", "startedAt", "subagentType", "n_notes", "duration_s"}

BASH_RESPONSE_FIELDS = {"stdout", "stderr", "interrupted"}

# ---------------------------------------------------------------------------
# Output file discovery
# ---------------------------------------------------------------------------

_OUTPUT_DIRS = [
    Path(__file__).parent.parent / "outputs" / "smoke_v4",
    Path(__file__).parent.parent / "outputs" / "smoke_v3",
]

_OUTPUT_SUFFIXES = [
    "v65",
    "v61_no_slot_leak",
    "v60_grep_shortlist",
    "v59_specs_fix",
    "v58_agent_cleanup",
    "v57_text_output",
]


def _find_output_file(agent: str) -> Path | None:
    for _OUTPUT_DIR in _OUTPUT_DIRS:
        for suffix in _OUTPUT_SUFFIXES:
            for n in (8, 4):
                p = _OUTPUT_DIR / f"{agent}_final{n}_{suffix}.jsonl"
                if p.exists():
                    return p
    return None


_AGENTS = ["main", "search", "judge", "transcription"]
_AGENT_PATHS: dict[str, Path | None] = {a: _find_output_file(a) for a in _AGENTS}


def _load_records(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


_records_cache: dict[str, list[dict]] = {}


def _get_records(agent: str) -> list[dict]:
    if agent not in _records_cache:
        path = _AGENT_PATHS[agent]
        if path and path.exists():
            _records_cache[agent] = _load_records(path)
        else:
            _records_cache[agent] = []
    return _records_cache[agent]


def _parse_tool_specs() -> dict[str, dict]:
    specs = json.loads(_V3_TOOL_SPECS)
    return {s["function"]["name"]: s["function"] for s in specs}


_V3_SPECS_PARSED = _parse_tool_specs()


def _skip_if_no_records(agent: str):
    return pytest.mark.skipif(
        _AGENT_PATHS[agent] is None,
        reason=f"No {agent} output file found in {_OUTPUT_DIRS}",
    )


def _iter_tool_calls(records: list[dict], tool_name: str | None = None):
    for rec in records:
        for i, m in enumerate(rec["messages"]):
            if m["role"] == "tool_call":
                parsed = json.loads(m["content"])
                if tool_name is None or parsed["name"] == tool_name:
                    yield rec, i, parsed


def _iter_tool_call_response_pairs(records: list[dict], tool_name: str):
    for rec in records:
        msgs = rec["messages"]
        for i, m in enumerate(msgs):
            if m["role"] == "tool_call":
                parsed = json.loads(m["content"])
                if parsed["name"] == tool_name:
                    for j in range(i + 1, len(msgs)):
                        if msgs[j]["role"] == "tool_response":
                            yield rec, i, parsed, msgs[j]
                            break
                        elif msgs[j]["role"] not in ("tool_call",):
                            break


# ============================================================================
# Layer 1: _V3_TOOL_SPECS compatibility with claw-code
# ============================================================================


def test_tool_specs_names_are_pascal_case():
    for name in _V3_SPECS_PARSED:
        assert name[0].isupper(), f"Tool name '{name}' should be PascalCase"
    expected = {"Bash", "Agent", "Read", "Skill"}
    assert set(_V3_SPECS_PARSED.keys()) == expected


def test_tool_specs_bash_required_params():
    bash = _V3_SPECS_PARSED["Bash"]
    required = set(bash["parameters"].get("required", []))
    assert "command" in required
    extra = required - CLAW_CODE_PARAMS["Bash"]["required"]
    assert not extra, f"Bash requires params claw-code doesn't: {extra}"


def test_tool_specs_agent_required_params():
    agent = _V3_SPECS_PARSED["Agent"]
    required = set(agent["parameters"].get("required", []))
    claw_required = CLAW_CODE_PARAMS["Agent"]["required"]
    extra = required - claw_required
    assert not extra, f"Agent requires params claw-code doesn't: {extra}"


def test_tool_specs_agent_declares_run_in_background():
    agent = _V3_SPECS_PARSED["Agent"]
    props = set(agent["parameters"].get("properties", {}).keys())
    assert "run_in_background" in props, "Agent schema should declare run_in_background"


def test_tool_specs_agent_description_consistent():
    agent = _V3_SPECS_PARSED["Agent"]
    desc = agent.get("description", "").lower()
    assert "agentid" not in desc.replace(" ", ""), \
        "Agent description mentions agentId but we strip it from responses"


def test_tool_specs_read_required_params():
    read = _V3_SPECS_PARSED["Read"]
    required = set(read["parameters"].get("required", []))
    assert "file_path" in required
    extra = required - CLAW_CODE_PARAMS["Read"]["required"]
    assert not extra, f"Read requires params claw-code doesn't: {extra}"


def test_tool_specs_skill_required_params():
    skill = _V3_SPECS_PARSED["Skill"]
    required = set(skill["parameters"].get("required", []))
    assert "skill" in required
    extra = required - CLAW_CODE_PARAMS["Skill"]["required"]
    assert not extra, f"Skill requires params claw-code doesn't: {extra}"


def test_tool_specs_no_unknown_param_names():
    for tool_name, spec in _V3_SPECS_PARSED.items():
        declared_props = set(spec["parameters"].get("properties", {}).keys())
        claw_all = CLAW_CODE_PARAMS[tool_name]["required"] | CLAW_CODE_PARAMS[tool_name]["optional"]
        unknown = declared_props - claw_all
        assert not unknown, f"{tool_name} declares params claw-code doesn't have: {unknown}"


# ============================================================================
# Layer 1: Generated tool_call validation
# ============================================================================


@pytest.mark.parametrize("agent", _AGENTS)
def test_tool_call_names_match_declared_tools(agent):
    if not _AGENT_PATHS[agent]:
        pytest.skip(f"No {agent} output file")
    records = _get_records(agent)
    for rec in records:
        tools_raw = rec.get("tools", "[]")
        if isinstance(tools_raw, str):
            tools_list = json.loads(tools_raw)
        else:
            tools_list = tools_raw
        declared = {t["function"]["name"] for t in tools_list}
        for _rec, _i, parsed in _iter_tool_calls([rec]):
            assert parsed["name"] in declared, \
                f"{rec['id']}: tool_call '{parsed['name']}' not in declared tools {declared}"


@pytest.mark.parametrize("agent", _AGENTS)
def test_tool_call_names_are_pascal_case(agent):
    if not _AGENT_PATHS[agent]:
        pytest.skip(f"No {agent} output file")
    valid_names = {"Bash", "Agent", "Read", "Skill"}
    for rec, _i, parsed in _iter_tool_calls(_get_records(agent)):
        assert parsed["name"] in valid_names, \
            f"{rec['id']}: tool name '{parsed['name']}' not PascalCase"


@pytest.mark.parametrize("agent", _AGENTS)
def test_bash_tool_call_has_command(agent):
    if not _AGENT_PATHS[agent]:
        pytest.skip(f"No {agent} output file")
    for rec, _i, parsed in _iter_tool_calls(_get_records(agent), "Bash"):
        args = parsed["arguments"]
        assert "command" in args and isinstance(args["command"], str) and args["command"].strip(), \
            f"{rec['id']}: Bash call missing non-empty command"


@pytest.mark.parametrize("agent", _AGENTS)
def test_bash_tool_call_no_unknown_args(agent):
    if not _AGENT_PATHS[agent]:
        pytest.skip(f"No {agent} output file")
    allowed = CLAW_CODE_PARAMS["Bash"]["required"] | CLAW_CODE_PARAMS["Bash"]["optional"]
    for rec, _i, parsed in _iter_tool_calls(_get_records(agent), "Bash"):
        unknown = set(parsed["arguments"].keys()) - allowed
        assert not unknown, f"{rec['id']}: Bash call has unknown args: {unknown}"


@_skip_if_no_records("main")
def test_agent_tool_call_has_required_args():
    for rec, _i, parsed in _iter_tool_calls(_get_records("main"), "Agent"):
        args = parsed["arguments"]
        assert "description" in args and isinstance(args["description"], str) and args["description"].strip(), \
            f"{rec['id']}: Agent call missing description"
        assert "prompt" in args and isinstance(args["prompt"], str) and args["prompt"].strip(), \
            f"{rec['id']}: Agent call missing prompt"


@_skip_if_no_records("main")
def test_agent_tool_call_has_run_in_background():
    for rec, _i, parsed in _iter_tool_calls(_get_records("main"), "Agent"):
        args = parsed["arguments"]
        assert args.get("run_in_background") is True, \
            f"{rec['id']}: Agent call missing run_in_background=true"


@_skip_if_no_records("main")
def test_agent_tool_call_no_unknown_args():
    allowed = CLAW_CODE_PARAMS["Agent"]["required"] | CLAW_CODE_PARAMS["Agent"]["optional"]
    for rec, _i, parsed in _iter_tool_calls(_get_records("main"), "Agent"):
        unknown = set(parsed["arguments"].keys()) - allowed
        assert not unknown, f"{rec['id']}: Agent call has unknown args: {unknown}"


@pytest.mark.parametrize("agent", _AGENTS)
def test_read_tool_call_has_file_path(agent):
    if not _AGENT_PATHS[agent]:
        pytest.skip(f"No {agent} output file")
    found_any = False
    for rec, _i, parsed in _iter_tool_calls(_get_records(agent), "Read"):
        found_any = True
        args = parsed["arguments"]
        assert "file_path" in args and isinstance(args["file_path"], str) and args["file_path"].strip(), \
            f"{rec['id']}: Read call missing file_path"
    if not found_any:
        pytest.skip(f"No Read tool_calls in {agent} records")


@_skip_if_no_records("main")
def test_skill_tool_call_has_skill():
    found_any = False
    for rec, _i, parsed in _iter_tool_calls(_get_records("main"), "Skill"):
        found_any = True
        args = parsed["arguments"]
        assert "skill" in args and isinstance(args["skill"], str) and args["skill"].strip(), \
            f"{rec['id']}: Skill call missing skill name"
    if not found_any:
        pytest.skip("No Skill tool_calls in main records")


# ============================================================================
# Layer 1: Generated tool_response validation
# ============================================================================


@pytest.mark.parametrize("agent", _AGENTS)
def test_bash_response_is_stdout_stderr_interrupted(agent):
    if not _AGENT_PATHS[agent]:
        pytest.skip(f"No {agent} output file")
    checked = 0
    for rec, _tc_i, tc_parsed, resp_msg in _iter_tool_call_response_pairs(_get_records(agent), "Bash"):
        try:
            resp = json.loads(resp_msg["content"])
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(resp, dict):
            continue
        if "stdout" in resp:
            checked += 1
            assert isinstance(resp.get("stdout"), str), f"{rec['id']}: stdout not string"
            assert isinstance(resp.get("stderr"), str), f"{rec['id']}: stderr not string"
            assert isinstance(resp.get("interrupted"), bool), f"{rec['id']}: interrupted not bool"
    assert checked > 0 or agent == "transcription", f"No Bash {'{stdout,stderr,interrupted}'} responses found in {agent}"


@_skip_if_no_records("main")
def test_agent_response_has_status_and_output_file():
    checked = 0
    for rec, _tc_i, tc_parsed, resp_msg in _iter_tool_call_response_pairs(_get_records("main"), "Agent"):
        resp = json.loads(resp_msg["content"])
        checked += 1
        assert "status" in resp and isinstance(resp["status"], str), \
            f"{rec['id']}: Agent response missing status"
        assert "outputFile" in resp and isinstance(resp["outputFile"], str), \
            f"{rec['id']}: Agent response missing outputFile"
    assert checked > 0, "No Agent tool_responses found in main records"


@_skip_if_no_records("main")
def test_agent_response_no_forbidden_fields():
    for rec, _tc_i, tc_parsed, resp_msg in _iter_tool_call_response_pairs(_get_records("main"), "Agent"):
        resp = json.loads(resp_msg["content"])
        for field in AGENT_RESPONSE_FORBIDDEN_FIELDS:
            assert field not in resp, \
                f"{rec['id']}: Agent response has forbidden field '{field}'"


# ============================================================================
# Layer 2: Conversation flow validation
# ============================================================================


@pytest.mark.parametrize("agent", _AGENTS)
def test_first_message_is_user(agent):
    if not _AGENT_PATHS[agent]:
        pytest.skip(f"No {agent} output file")
    for rec in _get_records(agent):
        assert rec["messages"][0]["role"] == "user", \
            f"{rec['id']}: first message is {rec['messages'][0]['role']}, not user"


@pytest.mark.parametrize("agent", _AGENTS)
def test_last_message_is_assistant(agent):
    if not _AGENT_PATHS[agent]:
        pytest.skip(f"No {agent} output file")
    for rec in _get_records(agent):
        assert rec["messages"][-1]["role"] == "assistant", \
            f"{rec['id']}: last message is {rec['messages'][-1]['role']}, not assistant"


@pytest.mark.parametrize("agent", _AGENTS)
def test_no_adjacent_same_role_except_tool(agent):
    if not _AGENT_PATHS[agent]:
        pytest.skip(f"No {agent} output file")
    for rec in _get_records(agent):
        prev_role = None
        for i, m in enumerate(rec["messages"]):
            role = m["role"]
            if prev_role == role and role not in ("tool_call", "tool_response"):
                pytest.fail(f"{rec['id']} msg {i}: adjacent {role}-{role}")
            prev_role = role


@pytest.mark.parametrize("agent", _AGENTS)
def test_parallel_tool_batch_counts_match(agent):
    if not _AGENT_PATHS[agent]:
        pytest.skip(f"No {agent} output file")
    for rec in _get_records(agent):
        msgs = rec["messages"]
        i = 0
        while i < len(msgs):
            if msgs[i]["role"] == "tool_call":
                tc_count = 0
                while i < len(msgs) and msgs[i]["role"] == "tool_call":
                    tc_count += 1
                    i += 1
                tr_count = 0
                while i < len(msgs) and msgs[i]["role"] == "tool_response":
                    tr_count += 1
                    i += 1
                assert tc_count == tr_count, \
                    f"{rec['id']}: {tc_count} tool_calls but {tr_count} tool_responses in batch"
            else:
                i += 1


@_skip_if_no_records("main")
def test_agent_dispatch_has_cat_followup():
    """Search and judge agent outputFiles should be cat'd. Transcription agents
    are consumed inline (the main agent proceeds with the verified result)."""
    for rec in _get_records("main"):
        msgs = rec["messages"]
        agent_output_files: list[tuple[str, str]] = []  # (outputFile, subagent_type)
        for i, m in enumerate(msgs):
            if m["role"] == "tool_call":
                parsed = json.loads(m["content"])
                if parsed["name"] == "Agent":
                    subtype = parsed["arguments"].get("subagent_type", "")
                    for j in range(i + 1, len(msgs)):
                        if msgs[j]["role"] == "tool_response":
                            try:
                                resp = json.loads(msgs[j]["content"])
                                if "outputFile" in resp:
                                    agent_output_files.append((resp["outputFile"], subtype))
                            except (json.JSONDecodeError, TypeError):
                                pass
                            break
                        elif msgs[j]["role"] not in ("tool_call",):
                            break

        all_bash_cmds = []
        for m in msgs:
            if m["role"] == "tool_call":
                parsed = json.loads(m["content"])
                if parsed["name"] == "Bash":
                    all_bash_cmds.append(parsed["arguments"].get("command", ""))

        for out_file, subtype in agent_output_files:
            if subtype == "melody_transcription":
                continue
            found_read = any(out_file in cmd for cmd in all_bash_cmds)
            assert found_read, \
                f"{rec['id']}: {subtype} outputFile '{out_file}' never read (cat/grep)"


@_skip_if_no_records("main")
def test_skill_call_appears_before_heavy_work():
    for rec in _get_records("main"):
        skill_idx = None
        for i, m in enumerate(rec["messages"]):
            if m["role"] == "tool_call":
                parsed = json.loads(m["content"])
                if parsed["name"] == "Skill":
                    skill_idx = i
                    break
        if skill_idx is not None:
            assert skill_idx < 10, \
                f"{rec['id']}: Skill call at index {skill_idx}, expected within first 10 messages"
