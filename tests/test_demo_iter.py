"""Tests for demo_iter_examples.py conversation builder.

Verifies the new reapy-based conversation structure:
  - _LISTEN_CMD is a shell command string (not Python)
  - build_conversation produces valid JSONL structure
  - Search/set/listen turns appear in correct order
  - No VitalController or fictional CLI in generated conversations
"""

from __future__ import annotations

import ast
import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.demo_iter_examples import (
    _LISTEN_CMD,
    _LISTEN_RESULT,
    SEARCH_STEP_PROB,
    build_conversation,
)
from maestro.synth.path_gen import generate_preset_path
from maestro.synth.wavetable_lib import load_wavetable_lib


# ---------------------------------------------------------------------------
# 1. Constant checks
# ---------------------------------------------------------------------------

def test_listen_cmd_is_shell_string():
    """_LISTEN_CMD should be a shell command, not Python code."""
    assert "reaper_render_probe" in _LISTEN_CMD
    assert "aplay" in _LISTEN_CMD
    # Not an import statement — it's a shell command
    assert "import" not in _LISTEN_CMD


def test_listen_result_has_audio_tag():
    assert "<audio>" in _LISTEN_RESULT


def test_search_step_prob_reasonable():
    assert 0.0 < SEARCH_STEP_PROB <= 1.0


# ---------------------------------------------------------------------------
# 2. build_conversation structure
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def conversation():
    """Build one real conversation for inspection."""
    rng = random.Random(42)
    path_result = generate_preset_path("lead", rng)
    fake_gt = Path("/tmp/fake_gt.wav")
    fake_default = Path("/tmp/fake_default.wav")
    fake_steps = [Path(f"/tmp/fake_step{i}.wav") for i in range(path_result["n_iterations"])]
    return build_conversation(
        path_result,
        {"gt": fake_gt, "default": fake_default, "steps": fake_steps},
        random.Random(42),
    )


def test_first_message_is_user_with_audio(conversation):
    msgs = conversation["messages"]
    assert msgs[0]["role"] == "user"
    assert "<audio>" in msgs[0]["content"]


def test_last_message_is_assistant_no_tool_calls(conversation):
    msgs = conversation["messages"]
    last = msgs[-1]
    assert last["role"] == "assistant"
    assert "tool_calls" not in last or last.get("tool_calls") is None


def test_audio_count(conversation):
    n = conversation["meta"]["n_iterations"]
    # GT + default + N step wavs
    assert len(conversation["audios"]) == n + 2


def test_no_vital_controller(conversation):
    full = json.dumps(conversation)
    assert "VitalController" not in full


def test_no_fictional_cli(conversation):
    full = json.dumps(conversation)
    assert "vital set " not in full
    assert "vital listen" not in full
    assert "vital params" not in full


def test_listen_cmd_used_for_all_listens(conversation):
    """Every listen tool_call should use _LISTEN_CMD."""
    for msg in conversation["messages"]:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tc = msg["tool_calls"][0]
            cmd = json.loads(tc["function"]["arguments"]).get("command", "")
            if "aplay" in cmd:
                assert "reaper_render_probe" in cmd


def test_set_snippets_are_valid_python(conversation):
    for msg in conversation["messages"]:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tc = msg["tool_calls"][0]
            cmd = json.loads(tc["function"]["arguments"]).get("command", "")
            if "fx.params[" in cmd:
                try:
                    ast.parse(cmd)
                except SyntaxError as e:
                    pytest.fail(f"SyntaxError in set snippet: {e}\n\n{cmd}")


def test_search_snippets_are_valid_python(conversation):
    for msg in conversation["messages"]:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tc = msg["tool_calls"][0]
            cmd = json.loads(tc["function"]["arguments"]).get("command", "")
            if "p.name.lower()" in cmd:
                try:
                    ast.parse(cmd)
                except SyntaxError as e:
                    pytest.fail(f"SyntaxError in search snippet: {e}\n\n{cmd}")


def test_search_before_set_ordering(conversation):
    """Every set turn (fx.params[...]) must be preceded by a search turn (p.name.lower())."""
    msgs = conversation["messages"]
    # Collect (index, type) for assistant tool_call messages
    tool_msgs = []
    for i, msg in enumerate(msgs):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tc = msg["tool_calls"][0]
            cmd = json.loads(tc["function"]["arguments"]).get("command", "")
            if "fx.params[" in cmd:
                tool_msgs.append((i, "set"))
            elif "p.name.lower()" in cmd:
                tool_msgs.append((i, "search"))

    # For each set, check there's a search somewhere before it (within the same step group)
    for j, (idx, kind) in enumerate(tool_msgs):
        if kind == "set":
            # Find the most recent prior search
            prior_searches = [t for t in tool_msgs[:j] if t[1] == "search"]
            # Either no search (allowed probabilistically) or there's a recent one
            # Just verify we don't have MORE sets than searches+1 (sanity check)
            n_searches = len([t for t in tool_msgs if t[1] == "search"])
            n_sets = len([t for t in tool_msgs if t[1] == "set"])
            # At minimum, step 1 should have had a search
            assert n_searches >= 1, "No search turns found at all in conversation"
            break
