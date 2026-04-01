"""Tests for the reapy-based training data pipeline.

Covers:
- path_gen.py action_snippet format (real Python, not fictional CLI)
- Conversation structural invariants (agentic tool_call format)
- No fictional vital set/listen/params CLI anywhere in generated data
"""

from __future__ import annotations

import ast
import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from maestro.synth.path_gen import generate_preset_path, _denormalize, PARAM_RANGES

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ARCHETYPES = ["bass", "pad", "sequence"]


@pytest.fixture(scope="module")
def paths():
    """Generate one path per archetype with a fixed seed."""
    results = {}
    for archetype in _ARCHETYPES:
        rng = random.Random(42)
        results[archetype] = generate_preset_path(archetype, rng)
    return results


# ---------------------------------------------------------------------------
# 1. action_snippet field exists (not python_script)
# ---------------------------------------------------------------------------

def test_action_snippet_field_present(paths):
    for archetype, path_data in paths.items():
        for i, it in enumerate(path_data["iterations"]):
            assert "action_snippet" in it, (
                f"{archetype} step {i+1}: missing action_snippet"
            )
            assert "python_script" not in it, (
                f"{archetype} step {i+1}: stale python_script field should not exist"
            )


# ---------------------------------------------------------------------------
# 2. Every action_snippet is valid Python
# ---------------------------------------------------------------------------

def test_action_snippet_valid_python(paths):
    for archetype, path_data in paths.items():
        for i, it in enumerate(path_data["iterations"]):
            snippet = it["action_snippet"]
            try:
                ast.parse(snippet)
            except SyntaxError as exc:
                pytest.fail(
                    f"{archetype} step {i+1}: action_snippet has syntax error: {exc}\n\n{snippet}"
                )


# ---------------------------------------------------------------------------
# 3. action_snippet uses real reapy high-level API pattern
# ---------------------------------------------------------------------------

def test_action_snippet_uses_reapy(paths):
    required = ["reapy", "reapy.Project", "fx.params["]
    for archetype, path_data in paths.items():
        for i, it in enumerate(path_data["iterations"]):
            snippet = it["action_snippet"]
            n_params = len(it["params_applied"])
            if n_params == 0:
                # Empty step: snippet won't have fx.params[ lines but must still have reapy
                assert "reapy" in snippet, f"{archetype} step {i+1}: 'reapy' missing from empty-param snippet"
                continue
            for token in required:
                assert token in snippet, (
                    f"{archetype} step {i+1}: '{token}' not in action_snippet"
                )


def test_action_snippet_uses_inside_reaper_context(paths):
    for archetype, path_data in paths.items():
        for i, it in enumerate(path_data["iterations"]):
            assert "inside_reaper" in it["action_snippet"], (
                f"{archetype} step {i+1}: missing reapy.inside_reaper() context manager"
            )


# ---------------------------------------------------------------------------
# 4. No fictional CLI anywhere in snippets
# ---------------------------------------------------------------------------

def test_no_fictional_vital_set_cli(paths):
    for archetype, path_data in paths.items():
        for i, it in enumerate(path_data["iterations"]):
            snippet = it["action_snippet"]
            assert not snippet.startswith("vital set"), (
                f"{archetype} step {i+1}: action_snippet starts with fictional 'vital set'"
            )
            assert "\nvital set " not in snippet, (
                f"{archetype} step {i+1}: fictional 'vital set' found in action_snippet"
            )


def test_no_fictional_vital_mod_cli(paths):
    for archetype, path_data in paths.items():
        for i, it in enumerate(path_data["iterations"]):
            snippet = it["action_snippet"]
            assert "vital mod " not in snippet, (
                f"{archetype} step {i+1}: fictional 'vital mod' found in action_snippet"
            )


# ---------------------------------------------------------------------------
# 5. Modulation routes are tracked but not in action_snippet (VST3 limitation)
# ---------------------------------------------------------------------------

def test_last_step_modulations_tracked_in_data(paths):
    """modulations_changed is populated on the last step but NOT in action_snippet.
    VST3 chunk write is unreliable; modulation routing is deferred."""
    for archetype, path_data in paths.items():
        last = path_data["iterations"][-1]
        snippet = last["action_snippet"]
        # Modulations are tracked in the data dict for future use
        assert "modulations_changed" in last
        # But NOT written into the snippet (chunk write doesn't work reliably for VST3)
        assert "set_modulation" not in snippet, (
            f"{archetype}: set_modulation found in action_snippet — "
            "chunk write approach was re-added but shouldn't be"
        )


# ---------------------------------------------------------------------------
# 6. Param values in snippet are normalized [0, 1] (reapy fx.params.value range)
# ---------------------------------------------------------------------------

def test_action_snippet_contains_normalized_values(paths):
    """
    Values in action_snippet must be normalized [0, 1] — reapy's fx.params[].value
    takes normalized values, not native Vital units.
    """
    import re
    for archetype, path_data in paths.items():
        for it in path_data["iterations"]:
            snippet = it["action_snippet"]
            for match in re.finditer(r'\.value\s*=\s*([0-9.]+)', snippet):
                val = float(match.group(1))
                assert 0.0 <= val <= 1.0, (
                    f"{archetype}: value {val} out of [0,1] in snippet — "
                    "should be normalized, not native Vital units"
                )


# ---------------------------------------------------------------------------
# 7. Agentic tool_call conversation structure invariants
# ---------------------------------------------------------------------------

def _build_minimal_conversation(path_data: dict) -> dict:
    """Build a minimal agentic conversation from a path, without rendering audio."""
    iters = path_data["iterations"]
    n = path_data["n_iterations"]
    messages = []
    audios = ["/tmp/gt.wav", "/tmp/default.wav"]  # GT + default

    messages.append({
        "role": "user",
        "content": "<audio>\nRecreat this sound in Vital using the terminal.",
    })

    # Default listen
    messages.append({
        "role": "assistant",
        "content": "Let me hear the default.",
        "tool_calls": [{"id": "tc_default", "type": "function",
                        "function": {"name": "bash",
                                     "arguments": json.dumps({"command": "...default listen..."})}}],
    })
    messages.append({"role": "tool", "tool_call_id": "tc_default",
                     "content": "<audio>\nRendered default."})

    for it in iters:
        step = it["step"]
        snippet = it["action_snippet"]
        set_id = f"tc_set_{step}"
        listen_id = f"tc_listen_{step}"

        messages.append({
            "role": "assistant",
            "content": "reasoning...",
            "tool_calls": [{"id": set_id, "type": "function",
                            "function": {"name": "bash",
                                         "arguments": json.dumps({"command": snippet})}}],
        })
        messages.append({"role": "tool", "tool_call_id": set_id,
                         "content": f"Set {len(it['params_applied'])} params"})
        messages.append({
            "role": "assistant",
            "content": "Let me listen.",
            "tool_calls": [{"id": listen_id, "type": "function",
                            "function": {"name": "bash",
                                         "arguments": json.dumps({"command": "...listen..."})}}],
        })
        messages.append({"role": "tool", "tool_call_id": listen_id,
                         "content": f"<audio>\nRendered step {step}."})
        audios.append(f"/tmp/step{step}.wav")

    messages.append({"role": "assistant", "content": "Recreation complete."})
    return {"messages": messages, "audios": audios, "n_iterations": n}


def test_conversation_first_message_is_user_with_audio(paths):
    for archetype, path_data in paths.items():
        conv = _build_minimal_conversation(path_data)
        first = conv["messages"][0]
        assert first["role"] == "user"
        assert "<audio>" in first["content"]


def test_conversation_last_message_is_assistant_no_tool_call(paths):
    for archetype, path_data in paths.items():
        conv = _build_minimal_conversation(path_data)
        last = conv["messages"][-1]
        assert last["role"] == "assistant"
        assert "tool_calls" not in last


def test_conversation_tool_call_ids_match_tool_results(paths):
    """Every tool result must reference an id from the preceding assistant tool_call."""
    for archetype, path_data in paths.items():
        conv = _build_minimal_conversation(path_data)
        msgs = conv["messages"]
        pending_ids: set[str] = set()
        for m in msgs:
            if m["role"] == "assistant" and "tool_calls" in m:
                for tc in m["tool_calls"]:
                    pending_ids.add(tc["id"])
            elif m["role"] == "tool":
                tc_id = m.get("tool_call_id", "")
                assert tc_id in pending_ids, (
                    f"{archetype}: tool result references unknown id '{tc_id}'"
                )
                pending_ids.discard(tc_id)


def test_conversation_audio_count(paths):
    """audios[] = 1 GT + 1 default + N step clips."""
    for archetype, path_data in paths.items():
        conv = _build_minimal_conversation(path_data)
        n = path_data["n_iterations"]
        assert len(conv["audios"]) == n + 2, (
            f"{archetype}: expected {n+2} audios (GT+default+{n} steps), "
            f"got {len(conv['audios'])}"
        )


def test_conversation_audio_tags_match_audios_list(paths):
    """Number of <audio> tags in tool results == len(audios)."""
    for archetype, path_data in paths.items():
        conv = _build_minimal_conversation(path_data)
        msgs = conv["messages"]
        n_audio_tags = sum(
            m.get("content", "").count("<audio>")
            for m in msgs
        )
        assert n_audio_tags == len(conv["audios"]), (
            f"{archetype}: <audio> tag count ({n_audio_tags}) != "
            f"audios list len ({len(conv['audios'])})"
        )
