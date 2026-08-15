"""Recovery-injection contracts: real tracebacks, honest search coverage.

Two recovery types the corpus previously lacked:
  1. CODE mistakes — a bash call emitted broken, its REAL traceback, then the
     fix. Without these the corpus has zero failed commands, so a deployed
     model has never seen an error message.
  2. Honest search misses — round 1 auditions only part of the library, so a
     GT outside that window is genuinely never heard and the judge's
     ``no_match`` is truthful (the old forced miss contradicted the
     evidence-derived shortlist labels).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import opencode_contract as oc
from scripts.agent_sft_common import (  # noqa: E402
    compute_round_coverage,
    execute_for_traceback,
    oc_bash_call_msg,
    oc_bash_response_msg,
    oc_emit_code_mistake_sequence,
    select_and_apply_mutation,
    slices_for_regions,
)

# A snippet that runs standalone (no reapy/dawdreamer), so the traceback below
# is genuinely produced by executing the mutated code rather than stubbed.
_CORRECT_PY = (
    "import json\n"
    "preset = {'settings': {'osc_1_level': 0.8}}\n"
    "print(json.dumps({'status': 'ok', 'n': len(preset['settings'])}))\n"
)
_CORRECT_CMD = f"python3 - <<'PY'\n{_CORRECT_PY}\nPY"


def _find_mutation(name: str):
    """Deterministically obtain a specific mutation applied to _CORRECT_CMD."""
    for seed in range(200):
        sel = select_and_apply_mutation(
            "tuple", _CORRECT_CMD, random.Random(seed), reaper_available=False)
        if sel and sel[0].name == name:
            return sel
    raise AssertionError(f"mutation {name!r} never selected")


def test_traceback_comes_from_real_execution():
    mutation, broken = _find_mutation("key_error_settings")
    assert broken != _CORRECT_CMD
    tb = execute_for_traceback(broken, cwd=str(ROOT), as_bash=True, timeout=30.0)
    assert tb, "broken code must actually fail"
    assert "KeyError" in tb and "setting" in tb, tb
    # The correct command must NOT fail — otherwise the 'mistake' is meaningless.
    assert execute_for_traceback(
        _CORRECT_CMD, cwd=str(ROOT), as_bash=True, timeout=30.0) is None


def test_emission_is_the_six_message_recovery_shape():
    mutation, broken = _find_mutation("key_error_settings")
    tb = execute_for_traceback(broken, cwd=str(ROOT), as_bash=True, timeout=30.0)
    assert tb

    messages: list[dict] = []
    messages.append({"role": "assistant", "content": "Applying the wavetable tuple."})
    oc_emit_code_mistake_sequence(messages, broken, tb, mutation)
    # The caller's normal path then emits the correct call + its success output.
    messages.append(oc_bash_call_msg(_CORRECT_CMD))
    messages.append(oc_bash_response_msg('{"status": "ok", "n": 1}'))

    assert [m["role"] for m in messages] == [
        "assistant", "tool_call", "tool_response",
        "assistant", "tool_call", "tool_response",
    ]
    broken_call = json.loads(messages[1]["content"])
    fixed_call = json.loads(messages[4]["content"])
    assert broken_call["name"] == "bash" and fixed_call["name"] == "bash"
    assert broken_call["arguments"]["command"] != fixed_call["arguments"]["command"]
    # Failure rides the contract's failure shape, carrying the real traceback.
    assert messages[2]["content"] == oc.bash_output("", exit_code=1, stderr=tb)
    assert messages[2]["content"].startswith("Command failed with exit code 1.")
    assert "KeyError" in messages[2]["content"]
    # The diagnosis names the error and the fix, so the model learns to read it.
    assert messages[3]["content"] == mutation.diagnosis
    assert "settings" in messages[3]["content"]


def test_round_coverage_partitions_library_exactly_once():
    for trial in range(500):
        rng = random.Random(trial)
        total = rng.choice([1, 5, 48, 96, 282, 283])
        coverage = rng.choice([0.25, 0.5, 0.6, 0.9])
        r1, r2 = compute_round_coverage(total, coverage, rng)
        seen = [i for a, b in r1 + r2 for i in range(a, b)]
        assert sorted(seen) == list(range(total)), (total, coverage, r1, r2)
        assert len(seen) == len(set(seen)), "rounds must not re-audition"
        # Shards inherit the same property.
        shards = slices_for_regions(r1, 48) + slices_for_regions(r2, 48)
        assert sorted(i for a, b in shards for i in range(a, b)) == list(range(total))
        assert all(b - a <= 48 for a, b in shards)


def test_gt_inside_round1_needs_one_round_outside_needs_two():
    total, coverage = 282, 0.6
    inside = outside = 0
    for trial in range(300):
        rng = random.Random(trial)
        r1, r2 = compute_round_coverage(total, coverage, rng)
        gt = rng.randrange(total)
        in_r1 = any(a <= gt < b for a, b in r1)
        if in_r1:
            inside += 1
        else:
            outside += 1
            # The GT must be reachable in round 2 — never lost entirely.
            assert any(a <= gt < b for a, b in r2), (gt, r1, r2)
    # Both branches must actually occur, near the configured coverage.
    assert inside > 0 and outside > 0
    assert 0.45 < inside / (inside + outside) < 0.75


def test_full_coverage_disables_the_second_round():
    r1, r2 = compute_round_coverage(282, 1.0, random.Random(0))
    assert r1 == [(0, 282)] and r2 == []
