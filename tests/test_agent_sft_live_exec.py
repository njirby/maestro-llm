"""Live REAPER integration tests for grade_agent_sft live execution checks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.grade_agent_sft import run_live_execution_checks_for_record

pytestmark = pytest.mark.reaper


def _record_for_command(command: str, response_payload: dict) -> dict:
    return {
        "id": "live_exec_record",
        "task_type": "main",
        "messages": [
            {"role": "user", "content": "<audio>\nRecreate this sound."},
            {"role": "assistant", "content": "Executing command."},
            {"role": "tool_call", "content": json.dumps({"name": "bash", "arguments": {"command": command}})},
            {"role": "tool_response", "content": json.dumps(response_payload)},
            {"role": "assistant", "content": "Done."},
        ],
        "audios": ["/tmp/t.wav"],
        "meta": {"step_labels": []},
        "labels": {},
    }


def test_live_exec_check_runs_bash_command_in_reaper_context():
    command = (
        "python3 - <<'PY'\n"
        "import json\n"
        "import reapy\n"
        "with reapy.inside_reaper():\n"
        "    project = reapy.Project()\n"
        "    _ = len(project.tracks)\n"
        "print(json.dumps({'status': 'ok'}))\n"
        "PY"
    )
    record = _record_for_command(command, {"status": "ok"})
    result = run_live_execution_checks_for_record(record, timeout_sec=10.0)
    assert result["execution_calls_assessed"] == 1
    assert result["execution_calls_passed"] == 1
    assert result["execution_fidelity"] == 1.0


def test_live_exec_check_detects_key_field_mismatch():
    command = (
        "python3 - <<'PY'\n"
        "import json\n"
        "print(json.dumps({'status': 'ok', 'applied_wavetable': 'A'}))\n"
        "PY"
    )
    record = _record_for_command(command, {"status": "ok", "applied_wavetable": "B"})
    result = run_live_execution_checks_for_record(record, timeout_sec=10.0)
    assert result["execution_calls_assessed"] == 1
    assert result["execution_calls_passed"] == 0
    assert result["execution_fidelity"] == 0.0
    assert any("applied_wavetable_mismatch" in e for e in result["execution_checks"][0]["errors"])
