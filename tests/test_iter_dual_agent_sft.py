from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.build_iter_sft_dataset import (
    _build_main_search_tool_call,
    _build_search_agent_record,
    validate_search_record,
)


def test_build_main_search_tool_call_bash_mode():
    call = _build_main_search_tool_call(
        step_num=3,
        keyword="filter 1",
        queries=["filter 1"],
        search_snippet="print('search')",
        mode="bash",
    )
    assert call["id"] == "tc_search_3"
    assert call["function"]["name"] == "bash"
    args = json.loads(call["function"]["arguments"])
    assert args["command"] == "print('search')"


def test_build_main_search_tool_call_search_agent_mode():
    call = _build_main_search_tool_call(
        step_num=4,
        keyword="modulation 10",
        queries=["modulation 10", "modulation"],
        search_snippet="print('ignored in delegated mode')",
        mode="search_agent",
    )
    assert call["id"] == "tc_search_4"
    assert call["function"]["name"] == "search_agent"
    args = json.loads(call["function"]["arguments"])
    assert args["query"] == "modulation 10"
    assert args["queries"] == ["modulation 10", "modulation"]
    assert args["parallelism"] == 2
    assert args["output_contract"] == "json_bundle_with_reports_and_consensus"
    assert "modulation slot 10" in args["goal"].lower()


def test_build_search_agent_record_and_validate():
    rec = _build_search_agent_record(
        sample_id="lead_deadbeef",
        step_num=2,
        archetype="lead",
        keyword="filter 1",
        search_snippet="import reapy\nprint('ok')",
        search_result="Filter 1 Cutoff: 0.5123\nFilter 1 Resonance: 0.2012",
        primary_family="filter1",
        support_family="env2",
        intent_tags=["focus:filter1", "stage:core"],
    )
    assert rec["id"] == "lead_deadbeef_search_step02_agent01"
    assert len(rec["messages"]) == 4
    assert rec["messages"][1]["tool_calls"][0]["function"]["name"] == "bash"
    assert rec["messages"][-1]["role"] == "assistant"
    assert "tool_calls" not in rec["messages"][-1]
    validate_search_record(rec)


def test_validate_search_record_rejects_missing_tool_reference():
    rec = _build_search_agent_record(
        sample_id="bass_deadbeef",
        step_num=1,
        archetype="bass",
        keyword="oscillator 1",
        search_snippet="print('ok')",
        search_result="Oscillator 1 Level: 0.7000",
        primary_family="osc",
        support_family=None,
    )
    rec["messages"][2]["tool_call_id"] = "tc_bad"
    with pytest.raises(AssertionError):
        validate_search_record(rec)
