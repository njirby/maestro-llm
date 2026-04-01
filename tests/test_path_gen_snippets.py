"""
Unit tests for path_gen snippet generation (no REAPER required).

Tests the new reapy-based action_snippet, search_snippet, and search_result
fields added to each iteration in generate_preset_path().
"""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from maestro.synth.path_gen import (
    _json_key_to_display,
    _search_keyword,
    _generate_search_result,
    generate_preset_path,
    PARAM_RANGES,
)


# ---------------------------------------------------------------------------
# _json_key_to_display
# ---------------------------------------------------------------------------

class TestJsonKeyToDisplay:
    def test_osc(self):
        assert _json_key_to_display("osc_1_level") == "Oscillator 1 Level"

    def test_filter(self):
        assert _json_key_to_display("filter_1_cutoff") == "Filter 1 Cutoff"

    def test_lfo(self):
        assert _json_key_to_display("lfo_2_frequency") == "LFO 2 Frequency"

    def test_env(self):
        assert _json_key_to_display("env_1_attack") == "Envelope 1 Attack"

    def test_eq(self):
        assert _json_key_to_display("eq_band_1_gain") == "EQ Band 1 Gain"

    def test_no_abbreviation(self):
        assert _json_key_to_display("chorus_dry_wet") == "Chorus Dry Wet"

    def test_single_word(self):
        assert _json_key_to_display("bypass") == "Bypass"


# ---------------------------------------------------------------------------
# _search_keyword
# ---------------------------------------------------------------------------

class TestSearchKeyword:
    def test_filter_dominates(self):
        params = ["filter_1_cutoff", "filter_1_resonance", "osc_1_level"]
        # filter_1 appears twice, osc_1 once → filter_1 dominates
        kw = _search_keyword(params)
        assert "filter" in kw

    def test_osc_group(self):
        params = ["osc_1_level", "osc_1_transpose", "osc_1_detune"]
        kw = _search_keyword(params)
        assert "oscillator" in kw

    def test_env_group(self):
        params = ["env_1_attack", "env_1_decay", "env_1_sustain"]
        kw = _search_keyword(params)
        assert "envelope" in kw

    def test_lfo_group(self):
        params = ["lfo_1_frequency", "lfo_1_phase"]
        kw = _search_keyword(params)
        assert "lfo" in kw

    def test_single_param(self):
        kw = _search_keyword(["filter_1_cutoff"])
        assert "filter" in kw

    def test_returns_lowercase(self):
        kw = _search_keyword(["osc_1_level"])
        assert kw == kw.lower()


# ---------------------------------------------------------------------------
# _generate_search_result
# ---------------------------------------------------------------------------

class TestGenerateSearchResult:
    def test_format_lines(self):
        cumulative = {}  # will use defaults
        result = _generate_search_result("filter 1", cumulative)
        assert result, "search result should not be empty for 'filter 1'"
        for line in result.splitlines():
            assert ": " in line, f"line missing ': ' separator: {line!r}"
            name, val_str = line.rsplit(": ", 1)
            val = float(val_str)
            assert 0.0 <= val <= 1.0, f"value {val} out of [0,1] for param {name!r}"

    def test_keyword_in_every_result(self):
        result = _generate_search_result("envelope 1", {})
        for line in result.splitlines():
            name = line.rsplit(": ", 1)[0]
            assert "envelope 1" in name.lower(), f"unexpected param: {name!r}"

    def test_respects_cumulative_values(self):
        # Set filter_1_cutoff to its max (normalized = 1.0)
        r = PARAM_RANGES["filter_1_cutoff"]
        cumulative = {"filter_1_cutoff": r["max"]}
        result = _generate_search_result("filter 1 cutoff", cumulative)
        lines = {l.rsplit(": ", 1)[0]: float(l.rsplit(": ", 1)[1])
                 for l in result.splitlines()}
        assert "Filter 1 Cutoff" in lines
        assert abs(lines["Filter 1 Cutoff"] - 1.0) < 1e-4

    def test_empty_for_unknown_keyword(self):
        result = _generate_search_result("xyzzy_nonexistent_param", {})
        assert result == ""

    def test_sorted_alphabetically(self):
        result = _generate_search_result("filter 1", {})
        names = [l.rsplit(": ", 1)[0] for l in result.splitlines()]
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# generate_preset_path iteration fields
# ---------------------------------------------------------------------------

class TestIterationSnippetFields:
    @pytest.fixture(scope="class")
    def path_result(self):
        rng = random.Random(42)
        return generate_preset_path("bass", rng)

    def test_has_search_fields(self, path_result):
        for it in path_result["iterations"]:
            assert "action_snippet" in it
            assert "search_snippet" in it
            assert "search_result" in it
            assert "search_keyword" in it

    def test_action_snippet_is_valid_python(self, path_result):
        for it in path_result["iterations"]:
            try:
                compile(it["action_snippet"], "<action_snippet>", "exec")
            except SyntaxError as e:
                pytest.fail(f"SyntaxError in action_snippet step {it['step']}: {e}")

    def test_search_snippet_is_valid_python(self, path_result):
        for it in path_result["iterations"]:
            try:
                compile(it["search_snippet"], "<search_snippet>", "exec")
            except SyntaxError as e:
                pytest.fail(f"SyntaxError in search_snippet step {it['step']}: {e}")

    def test_no_vital_controller(self, path_result):
        for it in path_result["iterations"]:
            assert "VitalController" not in it["action_snippet"]
            assert "VitalController" not in it["search_snippet"]

    def test_action_snippet_uses_reapy_high_level(self, path_result):
        for it in path_result["iterations"]:
            assert "reapy.Project" in it["action_snippet"]
            if it["params_applied"]:  # non-empty steps must have fx.params[ assignments
                assert "fx.params[" in it["action_snippet"]

    def test_action_snippet_values_in_0_1(self, path_result):
        import re
        for it in path_result["iterations"]:
            # Extract .value = X lines
            for match in re.finditer(r'\.value\s*=\s*([0-9.]+)', it["action_snippet"]):
                val = float(match.group(1))
                assert 0.0 <= val <= 1.0, f"value {val} out of [0,1] in step {it['step']}"

    def test_search_result_format(self, path_result):
        for it in path_result["iterations"]:
            result = it["search_result"]
            if not result:
                continue
            for line in result.splitlines():
                assert ": " in line, f"bad search result line: {line!r}"
                val = float(line.rsplit(": ", 1)[1])
                assert 0.0 <= val <= 1.0

    def test_search_result_reflects_pre_step_state(self, path_result):
        # On step 1, all values should be init/default values (cumulative hasn't changed yet)
        first_it = path_result["iterations"][0]
        keyword = first_it["search_keyword"]
        result = first_it["search_result"]
        if not result:
            return
        # At step 1, values should come from init preset — verify they're not all 1.0
        # (which would indicate wrong cumulative state was used)
        vals = [float(l.rsplit(": ", 1)[1]) for l in result.splitlines()]
        assert not all(v == 1.0 for v in vals), "all values are 1.0 — wrong cumulative state?"

    def test_search_keyword_present_in_result_names(self, path_result):
        for it in path_result["iterations"]:
            keyword = it["search_keyword"]
            for line in it["search_result"].splitlines():
                name = line.rsplit(": ", 1)[0]
                assert keyword in name.lower(), (
                    f"keyword '{keyword}' not in result name '{name}'"
                )
