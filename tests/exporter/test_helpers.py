# (c) JFrog Ltd. (2026)

"""Pure-helper tests for :mod:`belt.exporter.helpers`.

Helpers are the only place exporter business logic is allowed to live
(Design Principle 2 keeps it out of entities). Every helper has its own
focused test so a regression in one cannot mask a regression in another.
"""

from __future__ import annotations

from belt.entities import ScenarioScore
from belt.exporter.helpers import collapse_trials, get_bool_option, get_int_option, truncate_with_marker


class TestCollapseTrials:
    def test_groups_trial_suffix(self):
        scores = [
            ScenarioScore(scenario_name="s__trial_0", group="g", overall_pass=True),
            ScenarioScore(scenario_name="s__trial_1", group="g", overall_pass=False),
            ScenarioScore(scenario_name="other", group="g", overall_pass=True),
        ]
        groups = collapse_trials(scores)
        assert sorted(groups) == ["g/other", "g/s"]
        assert len(groups["g/s"]) == 2
        assert len(groups["g/other"]) == 1

    def test_preserves_first_seen_order_within_a_group(self):
        scores = [
            ScenarioScore(scenario_name="s__trial_2", group="g", overall_pass=True),
            ScenarioScore(scenario_name="s__trial_0", group="g", overall_pass=False),
        ]
        order = [s.scenario_name for s in collapse_trials(scores)["g/s"]]
        assert order == ["s__trial_2", "s__trial_0"]


class TestGetIntOption:
    def test_int_passthrough(self):
        assert get_int_option({"x": 7}, "x", 0) == 7

    def test_string_parsed(self):
        assert get_int_option({"x": "42"}, "x", 0) == 42

    def test_bool_rejected(self):
        # ``True`` is an int subclass; rejecting it explicitly avoids
        # accidentally treating ``include_stdout: true`` as ``1``.
        assert get_int_option({"x": True}, "x", 99) == 99

    def test_garbage_falls_back(self):
        assert get_int_option({"x": "abc"}, "x", 99) == 99
        assert get_int_option({"x": object()}, "x", 99) == 99
        assert get_int_option({}, "x", 99) == 99


class TestGetBoolOption:
    def test_bool_passthrough(self):
        assert get_bool_option({"x": True}, "x", False) is True
        assert get_bool_option({"x": False}, "x", True) is False

    def test_truthy_strings(self):
        for v in ("1", "true", "TRUE", "Yes", "on"):
            assert get_bool_option({"x": v}, "x", False) is True

    def test_falsy_strings_default_false(self):
        for v in ("0", "false", "no", "off", ""):
            assert get_bool_option({"x": v}, "x", True) is False

    def test_default_used_when_absent(self):
        assert get_bool_option({}, "x", True) is True
        assert get_bool_option({}, "x", False) is False


class TestTruncateWithMarker:
    def test_no_truncation_when_under_cap(self):
        assert truncate_with_marker("hello", 100) == "hello"

    def test_truncation_marker_appended(self):
        out = truncate_with_marker("a" * 1000, 50)
        assert len(out.encode("utf-8")) <= 50 + 64  # cap + marker
        assert "[truncated;" in out

    def test_zero_max_bytes(self):
        assert truncate_with_marker("anything", 0) == ""

    def test_handles_multibyte_safely(self):
        # 4-byte emoji at the boundary - decode() with errors='replace'
        # must not raise.
        text = "\U0001f600" * 100
        out = truncate_with_marker(text, 5)
        assert isinstance(out, str)
