# (c) JFrog Ltd. (2026)

"""Integration tests for trajectory checks driven through ``check_trajectory``.

The broader rules-scoring suite lives in ``test_rules.py``; this file is the
narrow integration coverage for the result-matching checks
(``tool_result_contains`` and ``tool_result_pattern``) that the helper-level
tests in ``test_helpers.py`` complement.
"""

from __future__ import annotations

from belt.entities import ToolCall, TurnExpectation, TurnOutput
from belt.scorer.rules.trajectory import check_trajectory


def _checks(output: TurnOutput, expect: dict) -> dict[str, bool]:
    exp = TurnExpectation(**expect)
    results = check_trajectory(0, output, exp)
    return {r.check: r.passed for r in results}


def _out(**kwargs) -> TurnOutput:
    return TurnOutput(raw_cli="", **kwargs)


# -- tool_result_contains ---------------------------------------------------


def test_tool_result_contains_match() -> None:
    to = _out(
        tool_calls=[
            ToolCall(name="Read", call_id="c1", args={"path": "pyproject.toml"}, result={"output": 'name = "belt"'})
        ]
    )
    r = _checks(to, {"tool_result_contains": {"Read": "belt"}})
    assert r["tool_result_contains(Read)"] is True


def test_tool_result_contains_no_match() -> None:
    to = _out(
        tool_calls=[ToolCall(name="Read", call_id="c1", args={}, result={"output": "something unrelated"})],
    )
    r = _checks(to, {"tool_result_contains": {"Read": "belt"}})
    assert r["tool_result_contains(Read)"] is False


def test_tool_result_contains_tool_not_invoked() -> None:
    to = _out(tool_calls=[ToolCall(name="Write", call_id="c1", args={}, result={"output": "belt"})])
    r = _checks(to, {"tool_result_contains": {"Read": "belt"}})
    assert r["tool_result_contains(Read)"] is False


def test_tool_result_contains_multiple_calls_any_match() -> None:
    to = _out(
        tool_calls=[
            ToolCall(name="Read", call_id="c1", args={}, result={"output": "first file"}),
            ToolCall(name="Read", call_id="c2", args={}, result={"output": "second file with belt inside"}),
        ]
    )
    r = _checks(to, {"tool_result_contains": {"Read": "belt"}})
    assert r["tool_result_contains(Read)"] is True


def test_tool_result_contains_none_result_does_not_match() -> None:
    to = _out(tool_calls=[ToolCall(name="Read", call_id="c1", args={}, result=None)])
    r = _checks(to, {"tool_result_contains": {"Read": "belt"}})
    assert r["tool_result_contains(Read)"] is False


def test_tool_result_contains_not_checked_when_empty() -> None:
    r = _checks(_out(), {})
    assert not any("tool_result_contains" in k for k in r)


# -- tool_result_pattern ----------------------------------------------------


def test_tool_result_pattern_match() -> None:
    to = _out(
        tool_calls=[ToolCall(name="Read", call_id="c1", args={}, result={"output": 'name = "belt"'})],
    )
    r = _checks(to, {"tool_result_pattern": {"Read": r'name\s*=\s*"belt"'}})
    assert r["tool_result_pattern(Read)"] is True


def test_tool_result_pattern_no_match() -> None:
    to = _out(
        tool_calls=[ToolCall(name="Read", call_id="c1", args={}, result={"output": "no project name here"})],
    )
    r = _checks(to, {"tool_result_pattern": {"Read": r'name\s*=\s*"belt"'}})
    assert r["tool_result_pattern(Read)"] is False


def test_tool_result_pattern_tool_not_invoked() -> None:
    to = _out(tool_calls=[ToolCall(name="Bash", call_id="c1", args={}, result={"output": 'name = "belt"'})])
    r = _checks(to, {"tool_result_pattern": {"Read": r"belt"}})
    assert r["tool_result_pattern(Read)"] is False


def test_tool_result_pattern_multiple_calls_any_match() -> None:
    to = _out(
        tool_calls=[
            ToolCall(name="Read", call_id="c1", args={}, result={"output": "no relevant content"}),
            ToolCall(name="Read", call_id="c2", args={}, result={"output": '{"project": "belt"}'}),
        ]
    )
    r = _checks(to, {"tool_result_pattern": {"Read": r'"project":\s*"belt"'}})
    assert r["tool_result_pattern(Read)"] is True


def test_tool_result_pattern_invalid_regex_rejected_at_validation() -> None:
    """A bad ``tool_result_pattern`` regex aborts at scenario load.

    Bad regexes route through ``belt._regex_policy.compile_user_regex``
    via the ``TurnExpectation`` validator and raise at construction
    time, so the scorer never sees a malformed pattern - and never
    needs a silent-fail branch.
    """
    import pytest

    with pytest.raises(Exception) as exc_info:
        _checks(
            _out(tool_calls=[ToolCall(name="Read", call_id="c1", args={}, result={"output": "anything"})]),
            {"tool_result_pattern": {"Read": r"(unclosed"}},
        )
    msg = str(exc_info.value)
    assert "invalid regex" in msg
    assert "(unclosed" in msg


def test_tool_result_pattern_not_checked_when_empty() -> None:
    r = _checks(_out(), {})
    assert not any("tool_result_pattern" in k for k in r)


# -- both fields together ---------------------------------------------------


def test_tool_result_contains_and_pattern_combined() -> None:
    to = _out(
        tool_calls=[ToolCall(name="Read", call_id="c1", args={}, result={"output": 'name = "belt"'})],
    )
    r = _checks(
        to,
        {
            "tool_result_contains": {"Read": "belt"},
            "tool_result_pattern": {"Read": r'name\s*=\s*"belt"'},
        },
    )
    assert r["tool_result_contains(Read)"] is True
    assert r["tool_result_pattern(Read)"] is True
