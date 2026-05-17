# (c) JFrog Ltd. (2026)

"""Tests for the rule-based scorer checks.

The scorer is fully agent-agnostic - it reads only TurnOutput fields.
These tests verify that contract without any agent-specific imports.
"""

from __future__ import annotations

import pytest

from belt.entities import StateExpectation, ToolCall, TurnExpectation, TurnOutput, TurnTiming
from belt.scorer.payloads import CheckEntry
from belt.scorer.rules import RuleBasedScorer
from belt.scorer.rules.scorer import _check_turn
from belt.scorer.rules.state import check_state

scorer = RuleBasedScorer()


def _checks(output: TurnOutput, expect: dict) -> dict[str, bool]:
    """Run checks, return {check_name: passed}."""
    exp = TurnExpectation(**expect)
    results = _check_turn(0, output, exp)
    return {r.check: r.passed for r in results}


def _check_results(output: TurnOutput, expect: dict, turn_idx: int = 0) -> list[CheckEntry]:
    """Run checks, return full CheckEntry list."""
    exp = TurnExpectation(**expect)
    return _check_turn(turn_idx, output, exp)


def _out(raw_cli: str = "", **kwargs) -> TurnOutput:
    """Build a TurnOutput with defaults."""
    return TurnOutput(raw_cli=raw_cli, **kwargs)


# ── execution/no_errors ──


def test_no_errors_clean() -> None:
    assert _checks(_out("clean output"), {})["no_errors"] is True


@pytest.mark.parametrize("pattern", ["Traceback (most recent call last)", "RuntimeError: boom", "validation error"])
def test_no_errors_detected(pattern: str) -> None:
    assert _checks(_out(f"some output\n{pattern}\nmore output"), {})["no_errors"] is False


def test_no_errors_via_has_error_field() -> None:
    assert _checks(_out("clean text", has_error=True), {})["no_errors"] is False


def test_no_errors_discussing_exceptions_is_not_false_positive() -> None:
    """Agent discussing error types should not trigger no_errors failure."""
    text = "You should catch ValueError and TypeError in your code. A KeyError happens when..."
    assert _checks(_out(text), {})["no_errors"] is True


def test_no_errors_actual_traceback_still_detected() -> None:
    text = "Output:\nTraceback (most recent call last):\n  File 'x.py'\nValueError: bad input\n"
    assert _checks(_out(text), {})["no_errors"] is False


def test_no_errors_structured_false_skips_pattern_scan() -> None:
    """When agent explicitly sets has_error=False, ERROR_PATTERNS in text must not override."""
    text = "The function raises ZeroDivisionError: division by zero when input is 0"
    assert _checks(_out(text, has_error=False), {})["no_errors"] is True


def test_no_errors_structured_none_falls_back_to_patterns() -> None:
    """When has_error is None (no structured signal), ERROR_PATTERNS scan runs as fallback."""
    text = "RuntimeError: something went wrong"
    assert _checks(_out(text, has_error=None), {})["no_errors"] is False


# ── execution/not_contains ──


def test_not_contains_pass() -> None:
    r = _checks(_out("all good"), {"not_contains": ["forbidden"]})
    assert r["not_contains(forbidden)"] is True


def test_not_contains_fail() -> None:
    r = _checks(_out("this is forbidden text"), {"not_contains": ["forbidden"]})
    assert r["not_contains(forbidden)"] is False


# ── trajectory/tool_invoked ──


def test_tool_invoked_via_tool_calls() -> None:
    to = _out(tool_calls=[ToolCall(name="manage_watches", call_id="c1")])
    r = _checks(to, {"tools_invoked": ["manage_watches"]})
    assert r["tool_invoked(manage_watches)"] is True


def test_tool_invoked_via_cli_text() -> None:
    r = _checks(_out("manage_watches call_abc"), {"tools_invoked": ["manage_watches"]})
    assert r["tool_invoked(manage_watches)"] is True


def test_tool_invoked_absent() -> None:
    r = _checks(_out("get_registry_overview"), {"tools_invoked": ["manage_watches"]})
    assert r["tool_invoked(manage_watches)"] is False


# ── trajectory/tools_invoked_any (OR logic) ──


def test_tools_invoked_any_first_match() -> None:
    to = _out(tool_calls=[ToolCall(name="search_releases", call_id="c1")])
    r = _checks(to, {"tools_invoked_any": [["search_releases", "get_registry_overview"]]})
    assert r["tool_invoked_any(search_releases|get_registry_overview)"] is True


def test_tools_invoked_any_second_match() -> None:
    to = _out(tool_calls=[ToolCall(name="get_registry_overview", call_id="c1")])
    r = _checks(to, {"tools_invoked_any": [["search_releases", "get_registry_overview"]]})
    assert r["tool_invoked_any(search_releases|get_registry_overview)"] is True


def test_tools_invoked_any_no_match() -> None:
    to = _out(tool_calls=[ToolCall(name="some_other_tool", call_id="c1")])
    r = _checks(to, {"tools_invoked_any": [["search_releases", "get_registry_overview"]]})
    assert r["tool_invoked_any(search_releases|get_registry_overview)"] is False


def test_tools_invoked_any_not_checked_when_empty() -> None:
    r = _checks(_out("anything"), {})
    assert not any("tool_invoked_any" in k for k in r)


# ── response/has_reply (agent-populated) ──


def test_has_reply_true() -> None:
    to = _out(has_reply=True)
    r = _checks(to, {"has_reply": True})
    assert r["has_reply"] is True


def test_has_reply_false() -> None:
    to = _out(has_reply=False)
    r = _checks(to, {"has_reply": True})
    assert r["has_reply"] is False


# ── response/contains ──


def test_contains_in_reply_text() -> None:
    to = _out(reply_text="The environment load-test was created")
    r = _checks(to, {"contains": ["load-test"]})
    assert r["contains(load-test)"] is True


def test_contains_case_insensitive() -> None:
    to = _out(reply_text="The environment LOAD-TEST was created")
    r = _checks(to, {"contains": ["load-test"]})
    assert r["contains(load-test)"] is True


def test_contains_in_cli_fallback() -> None:
    to = _out(raw_cli="The environment load-test was created", reply_text="")
    r = _checks(to, {"contains": ["load-test"]})
    assert r["contains(load-test)"] is True


def test_contains_absent() -> None:
    to = _out(raw_cli="staging was created", reply_text="staging was created")
    r = _checks(to, {"contains": ["load-test"]})
    assert r["contains(load-test)"] is False


# ── response/reply_pattern ──
#
# Strict-format sibling of ``contains``: regex-based, ALL must match,
# matched against ``reply_text`` only (no ``raw_cli`` fallback). The
# no-fallback rule is the load-bearing distinction from ``contains`` -
# CLI noise must never produce a false green.


def test_reply_pattern_empty_list_is_no_op() -> None:
    """No ``reply_pattern`` entries means no checks emitted."""
    to = _out(reply_text="anything goes here")
    results = _check_results(to, {"reply_pattern": []})
    pattern_checks = [r for r in results if r.check.startswith("reply_pattern(")]
    assert pattern_checks == []


def test_reply_pattern_single_match_passes() -> None:
    to = _out(reply_text="ORDER|42|Ada Lovelace|shipped|1Z999")
    r = _checks(to, {"reply_pattern": [r"^ORDER\|"]})
    assert r[r"reply_pattern(^ORDER\|)"] is True


def test_reply_pattern_single_mismatch_fails() -> None:
    to = _out(reply_text="here is your order: 42")
    r = _checks(to, {"reply_pattern": [r"^ORDER\|"]})
    assert r[r"reply_pattern(^ORDER\|)"] is False


def test_reply_pattern_all_must_match_semantics() -> None:
    """ALL patterns must match (each emits its own check); one failing
    entry does not prevent the others from being evaluated."""
    to = _out(reply_text="ORDER|42|Ada|shipped|1Z999")
    r = _checks(
        to,
        {"reply_pattern": [r"^ORDER\|", r"\|shipped\|", r"\|cancelled\|"]},
    )
    assert r[r"reply_pattern(^ORDER\|)"] is True
    assert r[r"reply_pattern(\|shipped\|)"] is True
    assert r[r"reply_pattern(\|cancelled\|)"] is False


def test_reply_pattern_anchored_pattern_matches_whole_reply() -> None:
    """``$`` anchors to end-of-string when the reply has no trailing newline."""
    to = _out(reply_text="ORDER|42|Ada|shipped|1Z999")
    r = _checks(to, {"reply_pattern": [r"\|1Z999$"]})
    assert r[r"reply_pattern(\|1Z999$)"] is True


def test_reply_pattern_no_multiline_default() -> None:
    """Without inline ``(?m)``, ``^foo$`` against a multi-line reply does not match.

    Locks the documented "no per-line magic by default" contract: the
    full reply is one string; per-line semantics require explicit opt-in.
    """
    to = _out(reply_text="first line\nORDER|42\nthird line")
    r = _checks(to, {"reply_pattern": [r"^ORDER\|42$"]})
    assert r[r"reply_pattern(^ORDER\|42$)"] is False


def test_reply_pattern_multiline_opt_in_via_inline_flag() -> None:
    """``(?m)`` flips on per-line ``^``/``$`` for that pattern."""
    to = _out(reply_text="first line\nORDER|42\nthird line")
    r = _checks(to, {"reply_pattern": [r"(?m)^ORDER\|42$"]})
    assert r[r"reply_pattern((?m)^ORDER\|42$)"] is True


def test_reply_pattern_does_not_fall_back_to_raw_cli() -> None:
    """The load-bearing distinction from ``contains``.

    A token that lives only in CLI noise (debug output, agent-trace
    dumps, stderr) must not green a strict-format assertion.
    ``contains`` accepts a ``raw_cli`` fallback by design (permissive
    substring assertion); ``reply_pattern`` is strictly reply-only.
    """
    to = _out(raw_cli="[debug] ORDER|42|...", reply_text="sorry, can't do that")
    r = _checks(to, {"contains": ["ORDER|42"], "reply_pattern": [r"ORDER\|42"]})
    assert r["contains(ORDER|42)"] is True  # locks existing semantics
    assert r[r"reply_pattern(ORDER\|42)"] is False  # locks new semantics


def test_reply_pattern_failure_details_carry_truncated_reply() -> None:
    """Failure ``details`` quote a truncated reply so reviewers see what the agent
    actually said. The truncation guards against pathological 100KB replies."""
    long_reply = "x" * 500
    to = _out(reply_text=long_reply)
    results = _check_results(to, {"reply_pattern": [r"^ORDER\|"]})
    failure = next(r for r in results if r.check.startswith("reply_pattern("))
    assert failure.passed is False
    assert "reply did not match" in failure.details
    assert len(failure.details) < 400  # truncated, not full 500 chars


# ── efficiency/max_llm_turns (agent-populated) ──


def test_max_llm_turns_within() -> None:
    to = _out(llm_turn_count=2)
    r = _checks(to, {"max_llm_turns": 3})
    assert r["max_llm_turns(<=3)"] is True


def test_max_llm_turns_exceeded() -> None:
    to = _out(llm_turn_count=5)
    r = _checks(to, {"max_llm_turns": 3})
    assert r["max_llm_turns(<=3)"] is False


def test_max_llm_turns_none_treated_as_zero() -> None:
    to = _out(llm_turn_count=None)
    r = _checks(to, {"max_llm_turns": 0})
    assert r["max_llm_turns(<=0)"] is True


# ── efficiency/max_tool_calls ──


def test_max_tool_calls_within() -> None:
    to = _out(tool_calls=[ToolCall(name="t1", call_id="c1"), ToolCall(name="t2", call_id="c2")])
    r = _checks(to, {"max_tool_calls": 3})
    assert r["max_tool_calls(<=3)"] is True


def test_max_tool_calls_exceeded() -> None:
    to = _out(tool_calls=[ToolCall(name=f"t{i}", call_id=f"c{i}") for i in range(5)])
    r = _checks(to, {"max_tool_calls": 3})
    assert r["max_tool_calls(<=3)"] is False


# ── performance (agent-populated timing) ──


def test_perf_total_within() -> None:
    to = _out(timing=TurnTiming(ttft=1.2, ttlt=1.7, total=1.7))
    r = _checks(to, {"max_total_seconds": 5.0})
    assert r["max_total_seconds(<=5.0s)"] is True


def test_perf_total_exceeded() -> None:
    to = _out(timing=TurnTiming(ttft=3.0, ttlt=5.0, total=5.0))
    r = _checks(to, {"max_total_seconds": 3.0})
    assert r["max_total_seconds(<=3.0s)"] is False


def test_perf_ttft_within() -> None:
    to = _out(timing=TurnTiming(ttft=0.8, ttlt=1.8, total=1.8))
    r = _checks(to, {"max_ttft_seconds": 2.0})
    assert r["max_ttft_seconds(<=2.0s)"] is True


def test_perf_ttfe_within() -> None:
    to = _out(timing=TurnTiming(ttfe=0.3, ttft=1.0, ttlt=2.0, total=2.0))
    r = _checks(to, {"max_ttfe_seconds": 1.0})
    assert r["max_ttfe_seconds(<=1.0s)"] is True


def test_perf_ttlt_exceeded() -> None:
    to = _out(timing=TurnTiming(ttft=3.0, ttlt=7.0, total=7.0))
    r = _checks(to, {"max_ttlt_seconds": 5.0})
    assert r["max_ttlt_seconds(<=5.0s)"] is False


def test_perf_not_checked_when_none() -> None:
    r = _checks(_out(), {})
    assert not any("performance" in k for k in r)


def test_perf_not_parsed_when_no_timing() -> None:
    """When agent provides no timing, perf checks report 'not parsed'."""
    to = _out(timing=None)
    results = _check_results(to, {"max_total_seconds": 5.0})
    perf = [r for r in results if "performance" in r.dimension]
    assert len(perf) == 1
    assert perf[0].passed is False
    assert "not parsed" in perf[0].details


# ── turn_idx propagation ──


def test_turn_idx_propagated() -> None:
    to = _out(tool_calls=[])
    results = _check_results(to, {"tools_invoked": ["my_tool"]}, turn_idx=3)
    for r in results:
        assert r.turn_idx == 3


def test_turn_idx_zero() -> None:
    results = _check_results(_out(), {"no_errors": True}, turn_idx=0)
    assert all(r.turn_idx == 0 for r in results)


# ── timeout sentinel matches no_errors ──


def test_timeout_sentinel_caught() -> None:
    """Runner writes RuntimeError on timeout - must fail no_errors."""
    r = _checks(_out("RuntimeError: agent failed on turn 0 - timed out after 180s\n"), {"no_errors": True})
    assert r["no_errors"] is False


def test_crash_sentinel_caught() -> None:
    """Runner writes RuntimeError on pre-turn crash - must fail no_errors."""
    r = _checks(_out("RuntimeError: scenario setup failed - thread creation failed\n"), {"no_errors": True})
    assert r["no_errors"] is False


# ── cost/max_cost_usd ──


def test_max_cost_usd_within() -> None:
    to = _out(cost_usd=0.02)
    r = _checks(to, {"max_cost_usd": 0.05})
    assert r["max_cost_usd(<=$0.05)"] is True


def test_max_cost_usd_exceeded() -> None:
    to = _out(cost_usd=0.10)
    r = _checks(to, {"max_cost_usd": 0.05})
    assert r["max_cost_usd(<=$0.05)"] is False


def test_max_cost_usd_not_reported() -> None:
    to = _out(cost_usd=None)
    r = _checks(to, {"max_cost_usd": 0.05})
    assert r["max_cost_usd(<=$0.05)"] is False


def test_max_cost_not_checked_when_none() -> None:
    r = _checks(_out(), {})
    assert not any("cost" in k for k in r)


# ── execution/error_type_is ──


def test_error_type_is_match() -> None:
    to = _out(error_type="overloaded")
    r = _checks(to, {"error_type_is": "overloaded", "no_errors": False})
    assert r["error_type_is(overloaded)"] is True


def test_error_type_is_mismatch() -> None:
    to = _out(error_type="rate_limit")
    r = _checks(to, {"error_type_is": "overloaded", "no_errors": False})
    assert r["error_type_is(overloaded)"] is False


def test_error_type_is_none() -> None:
    to = _out(error_type=None)
    r = _checks(to, {"error_type_is": "overloaded", "no_errors": False})
    assert r["error_type_is(overloaded)"] is False


def test_error_type_is_case_insensitive() -> None:
    to = _out(error_type="Overloaded")
    r = _checks(to, {"error_type_is": "overloaded", "no_errors": False})
    assert r["error_type_is(overloaded)"] is True


# ── trajectory/tools_invoked_in_order ──


def test_tools_invoked_in_order_pass() -> None:
    to = _out(tool_sequence=["Read", "Edit", "Write"])
    r = _checks(to, {"tools_invoked_in_order": ["Read", "Write"]})
    assert r["tools_invoked_in_order(Read→Write)"] is True


def test_tools_invoked_in_order_exact() -> None:
    to = _out(tool_sequence=["Read", "Edit", "Write"])
    r = _checks(to, {"tools_invoked_in_order": ["Read", "Edit", "Write"]})
    assert r["tools_invoked_in_order(Read→Edit→Write)"] is True


def test_tools_invoked_in_order_fail() -> None:
    to = _out(tool_sequence=["Write", "Read"])
    r = _checks(to, {"tools_invoked_in_order": ["Read", "Write"]})
    assert r["tools_invoked_in_order(Read→Write)"] is False


def test_tools_invoked_in_order_falls_back_to_tool_calls() -> None:
    to = _out(tool_calls=[ToolCall(name="Read", call_id="c1"), ToolCall(name="Write", call_id="c2")])
    r = _checks(to, {"tools_invoked_in_order": ["Read", "Write"]})
    assert r["tools_invoked_in_order(Read→Write)"] is True


def test_tools_invoked_in_order_not_checked_when_empty() -> None:
    r = _checks(_out(), {})
    assert not any("tools_invoked_in_order" in k for k in r)


# ── trajectory/only_used_tools ──


def test_only_used_tools_pass() -> None:
    to = _out(tool_calls=[ToolCall(name="Read", call_id="c1"), ToolCall(name="Write", call_id="c2")])
    r = _checks(to, {"only_used_tools": ["Read", "Write", "Edit"]})
    assert r["only_used_tools(Edit,Read,Write)"] is True


def test_only_used_tools_violation() -> None:
    to = _out(tool_calls=[ToolCall(name="Read", call_id="c1"), ToolCall(name="Bash", call_id="c2")])
    r = _checks(to, {"only_used_tools": ["Read", "Write"]})
    assert r["only_used_tools(Read,Write)"] is False


def test_only_used_tools_empty_calls() -> None:
    to = _out(tool_calls=[])
    r = _checks(to, {"only_used_tools": ["Read"]})
    assert r["only_used_tools(Read)"] is True


# ── trajectory/forbidden_tools ──


def test_forbidden_tools_pass() -> None:
    to = _out(tool_calls=[ToolCall(name="Read", call_id="c1")])
    r = _checks(to, {"forbidden_tools": ["Edit", "Write"]})
    assert r["forbidden_tools(Edit,Write)"] is True


def test_forbidden_tools_violation() -> None:
    to = _out(tool_calls=[ToolCall(name="Read", call_id="c1"), ToolCall(name="Edit", call_id="c2")])
    r = _checks(to, {"forbidden_tools": ["Edit", "Write"]})
    assert r["forbidden_tools(Edit,Write)"] is False


def test_forbidden_tools_empty_calls() -> None:
    to = _out(tool_calls=[])
    r = _checks(to, {"forbidden_tools": ["Edit"]})
    assert r["forbidden_tools(Edit)"] is True


# ── trajectory/tool_args_contain ──


def test_tool_args_contain_match() -> None:
    to = _out(tool_calls=[ToolCall(name="Read", call_id="c1", args={"file_path": "src/entities.py"})])
    r = _checks(to, {"tool_args_contain": {"Read": {"file_path": "entities.py"}}})
    assert r["tool_args_contain(Read)"] is True


def test_tool_args_contain_no_match() -> None:
    to = _out(tool_calls=[ToolCall(name="Read", call_id="c1", args={"file_path": "src/other.py"})])
    r = _checks(to, {"tool_args_contain": {"Read": {"file_path": "entities.py"}}})
    assert r["tool_args_contain(Read)"] is False


def test_tool_args_contain_tool_not_invoked() -> None:
    to = _out(tool_calls=[ToolCall(name="Write", call_id="c1", args={})])
    r = _checks(to, {"tool_args_contain": {"Read": {"file_path": "entities.py"}}})
    assert r["tool_args_contain(Read)"] is False


def test_tool_args_contain_multiple_calls_any_match() -> None:
    to = _out(
        tool_calls=[
            ToolCall(name="Read", call_id="c1", args={"file_path": "wrong.py"}),
            ToolCall(name="Read", call_id="c2", args={"file_path": "src/entities.py"}),
        ]
    )
    r = _checks(to, {"tool_args_contain": {"Read": {"file_path": "entities.py"}}})
    assert r["tool_args_contain(Read)"] is True


# ── trajectory/has_thinking ──


def test_has_thinking_expected_and_present() -> None:
    to = _out(thinking_text="Let me think about this...")
    r = _checks(to, {"has_thinking": True})
    assert r["has_thinking"] is True


def test_has_thinking_expected_but_absent() -> None:
    to = _out(thinking_text=None)
    r = _checks(to, {"has_thinking": True})
    assert r["has_thinking"] is False


def test_has_thinking_not_expected_and_absent() -> None:
    to = _out(thinking_text=None)
    r = _checks(to, {"has_thinking": False})
    assert r["has_thinking"] is True


def test_has_thinking_not_expected_but_present() -> None:
    to = _out(thinking_text="Unexpected thinking")
    r = _checks(to, {"has_thinking": False})
    assert r["has_thinking"] is False


def test_has_thinking_empty_string_treated_as_absent() -> None:
    to = _out(thinking_text="   ")
    r = _checks(to, {"has_thinking": True})
    assert r["has_thinking"] is False


def test_has_thinking_not_checked_when_none() -> None:
    r = _checks(_out(), {})
    assert "has_thinking" not in r


# ── state checks ──


def _state_checks(output: TurnOutput, state_expect: dict) -> dict[str, bool]:
    se = StateExpectation(**state_expect)
    results = check_state(0, output, se)
    return {r.check: r.passed for r in results}


def test_state_file_exists_pass() -> None:
    to = _out(workspace_files={"src/fix.py": "def fixed_function(): pass"})
    r = _state_checks(to, {"files_exist": ["src/fix.py"]})
    assert r["file_exists(src/fix.py)"] is True


def test_state_file_exists_fail() -> None:
    to = _out(workspace_files={"src/fix.py": None})
    r = _state_checks(to, {"files_exist": ["src/fix.py"]})
    assert r["file_exists(src/fix.py)"] is False


def test_state_file_contains_pass() -> None:
    to = _out(workspace_files={"src/fix.py": "def fixed_function(): pass"})
    r = _state_checks(to, {"files_contain": {"src/fix.py": "fixed_function"}})
    assert r["file_contains(src/fix.py)"] is True


def test_state_file_contains_fail_wrong_content() -> None:
    to = _out(workspace_files={"src/fix.py": "def broken(): pass"})
    r = _state_checks(to, {"files_contain": {"src/fix.py": "fixed_function"}})
    assert r["file_contains(src/fix.py)"] is False


def test_state_file_contains_fail_missing_file() -> None:
    to = _out(workspace_files={"src/fix.py": None})
    r = _state_checks(to, {"files_contain": {"src/fix.py": "anything"}})
    assert r["file_contains(src/fix.py)"] is False


def test_state_file_not_exists_pass() -> None:
    to = _out(workspace_files={"temp.txt": None})
    r = _state_checks(to, {"files_not_exist": ["temp.txt"]})
    assert r["file_not_exists(temp.txt)"] is True


def test_state_file_not_exists_fail() -> None:
    to = _out(workspace_files={"temp.txt": "still here"})
    r = _state_checks(to, {"files_not_exist": ["temp.txt"]})
    assert r["file_not_exists(temp.txt)"] is False


# ── trajectory/skills_invoked ──


def test_skill_invoked_claude_code_skill_tool() -> None:
    """Claude Code: first-class Skill tool with args.skill matching."""
    to = _out(tool_calls=[ToolCall(name="Skill", call_id="c1", args={"skill": "plan-architecture"})])
    r = _checks(to, {"skills_invoked": ["plan-architecture"]})
    assert r["skill_invoked(plan-architecture)"] is True


def test_skill_invoked_claude_code_wrong_skill() -> None:
    """Claude Code: Skill tool called but with a different skill name."""
    to = _out(tool_calls=[ToolCall(name="Skill", call_id="c1", args={"skill": "deploy-k8s"})])
    r = _checks(to, {"skills_invoked": ["plan-architecture"]})
    assert r["skill_invoked(plan-architecture)"] is False


def test_skill_invoked_cursor_read_skill_file() -> None:
    """Cursor: Read tool on a SKILL.md path."""
    to = _out(
        tool_calls=[
            ToolCall(name="Read", call_id="c1", args={"path": "/home/user/.cursor/skills/plan-architecture/SKILL.md"})
        ]
    )
    r = _checks(to, {"skills_invoked": ["plan-architecture"]})
    assert r["skill_invoked(plan-architecture)"] is True


def test_skill_invoked_gemini_read_file() -> None:
    """Gemini: read_file tool with file_path containing skill path."""
    to = _out(
        tool_calls=[
            ToolCall(name="read_file", call_id="c1", args={"file_path": ".cursor/skills/plan-architecture/SKILL.md"})
        ]
    )
    r = _checks(to, {"skills_invoked": ["plan-architecture"]})
    assert r["skill_invoked(plan-architecture)"] is True


def test_skill_invoked_codex_shell_cat() -> None:
    """Codex: shell tool reading a skill file via cat."""
    to = _out(
        tool_calls=[
            ToolCall(
                name="shell", call_id="c1", args={"command": "cat /home/user/.codex/skills/plan-architecture/SKILL.md"}
            )
        ]
    )
    r = _checks(to, {"skills_invoked": ["plan-architecture"]})
    assert r["skill_invoked(plan-architecture)"] is True


def test_skill_invoked_codex_shell_list_arg() -> None:
    """Codex: shell tool with command as list."""
    to = _out(
        tool_calls=[
            ToolCall(
                name="shell", call_id="c1", args={"command": ["cat", "/home/.codex/skills/plan-architecture/SKILL.md"]}
            )
        ]
    )
    r = _checks(to, {"skills_invoked": ["plan-architecture"]})
    assert r["skill_invoked(plan-architecture)"] is True


def test_skill_invoked_cli_fallback() -> None:
    """Falls back to raw CLI text when tool_calls don't capture the skill."""
    to = _out(raw_cli="Reading skills/plan-architecture/SKILL.md\nThe architecture spec covers...")
    r = _checks(to, {"skills_invoked": ["plan-architecture"]})
    assert r["skill_invoked(plan-architecture)"] is True


def test_skill_invoked_not_found() -> None:
    """Skill not invoked by any mechanism."""
    to = _out(
        raw_cli="some output",
        tool_calls=[ToolCall(name="Read", call_id="c1", args={"path": "/src/main.py"})],
    )
    r = _checks(to, {"skills_invoked": ["plan-architecture"]})
    assert r["skill_invoked(plan-architecture)"] is False


def test_skill_invoked_case_insensitive_skill_tool() -> None:
    """Claude Code Skill tool: case-insensitive match on skill name."""
    to = _out(tool_calls=[ToolCall(name="Skill", call_id="c1", args={"skill": "Plan-Architecture"})])
    r = _checks(to, {"skills_invoked": ["plan-architecture"]})
    assert r["skill_invoked(plan-architecture)"] is True


def test_skill_invoked_plugin_prefixed_skill_tool() -> None:
    """Claude Code: plugin-bundled skills are emitted with `<plugin>:<skill>`
    in args.skill. Tier-1 matcher must accept the suffix so the scenario can
    still write `skills_invoked: ["<skill>"]` without leaking plugin internals."""
    to = _out(tool_calls=[ToolCall(name="Skill", call_id="c1", args={"skill": "orders-pack:processing-watch"})])
    r = _checks(to, {"skills_invoked": ["processing-watch"]})
    assert r["skill_invoked(processing-watch)"] is True


def test_skill_invoked_plugin_prefixed_does_not_match_unrelated_skill() -> None:
    """A plugin-prefixed value must not match a different skill name that
    happens to share a substring with the plugin."""
    to = _out(tool_calls=[ToolCall(name="Skill", call_id="c1", args={"skill": "orders-pack:processing-watch"})])
    r = _checks(to, {"skills_invoked": ["orders-pack"]})
    # "orders-pack" is the plugin namespace, not a skill name.
    assert r["skill_invoked(orders-pack)"] is False


def test_skill_invoked_multiple_skills() -> None:
    """Assert multiple skills in the same turn."""
    to = _out(
        tool_calls=[
            ToolCall(name="Skill", call_id="c1", args={"skill": "plan-architecture"}),
            ToolCall(name="Read", call_id="c2", args={"path": "/skills/deploy-k8s/SKILL.md"}),
        ]
    )
    r = _checks(to, {"skills_invoked": ["plan-architecture", "deploy-k8s"]})
    assert r["skill_invoked(plan-architecture)"] is True
    assert r["skill_invoked(deploy-k8s)"] is True


def test_skill_invoked_not_checked_when_empty() -> None:
    r = _checks(_out("anything"), {})
    assert not any("skill_invoked" in k for k in r)


def test_state_no_checks_when_empty() -> None:
    to = _out()
    r = _state_checks(to, {})
    assert r == {}


def test_state_multiple_checks_combined() -> None:
    to = _out(
        workspace_files={
            "created.py": "new code",
            "old.txt": None,
        }
    )
    r = _state_checks(
        to,
        {
            "files_exist": ["created.py"],
            "files_not_exist": ["old.txt"],
            "files_contain": {"created.py": "new code"},
        },
    )
    assert all(r.values())
