# (c) JFrog Ltd. (2026)

"""Unit tests for ``_render_turn_message`` placeholder rendering."""

from __future__ import annotations

import pytest

from belt.entities import TurnOutput
from belt.errors import ScenarioError
from belt.runner.orchestrator import _render_turn_message


def _output(reply_text: str = "", git_diff: str | None = None, tools: list[str] | None = None) -> TurnOutput:
    return TurnOutput(
        raw_cli="",
        reply_text=reply_text,
        git_diff=git_diff,
        tool_sequence=tools or [],
    )


class TestPassthrough:
    def test_no_braces_returns_unchanged(self):
        assert _render_turn_message("Plain message", []) == "Plain message"

    def test_unmatched_braces_left_alone(self):
        # ``{{ ... }}`` that doesn't match the placeholder shape (e.g. JSON
        # examples in the message) must pass through untouched.
        msg = "Look at this JSON: {{ key: 'value' }} -- still fine."
        assert _render_turn_message(msg, [_output(reply_text="x")]) == msg

    def test_empty_template(self):
        assert _render_turn_message("", []) == ""


class TestPrev:
    def test_prev_reply_text(self):
        prior = [_output(reply_text="hello world")]
        assert _render_turn_message("Echo: {{prev.reply_text}}", prior) == "Echo: hello world"

    def test_prev_git_diff(self):
        prior = [_output(git_diff="diff --git a/x b/x\n+y\n")]
        rendered = _render_turn_message("Diff:\n{{prev.git_diff}}", prior)
        assert rendered == "Diff:\ndiff --git a/x b/x\n+y\n"

    def test_prev_git_diff_none_renders_empty(self):
        prior = [_output(git_diff=None)]
        assert _render_turn_message("[{{prev.git_diff}}]", prior) == "[]"

    def test_prev_tool_sequence_comma_joined(self):
        prior = [_output(tools=["Read", "Edit", "Bash"])]
        assert _render_turn_message("Used: {{prev.tool_sequence}}", prior) == "Used: Read, Edit, Bash"

    def test_prev_tool_sequence_empty(self):
        prior = [_output(tools=[])]
        assert _render_turn_message("Used: [{{prev.tool_sequence}}]", prior) == "Used: []"

    def test_prev_on_turn_zero_raises(self):
        with pytest.raises(ScenarioError, match="turn 0"):
            _render_turn_message("Echo: {{prev.reply_text}}", [])

    def test_whitespace_inside_braces_tolerated(self):
        prior = [_output(reply_text="hi")]
        assert _render_turn_message("{{ prev.reply_text }}", prior) == "hi"


class TestExplicitTurnIndex:
    def test_turn_zero_from_later_turn(self):
        prior = [
            _output(reply_text="first"),
            _output(reply_text="second"),
        ]
        rendered = _render_turn_message("Recall turn 0: {{turn_0.reply_text}}", prior)
        assert rendered == "Recall turn 0: first"

    def test_turn_one_field(self):
        prior = [
            _output(reply_text="a"),
            _output(reply_text="b"),
        ]
        assert _render_turn_message("{{turn_1.reply_text}}", prior) == "b"

    def test_future_turn_raises(self):
        prior = [_output(reply_text="a"), _output(reply_text="b")]
        with pytest.raises(ScenarioError, match="turn 5"):
            _render_turn_message("{{turn_5.reply_text}}", prior)

    def test_turn_zero_from_turn_zero_raises(self):
        # ``turn_0`` referenced from turn 0 itself: list is empty, so it's a
        # future reference relative to current turn.
        with pytest.raises(ScenarioError, match="turn 0"):
            _render_turn_message("{{turn_0.reply_text}}", [])


class TestUnsupportedField:
    def test_typo_in_prev_field_raises(self):
        with pytest.raises(ScenarioError, match="supported fields"):
            _render_turn_message("{{prev.reply}}", [_output(reply_text="x")])

    def test_unsupported_field_in_turn_n_raises(self):
        with pytest.raises(ScenarioError, match="cost_usd"):
            _render_turn_message("{{turn_0.cost_usd}}", [_output(reply_text="x")])

    def test_error_message_lists_supported_fields(self):
        with pytest.raises(ScenarioError) as excinfo:
            _render_turn_message("{{prev.bogus}}", [_output(reply_text="x")])
        msg = str(excinfo.value)
        assert "reply_text" in msg
        assert "git_diff" in msg
        assert "tool_sequence" in msg


class TestMultiplePlaceholders:
    def test_multiple_placeholders_in_one_message(self):
        prior = [
            _output(reply_text="ans1", tools=["Read"]),
            _output(reply_text="ans2", tools=["Edit", "Write"]),
        ]
        template = (
            "Turn 0 said {{turn_0.reply_text}} (used {{turn_0.tool_sequence}}); "
            "previous turn said {{prev.reply_text}}."
        )
        assert _render_turn_message(template, prior) == ("Turn 0 said ans1 (used Read); previous turn said ans2.")

    def test_repeated_same_placeholder(self):
        prior = [_output(reply_text="x")]
        assert _render_turn_message("{{prev.reply_text}}/{{prev.reply_text}}", prior) == "x/x"
