# (c) JFrog Ltd. (2026)

"""Stream-rendering matrix: every built-in agent x every progress mode.

This file is the single regression surface for the contract that every agent's
NDJSON output renders correctly into the live-output panel. It covers:

  - **Trust contract:** agent ``parse_stream_event`` overrides return plain
    text (``summary_is_markup=False``); only framework-built ``result`` events
    carry trusted Rich markup.
  - **Six event types per agent:** user prompt, tool call, tool result /
    attachment, reply text, result event, error event.
  - **Three progress modes:** ``rich``, ``plain``, ``live`` - each must
    surface every event without dropping content or leaking raw markup.
  - **No silent drops:** if an agent emits an event we know about, it must
    parse to a non-None ``StreamEvent``.

For each agent we record a small set of realistic NDJSON event payloads (the
shapes the agent CLI actually produces) and assert that the parser surfaces
them with the expected icon and a summary that contains the meaningful
content. Truncation and exact summary formatting are exercised in
``tests/commands/test_watch.py``; this file focuses on the matrix and the
trust contract.
"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from belt._safe import rich_safe
from belt.agent.base import BaseAgentAdapter
from belt.agent.claude_code import ClaudeCodeAgentAdapter
from belt.agent.codex import CodexAgentAdapter
from belt.agent.copilot import CopilotAgentAdapter
from belt.agent.cursor import CursorAgentAdapter
from belt.agent.gemini import GeminiAgentAdapter
from belt.agent.goose import GooseAgentAdapter
from belt.agent.opencode import OpenCodeAgentAdapter
from belt.commands.watch import StreamEvent, StreamParser, _print_event

# ──────────────────────────────────────────────────────────────────────────
# Per-agent event fixtures
#
# Each entry is ``(label, event_dict, expected_icon, content_substring)``.
# The parser must produce a StreamEvent whose icon equals ``expected_icon``
# and whose summary contains ``content_substring``.
#
# Some agents do not emit a discrete event for every category. Where that is
# the case we still cover the category via the generic StreamParser path
# using a synthetic event the agent's stream is known to be compatible with.
# ──────────────────────────────────────────────────────────────────────────


def _claude_code_events():
    return [
        ("user_prompt", {"type": "user_input", "message": "Read the project and summarise"}, "👤", "Read the project"),
        (
            "tool_call",
            {"type": "assistant", "content": [{"type": "tool_use", "name": "Read", "input": {"path": "src/x.py"}}]},
            "🔧",
            "Read",
        ),
        (
            "tool_result",
            {"type": "user", "tool_use_result": {"content": "file contents here", "numLines": 42}},
            "📎",
            "42 lines",
        ),
        (
            "reply_text",
            {"type": "assistant", "content": [{"type": "text", "text": "Found the bug in auth.py"}]},
            "💬",
            "auth.py",
        ),
        ("result", {"type": "result", "total_cost_usd": 0.4173, "duration_ms": 71400}, "✅", "$0.4173"),
        ("error", {"type": "result", "is_error": True, "duration_ms": 5000}, "❌", "ERROR"),
    ]


def _codex_events():
    return [
        ("user_prompt", {"type": "user_input", "message": "Find the SQL injection"}, "👤", "SQL injection"),
        (
            "tool_call",
            {"type": "function_call", "name": "search", "arguments": {"query": "exec("}},
            "🔧",
            "search",
        ),
        # Codex emits tool results inline; covered by the generic user.tool_use_result shape.
        (
            "tool_result",
            {"type": "user", "tool_use_result": {"content": "match found", "numFiles": 3}},
            "📎",
            "3 files",
        ),
        (
            "reply_text",
            {"type": "assistant", "content": [{"type": "text", "text": "Patched in bookService.ts"}]},
            "💬",
            "bookService.ts",
        ),
        ("result", {"type": "result", "total_cost_usd": 0.0312, "duration_ms": 5400}, "✅", "$0.0312"),
        ("error", {"type": "result", "is_error": True, "duration_ms": 1200}, "❌", "ERROR"),
    ]


def _copilot_events():
    return [
        ("user_prompt", {"type": "user_input", "message": "Add input validation to /authors"}, "👤", "validation"),
        (
            "tool_call",
            {"type": "tool.execution_start", "data": {"toolName": "shell", "arguments": {"cmd": "ls"}}},
            "🔧",
            "shell",
        ),
        # Copilot's stream does not surface tool results as a distinct event in
        # the harness; the generic shape is exercised in claude_code/codex
        # cells. Cover the harness fallback explicitly.
        (
            "tool_result",
            {"type": "user", "tool_use_result": {"content": "ok", "numLines": 1}},
            "📎",
            "1 lines",
        ),
        (
            "reply_text",
            {"type": "assistant.message", "data": {"content": "Validation middleware added"}},
            "💬",
            "Validation middleware",
        ),
        ("result", {"type": "result", "total_cost_usd": 0.0850, "duration_ms": 12100}, "✅", "$0.0850"),
        ("error", {"type": "result", "is_error": True, "duration_ms": 800}, "❌", "ERROR"),
    ]


def _cursor_events():
    return [
        ("user_prompt", {"type": "user_input", "message": "Fix the off-by-one bug"}, "👤", "off-by-one"),
        (
            "tool_call",
            {
                "type": "tool_call",
                "subtype": "started",
                "tool_call": {"readToolCall": {"args": {"path": "/tmp/x.py"}}},
            },
            "🔧",
            "Read",
        ),
        # Cursor renders a completed tool call as ✅ (success) or ❌ (error)
        # via ``_render_cursor_tool_result`` - it does not use the generic
        # ``📎`` icon because the result is encoded inline in ``tool_call``.
        (
            "tool_result",
            {
                "type": "tool_call",
                "subtype": "completed",
                "tool_call": {
                    "readToolCall": {
                        "args": {"path": "/tmp/x.py"},
                        "result": {"success": {"contents": "data"}},
                    }
                },
            },
            "✅",
            "Read",
        ),
        (
            "reply_text",
            {"type": "assistant", "content": [{"type": "text", "text": "Computed offset is now (page-1)*limit"}]},
            "💬",
            "offset",
        ),
        ("result", {"type": "result", "total_cost_usd": 0.0987, "duration_ms": 8200}, "✅", "$0.0987"),
        ("error", {"type": "result", "is_error": True, "duration_ms": 600}, "❌", "ERROR"),
    ]


def _gemini_events():
    return [
        ("user_prompt", {"type": "user_input", "message": "Explain the architecture"}, "👤", "architecture"),
        (
            "tool_call",
            {
                "type": "message",
                "role": "model",
                "content": [{"type": "functionCall", "name": "readFile", "args": {"path": "src/index.ts"}}],
            },
            "🔧",
            "readFile",
        ),
        (
            "tool_result",
            {"type": "user", "tool_use_result": {"content": "result", "numLines": 5}},
            "📎",
            "5 lines",
        ),
        (
            "reply_text",
            {
                "type": "message",
                "role": "model",
                "content": [{"type": "text", "text": "Layered Express REST API"}],
            },
            "💬",
            "Express",
        ),
        ("result", {"type": "result", "status": "success", "stats": {"duration_ms": 3200}}, "✅", "3.2s"),
        ("error", {"type": "result", "status": "error", "error": {"type": "timeout"}}, "❌", None),
    ]


def _goose_events():
    return [
        ("user_prompt", {"type": "user_input", "message": "Add tests for /authors"}, "👤", "authors"),
        (
            "tool_call",
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolRequest",
                            "toolCall": {"value": {"name": "developer__shell", "arguments": {"cmd": "ls"}}},
                        }
                    ],
                },
            },
            "🔧",
            "developer__shell",
        ),
        # Goose suppresses toolResponse explicitly; a discrete tool result is
        # exercised via the generic harness path.
        (
            "tool_result",
            {"type": "user", "tool_use_result": {"content": "ok", "numFiles": 2}},
            "📎",
            "2 files",
        ),
        (
            "reply_text",
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Added authors.test.ts"}],
                },
            },
            "💬",
            "authors.test.ts",
        ),
        ("result", {"type": "result", "total_cost_usd": 0.1500, "duration_ms": 25000}, "✅", "$0.1500"),
        ("error", {"type": "result", "is_error": True, "duration_ms": 2000}, "❌", "ERROR"),
    ]


def _opencode_events():
    return [
        ("user_prompt", {"type": "user_input", "message": "Run the security audit"}, "👤", "security audit"),
        (
            "tool_call",
            {"type": "tool_use", "part": {"tool": "bash", "state": {"input": {"cmd": "ls"}}}},
            "🔧",
            "bash",
        ),
        (
            "tool_result",
            {"type": "user", "tool_use_result": {"content": "output", "numLines": 7}},
            "📎",
            "7 lines",
        ),
        ("reply_text", {"type": "text", "part": {"text": "Found 3 vulnerabilities"}}, "💬", "vulnerabilities"),
        # opencode's step_finish carries cost; it returns plain "done ($0.42)".
        (
            "result",
            {"type": "step_finish", "part": {"reason": "stop", "cost": 0.4173}},
            "✅",
            "$0.4173",
        ),
        (
            "error",
            {"type": "error", "error": {"name": "Timeout", "data": {"message": "exceeded 60s"}}},
            "❌",
            "Timeout",
        ),
    ]


AGENT_FIXTURES = [
    ("claude-code", ClaudeCodeAgentAdapter, _claude_code_events()),
    ("codex", CodexAgentAdapter, _codex_events()),
    ("copilot", CopilotAgentAdapter, _copilot_events()),
    ("cursor", CursorAgentAdapter, _cursor_events()),
    ("gemini", GeminiAgentAdapter, _gemini_events()),
    ("goose", GooseAgentAdapter, _goose_events()),
    ("opencode", OpenCodeAgentAdapter, _opencode_events()),
]


def _flatten_cases():
    """Flatten the matrix into ``(agent, event_label, event, icon, substr)`` tuples."""
    for agent_name, agent_cls, events in AGENT_FIXTURES:
        for label, event, icon, substr in events:
            yield agent_name, agent_cls, label, event, icon, substr


# ── §1 trust contract ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("agent_name", "agent_cls", "label", "event", "expected_icon", "expected_substr"),
    list(_flatten_cases()),
    ids=[f"{a}-{lbl}" for a, _, lbl, _, _, _ in _flatten_cases()],
)
def test_event_renders_to_expected_shape(agent_name, agent_cls, label, event, expected_icon, expected_substr):
    """Every agent x event-type combination must produce a non-None StreamEvent
    with the expected icon and (where asserted) content."""
    parser = StreamParser(agent_cls=agent_cls)
    line = json.dumps(event)
    result = parser.parse_line(line)

    assert result is not None, f"agent={agent_name} event={label}: parser dropped a known event"
    assert result.icon == expected_icon, (
        f"agent={agent_name} event={label}: expected icon {expected_icon!r}, "
        f"got {result.icon!r} (summary={result.summary!r})"
    )
    if expected_substr is not None:
        assert expected_substr in result.summary, (
            f"agent={agent_name} event={label}: summary {result.summary!r} "
            f"missing expected substring {expected_substr!r}"
        )


@pytest.mark.parametrize(
    ("agent_name", "agent_cls"),
    [(name, cls) for name, cls, _ in AGENT_FIXTURES],
    ids=[name for name, _, _ in AGENT_FIXTURES],
)
def test_only_result_events_carry_trusted_markup(agent_name, agent_cls):
    """The trust contract: only ``_render_result`` events have
    ``summary_is_markup=True``. All other events (including agent overrides
    for tool-call, reply-text, etc.) are plain text."""
    parser = StreamParser(agent_cls=agent_cls)
    fixtures_for_agent = next(fixtures for name, _, fixtures in AGENT_FIXTURES if name == agent_name)
    for label, event, icon, _ in fixtures_for_agent:
        parsed = parser.parse_line(json.dumps(event))
        assert parsed is not None
        if icon in ("✅", "❌") and label in ("result", "error"):
            # The framework ``_render_result`` pathway emits markup.
            # Some agents (like opencode) handle the result themselves
            # via overrides and emit plain text - both are valid as long
            # as the contract is honoured (markup -> True; plain -> False).
            if "[green]" in parsed.summary or "[/green]" in parsed.summary:
                assert (
                    parsed.summary_is_markup is True
                ), f"agent={agent_name} event={label}: summary contains markup but is not labelled trusted"
        else:
            assert parsed.summary_is_markup is False, (
                f"agent={agent_name} event={label}: non-result events must be plain text " f"(summary_is_markup=False)"
            )


# ── §2 progress-mode rendering ──────────────────────────────────────────


def _capture_rich(events: list[StreamEvent]) -> str:
    """Render events through ``_print_event`` (mirrors `rich` and `plain` modes)."""
    console = Console(file=StringIO(), width=200, force_terminal=True, no_color=True)
    for ev in events:
        _print_event(console, "scen-X", ev)
    return console.file.getvalue()


def _capture_live(events: list[StreamEvent]) -> str:
    """Render events through ``LiveProgress._build_stream_panel`` (mirrors `live` mode)."""
    from belt.progress import LiveProgress

    lp = LiveProgress(
        run_dir=Path("."),
        console=Console(file=StringIO(), width=200, force_terminal=True, no_color=True),
        max_lines=70,
    )
    lp._scenario_order = ["scen-X"]
    lp._scenario_pin_indices = {"scen-X": set()}
    formatted: list[str] = []
    for ev in events:
        payload = ev.summary if ev.summary_is_markup else rich_safe(ev.summary)
        formatted.append(f"  {ev.icon} {payload}")
    lp._scenario_streams = {"scen-X": formatted}

    panel = lp._build_stream_panel()
    out = Console(file=StringIO(), width=200, force_terminal=True, no_color=True)
    out.print(panel)
    return out.file.getvalue()


@pytest.mark.parametrize(
    ("agent_name", "agent_cls"),
    [(name, cls) for name, cls, _ in AGENT_FIXTURES],
    ids=[name for name, _, _ in AGENT_FIXTURES],
)
def test_rich_mode_renders_every_event_type(agent_name, agent_cls):
    """In ``--progress rich`` / ``plain`` mode (which both go through
    ``_print_event``), every fixture event must render without leaking raw
    markup tokens."""
    parser = StreamParser(agent_cls=agent_cls)
    fixtures = next(evs for name, _, evs in AGENT_FIXTURES if name == agent_name)
    events = [parser.parse_line(json.dumps(ev)) for _, ev, _, _ in fixtures]
    assert all(e is not None for e in events), f"agent={agent_name}: dropped event(s)"

    out = _capture_rich(events)

    # No trusted-markup leak.
    assert "[green]" not in out, f"agent={agent_name}: cost markup leaked: {out!r}"
    assert "[/green]" not in out, f"agent={agent_name}: cost markup leaked: {out!r}"


@pytest.mark.parametrize(
    ("agent_name", "agent_cls"),
    [(name, cls) for name, cls, _ in AGENT_FIXTURES],
    ids=[name for name, _, _ in AGENT_FIXTURES],
)
def test_live_mode_renders_every_event_type(agent_name, agent_cls):
    """In ``--progress live`` mode (panel renderer), every fixture event must
    render without leaking trusted markup as literal text."""
    parser = StreamParser(agent_cls=agent_cls)
    fixtures = next(evs for name, _, evs in AGENT_FIXTURES if name == agent_name)
    events = [parser.parse_line(json.dumps(ev)) for _, ev, _, _ in fixtures]
    assert all(e is not None for e in events), f"agent={agent_name}: dropped event(s)"

    out = _capture_live(events)

    # Trusted markup must be rendered (no literal ``[green]`` substring).
    assert "[green]" not in out, f"agent={agent_name}: cost markup leaked: {out!r}"
    assert "[/green]" not in out, f"agent={agent_name}: cost markup leaked: {out!r}"
    # Scenario label header is present.
    assert "scen-X" in out


# ── §3 base-default sufficiency ─────────────────────────────────────────


def test_base_default_parse_stream_event_returns_none():
    """``BaseAgentAdapter.parse_stream_event`` is a no-op; a subclass that does
    not override it falls through to the generic ``StreamParser`` rendering.

    This is intentional: agents whose NDJSON shapes match the generic event
    types (``user_input``, ``tool_use``, ``function_call``, ``assistant``,
    ``message``, ``result``, ``user``, ``system``) need no override at all."""
    assert BaseAgentAdapter.parse_stream_event({"type": "anything"}) is None


@pytest.mark.parametrize(
    "agent_cls",
    [ClaudeCodeAgentAdapter, CodexAgentAdapter, GeminiAgentAdapter],
    ids=["claude-code", "codex", "gemini"],
)
def test_base_default_agents_render_via_generic(agent_cls):
    """The 3 agents that use the base default for ``parse_stream_event``
    still render their events correctly via the generic StreamParser path -
    they do not need a per-agent override."""
    parser = StreamParser(agent_cls=agent_cls)
    # Agent's parse_stream_event returns None -> generic path picks up.
    event = json.dumps({"type": "result", "total_cost_usd": 0.05, "duration_ms": 1000})
    result = parser.parse_line(event)
    assert result is not None
    assert result.icon == "✅"
    assert "$0.0500" in result.summary
    assert result.summary_is_markup is True
