# (c) JFrog Ltd. (2026)

"""Tests for ClaudeCodeAgentAdapter - NDJSON parsing, subprocess execution, session management,
multi-turn, timeout handling, error fallback, scoring strategy, display info, cli options,
streaming timing, thinking blocks, tool sequence, error_type, cost."""

from __future__ import annotations

import json
import subprocess
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from belt.agent.base import AgentNotAvailableError
from belt.agent.claude_code import ClaudeCodeAgentAdapter
from belt.entities import AgentConfig, GroupConfig
from belt.parser.ndjson import parse_ndjson


def _ndjson(*events: dict) -> str:
    return "\n".join(json.dumps(e) for e in events)


def _config() -> AgentConfig:
    return AgentConfig(
        group_config=GroupConfig(agent="claude-code"),
        scenario_name="test",
    )


def _mock_popen(stdout_text: str, returncode: int = 0, stderr_text: str = ""):
    """Create a mock Popen that yields stdout lines and has expected attributes."""
    mock = MagicMock()
    mock.stdout = StringIO(stdout_text)
    mock.stderr = StringIO(stderr_text)
    mock.returncode = returncode
    mock.wait = MagicMock()
    mock.pid = 99999
    return mock


# ── NDJSON parsing ──


class TestParseNdjson:
    def test_empty_string(self):
        assert parse_ndjson("") == []

    def test_valid_events(self):
        raw = _ndjson({"type": "a"}, {"type": "b"})
        assert len(parse_ndjson(raw)) == 2

    def test_skips_invalid_lines(self):
        raw = '{"type": "a"}\nnot json\n{"type": "b"}'
        result = parse_ndjson(raw)
        assert len(result) == 2

    def test_skips_blank_lines(self):
        raw = '{"type": "a"}\n\n\n{"type": "b"}\n'
        assert len(parse_ndjson(raw)) == 2

    def test_handles_whitespace_only_lines(self):
        raw = '{"type": "a"}\n   \n\t\n{"type": "b"}'
        assert len(parse_ndjson(raw)) == 2


# ── check_available ──


class TestCheckAvailable:
    def test_available_when_on_path(self):
        with patch("shutil.which", return_value="/usr/local/bin/claude"):
            ClaudeCodeAgentAdapter.check_available()

    def test_not_available_when_missing(self):
        with patch("shutil.which", return_value=None):
            with pytest.raises(AgentNotAvailableError, match="claude CLI not found"):
                ClaudeCodeAgentAdapter.check_available()

    def test_error_includes_install_suggestion(self):
        with patch("shutil.which", return_value=None):
            with pytest.raises(AgentNotAvailableError) as exc_info:
                ClaudeCodeAgentAdapter.check_available()
            assert "npm install" in str(exc_info.value)
            assert "@anthropic-ai/claude-code" in str(exc_info.value)


class TestAuthSignals:
    """Doctor renders these - bug here = silently wrong UX in `belt doctor`."""

    def test_declares_anthropic_api_key(self):
        assert "ANTHROPIC_API_KEY" in ClaudeCodeAgentAdapter.CREDENTIAL_ENV

    def test_declares_claude_credential_paths(self):
        # Both ~/.claude.json (current) and ~/.claude/ (forward-compat for an
        # announced upstream move) - first existing wins.
        paths = [str(p) for p in ClaudeCodeAgentAdapter.CREDENTIAL_PATHS]
        assert any(".claude.json" in p for p in paths)
        assert any(p.endswith(".claude") for p in paths)

    def test_env_signal_detected(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        signals = ClaudeCodeAgentAdapter.auth_signals()
        assert "env ANTHROPIC_API_KEY" in signals

    def test_no_signal_when_neither_env_nor_path(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        # Re-evaluate on a fresh class with paths bound to tmp_path
        # (simpler: just call _detect_auth_signals directly)
        from belt.agent.base import _detect_auth_signals

        sigs = _detect_auth_signals(("ANTHROPIC_API_KEY",), (tmp_path / ".claude.json", tmp_path / ".claude"))
        assert sigs == []


# ── execute (subprocess mocking) ──


class TestExecute:
    def test_successful_execution(self):
        agent = ClaudeCodeAgentAdapter()
        agent.setup(_config())
        expected_output = _ndjson(
            {"type": "assistant", "content": [{"type": "text", "text": "Hello"}]},
            {"type": "result", "session_id": "s1", "is_error": False},
        )
        mock = _mock_popen(expected_output)
        with patch("subprocess.Popen", return_value=mock) as mock_popen:
            raw = agent.execute("say hello", [])
            assert "Hello" in raw
            args = mock_popen.call_args[0][0]
            assert args[0] == "claude"
            assert "-p" in args
            assert "--verbose" in args
            assert "--output-format" in args
            assert "stream-json" in args
            assert args[-1] == "say hello"

    def test_nonzero_exit_appends_stderr(self):
        agent = ClaudeCodeAgentAdapter()
        agent.setup(_config())
        mock = _mock_popen('{"type": "result"}\n', returncode=1, stderr_text="connection error")
        with patch("subprocess.Popen", return_value=mock):
            raw = agent.execute("fail", [])
            assert "connection error" in raw

    def test_timeout_raises(self):
        agent = ClaudeCodeAgentAdapter()
        agent.setup(_config())
        mock = _mock_popen("")
        mock.wait.side_effect = subprocess.TimeoutExpired("claude", 300)
        mock.kill = MagicMock()
        with patch("subprocess.Popen", return_value=mock):
            with pytest.raises(subprocess.TimeoutExpired):
                agent.execute("slow prompt", [])

    def test_passes_extra_flags(self):
        agent = ClaudeCodeAgentAdapter()
        agent.setup(_config())
        mock = _mock_popen('{"type": "result", "is_error": false}\n')
        with patch("subprocess.Popen", return_value=mock) as mock_popen:
            agent.execute("test", ["--max-tokens", "1000"])
            args = mock_popen.call_args[0][0]
            assert "--max-tokens" in args
            assert "1000" in args

    def test_resume_with_session_id(self):
        agent = ClaudeCodeAgentAdapter()
        agent.setup(_config())
        agent._session_id = "session-xyz"
        mock = _mock_popen('{"type": "result", "is_error": false}\n')
        with patch("subprocess.Popen", return_value=mock) as mock_popen:
            agent.execute("follow up", [])
            args = mock_popen.call_args[0][0]
            assert "--resume" in args
            idx = args.index("--resume")
            assert args[idx + 1] == "session-xyz"

    def test_no_resume_on_first_turn(self):
        agent = ClaudeCodeAgentAdapter()
        agent.setup(_config())
        mock = _mock_popen('{"type": "result", "is_error": false}\n')
        with patch("subprocess.Popen", return_value=mock) as mock_popen:
            agent.execute("first message", [])
            args = mock_popen.call_args[0][0]
            assert "--resume" not in args

    def test_flags_control_behavior_not_agent(self):
        """Design principle: scenario flags control behavior, agent passes through."""
        agent = ClaudeCodeAgentAdapter()
        agent.setup(_config())
        mock = _mock_popen('{"type": "result", "is_error": false}\n')
        with patch("subprocess.Popen", return_value=mock) as mock_popen:
            agent.execute("test", ["--allowedTools", "Read,Write", "--model", "claude-opus-4"])
            args = mock_popen.call_args[0][0]
            assert "--allowedTools" in args
            assert "--model" in args
            assert "claude-opus-4" in args


# ── fetch_results ──


class TestFetchResults:
    def _agent(self) -> ClaudeCodeAgentAdapter:
        return ClaudeCodeAgentAdapter()

    def test_extracts_reply_text(self):
        raw = _ndjson(
            {"type": "assistant", "content": [{"type": "text", "text": "Hello world"}]},
            {"type": "result", "session_id": "s1", "is_error": False},
        )
        to = self._agent().fetch_results(raw)
        assert to.reply_text == "Hello world"
        assert to.has_reply is True
        assert to.has_error is False

    def test_extracts_reply_from_nested_message(self):
        raw = _ndjson(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Nested reply"}],
                    "role": "assistant",
                },
            },
            {"type": "result", "session_id": "s1", "is_error": False},
        )
        to = self._agent().fetch_results(raw)
        assert to.reply_text == "Nested reply"
        assert to.has_reply is True

    def test_extracts_tool_calls_from_nested_message(self):
        raw = _ndjson(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "name": "Read", "id": "tc_1", "input": {"path": "/foo"}},
                        {"type": "text", "text": "Reading..."},
                    ],
                },
            },
            {"type": "result", "session_id": "s1", "is_error": False},
        )
        to = self._agent().fetch_results(raw)
        assert len(to.tool_calls) == 1
        assert to.tool_calls[0].name == "Read"
        assert to.reply_text == "Reading..."

    def test_extracts_tool_calls_from_assistant(self):
        raw = _ndjson(
            {
                "type": "assistant",
                "content": [
                    {"type": "tool_use", "name": "Read", "id": "tc_1", "input": {"path": "/foo"}},
                    {"type": "text", "text": "Reading file..."},
                ],
            },
            {"type": "result", "session_id": "s1", "is_error": False},
        )
        to = self._agent().fetch_results(raw)
        assert len(to.tool_calls) == 1
        assert to.tool_calls[0].name == "Read"
        assert to.tool_calls[0].args == {"path": "/foo"}

    def test_extracts_tool_calls_from_tool_use_event(self):
        raw = _ndjson(
            {"type": "tool_use", "name": "Write", "id": "tc_2", "input": {"path": "/bar"}},
            {"type": "result", "session_id": "s1", "is_error": False},
        )
        to = self._agent().fetch_results(raw)
        assert len(to.tool_calls) == 1
        assert to.tool_calls[0].name == "Write"

    def test_attaches_tool_result_with_list_content(self):
        """MCP / Bash / Read all return tool results as a list of typed
        content items. The adapter must attach the text payload onto the
        matching ToolCall.result so `tool_result_contains` /
        `tool_result_pattern` can match against it."""
        raw = _ndjson(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "mcp__orders_db__get_order",
                            "id": "toolu_abc",
                            "input": {"order_id": 42},
                        }
                    ],
                },
            },
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_abc",
                            "content": [
                                {
                                    "type": "text",
                                    "text": '{"order_id":42,"customer":"Ada Lovelace","tracking_number":"AGB-EVAL-42"}',
                                }
                            ],
                        }
                    ],
                },
            },
            {"type": "result", "session_id": "s1", "is_error": False},
        )
        to = self._agent().fetch_results(raw)
        assert len(to.tool_calls) == 1
        tc = to.tool_calls[0]
        assert tc.name == "mcp__orders_db__get_order"
        assert tc.result is not None
        assert "AGB-EVAL-42" in tc.result["text"]
        assert "Ada Lovelace" in tc.result["text"]
        assert tc.result["content"] == [
            {
                "type": "text",
                "text": '{"order_id":42,"customer":"Ada Lovelace","tracking_number":"AGB-EVAL-42"}',
            }
        ]

    def test_attaches_tool_result_with_string_content(self):
        """Some Claude Code tools (e.g. Skill launch confirmations) carry
        the tool_result content as a bare string. Adapter must handle both."""
        raw = _ndjson(
            {
                "type": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Skill",
                        "id": "toolu_skill",
                        "input": {"skill": "orders-helper"},
                    }
                ],
            },
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_skill",
                            "content": "Launching skill: orders-helper",
                        }
                    ],
                },
            },
            {"type": "result", "session_id": "s1", "is_error": False},
        )
        to = self._agent().fetch_results(raw)
        assert len(to.tool_calls) == 1
        tc = to.tool_calls[0]
        assert tc.result is not None
        assert tc.result["text"] == "Launching skill: orders-helper"
        assert tc.result["content"] == "Launching skill: orders-helper"

    def test_tool_result_captures_is_error_flag(self):
        """Tool failures arrive with is_error=true on the tool_result block."""
        raw = _ndjson(
            {
                "type": "assistant",
                "content": [{"type": "tool_use", "name": "Bash", "id": "toolu_bash", "input": {"command": "false"}}],
            },
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_bash",
                            "is_error": True,
                            "content": [{"type": "text", "text": "exit code 1"}],
                        }
                    ],
                },
            },
            {"type": "result", "session_id": "s1", "is_error": False},
        )
        to = self._agent().fetch_results(raw)
        assert to.tool_calls[0].result == {
            "content": [{"type": "text", "text": "exit code 1"}],
            "text": "exit code 1",
            "is_error": True,
        }

    def test_tool_result_with_no_matching_tool_use_is_dropped(self):
        """A tool_result whose tool_use_id was never seen as a tool_use is
        silently ignored, and robust to malformed transcripts without raising."""
        raw = _ndjson(
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_orphan",
                            "content": "should be ignored",
                        }
                    ],
                },
            },
            {"type": "result", "session_id": "s1", "is_error": False},
        )
        to = self._agent().fetch_results(raw)
        assert to.tool_calls == []

    def test_stores_session_id(self):
        agent = self._agent()
        raw = _ndjson({"type": "result", "session_id": "session-abc", "is_error": False})
        agent.fetch_results(raw)
        assert agent._session_id == "session-abc"

    def test_detects_error(self):
        raw = _ndjson({"type": "result", "session_id": "s1", "is_error": True})
        to = self._agent().fetch_results(raw)
        assert to.has_error is True

    def test_extracts_timing(self):
        raw = _ndjson({"type": "result", "session_id": "s1", "duration_ms": 5000, "is_error": False})
        to = self._agent().fetch_results(raw)
        assert to.timing is not None
        assert to.timing.total == 5.0

    def test_no_timing_when_missing(self):
        raw = _ndjson({"type": "result", "session_id": "s1", "is_error": False})
        to = self._agent().fetch_results(raw)
        assert to.timing is None

    def test_extracts_cost_usd(self):
        raw = _ndjson({"type": "result", "session_id": "s1", "is_error": False, "total_cost_usd": 0.042})
        to = self._agent().fetch_results(raw)
        assert to.cost_usd == 0.042

    def test_extracts_cost_usd_legacy_key(self):
        raw = _ndjson({"type": "result", "session_id": "s1", "is_error": False, "cost_usd": 0.035})
        to = self._agent().fetch_results(raw)
        assert to.cost_usd == 0.035

    def test_no_cost_when_missing(self):
        raw = _ndjson({"type": "result", "session_id": "s1", "is_error": False})
        to = self._agent().fetch_results(raw)
        assert to.cost_usd is None

    def test_empty_output(self):
        to = self._agent().fetch_results("")
        assert to.reply_text == ""
        assert to.has_reply is False
        assert to.tool_calls == []

    def test_error_fallback_from_raw_output(self):
        """When no result event exists, 'error' in raw output triggers has_error."""
        to = self._agent().fetch_results("Error: something went wrong")
        assert to.has_error is True

    def test_no_error_when_clean_output(self):
        to = self._agent().fetch_results("some clean text without issues")
        assert to.has_error is False

    def test_is_error_false_trusted_over_text(self):
        """When result event says is_error=False, text containing 'error' must not override."""
        raw = _ndjson(
            {"type": "assistant", "content": [{"type": "text", "text": "This analyzes security errors"}]},
            {"type": "result", "session_id": "s1", "is_error": False},
        )
        to = self._agent().fetch_results(raw)
        assert to.has_error is False

    def test_multiple_assistant_events_concatenated(self):
        raw = _ndjson(
            {"type": "assistant", "content": [{"type": "text", "text": "Part 1."}]},
            {"type": "assistant", "content": [{"type": "text", "text": "Part 2."}]},
            {"type": "result", "session_id": "s1", "is_error": False},
        )
        to = self._agent().fetch_results(raw)
        assert "Part 1." in to.reply_text
        assert "Part 2." in to.reply_text

    def test_multiple_tool_calls_across_events(self):
        raw = _ndjson(
            {"type": "assistant", "content": [{"type": "tool_use", "name": "Read", "id": "tc1", "input": {}}]},
            {"type": "tool_use", "name": "Write", "id": "tc2", "input": {"path": "/x"}},
            {"type": "result", "session_id": "s1", "is_error": False},
        )
        to = self._agent().fetch_results(raw)
        assert len(to.tool_calls) == 2
        names = {tc.name for tc in to.tool_calls}
        assert names == {"Read", "Write"}

    def test_raw_cli_preserved(self):
        raw = _ndjson({"type": "result", "session_id": "s1", "is_error": False})
        to = self._agent().fetch_results(raw)
        assert to.raw_cli == raw

    def test_nested_message_preferred_over_empty_content(self):
        """When top-level content is empty list, falls through to message.content."""
        raw = _ndjson(
            {
                "type": "assistant",
                "content": [],
                "message": {"content": [{"type": "text", "text": "from message"}]},
            },
            {"type": "result", "session_id": "s1", "is_error": False},
        )
        to = self._agent().fetch_results(raw)
        assert to.reply_text == "from message"


# ── Gap 2: Tool sequence and LLM turn count ──


class TestToolSequenceAndLLMTurns:
    def test_tool_sequence_preserves_order(self):
        raw = _ndjson(
            {"type": "assistant", "content": [{"type": "tool_use", "name": "Read", "id": "t1", "input": {}}]},
            {"type": "assistant", "content": [{"type": "tool_use", "name": "Edit", "id": "t2", "input": {}}]},
            {"type": "tool_use", "name": "Write", "id": "t3", "input": {}},
            {"type": "result", "session_id": "s1", "is_error": False},
        )
        to = ClaudeCodeAgentAdapter().fetch_results(raw)
        assert to.tool_sequence == ["Read", "Edit", "Write"]

    def test_llm_turn_count(self):
        raw = _ndjson(
            {"type": "assistant", "content": [{"type": "text", "text": "Turn 1"}]},
            {"type": "assistant", "content": [{"type": "text", "text": "Turn 2"}]},
            {"type": "assistant", "content": [{"type": "text", "text": "Turn 3"}]},
            {"type": "result", "session_id": "s1", "is_error": False},
        )
        to = ClaudeCodeAgentAdapter().fetch_results(raw)
        assert to.llm_turn_count == 3

    def test_llm_turn_count_none_when_no_assistant(self):
        raw = _ndjson({"type": "result", "session_id": "s1", "is_error": False})
        to = ClaudeCodeAgentAdapter().fetch_results(raw)
        assert to.llm_turn_count is None


# ── Gap 3: error_type ──


class TestErrorType:
    def test_extracts_error_type_string(self):
        # Vendor labels that don't match a canonical pattern fall back
        # to ``UNKNOWN`` so consumers can rely on the closed token set
        # in :data:`belt.entities.ERROR_TYPES`.
        from belt.entities import UNKNOWN

        raw = _ndjson({"type": "result", "session_id": "s1", "is_error": True, "error": "overloaded"})
        to = ClaudeCodeAgentAdapter().fetch_results(raw)
        assert to.error_type == UNKNOWN

    def test_extracts_error_type_from_dict(self):
        # Vendor tokens that match a canonical pattern via the message
        # text are normalised so downstream consumers see the
        # framework token (``RATE_LIMITED``) rather than the raw vendor
        # string (``"rate_limit"``).
        from belt.entities import RATE_LIMITED

        raw = _ndjson(
            {
                "type": "result",
                "session_id": "s1",
                "is_error": True,
                "error": {"type": "rate_limit", "message": "rate_limit_error: too many"},
            }
        )
        to = ClaudeCodeAgentAdapter().fetch_results(raw)
        assert to.error_type == RATE_LIMITED

    def test_no_error_type_when_no_error(self):
        raw = _ndjson({"type": "result", "session_id": "s1", "is_error": False})
        to = ClaudeCodeAgentAdapter().fetch_results(raw)
        assert to.error_type is None


# ── Gap 5: Thinking blocks ──


class TestThinkingBlocks:
    def test_extracts_thinking_text(self):
        raw = _ndjson(
            {
                "type": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me analyze this carefully..."},
                    {"type": "text", "text": "Here is my answer."},
                ],
            },
            {"type": "result", "session_id": "s1", "is_error": False},
        )
        to = ClaudeCodeAgentAdapter().fetch_results(raw)
        assert to.thinking_text == "Let me analyze this carefully..."
        assert to.reply_text == "Here is my answer."

    def test_thinking_text_none_when_absent(self):
        raw = _ndjson(
            {"type": "assistant", "content": [{"type": "text", "text": "No thinking"}]},
            {"type": "result", "session_id": "s1", "is_error": False},
        )
        to = ClaudeCodeAgentAdapter().fetch_results(raw)
        assert to.thinking_text is None

    def test_multiple_thinking_blocks_concatenated(self):
        raw = _ndjson(
            {
                "type": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "First thought."},
                    {"type": "thinking", "thinking": "Second thought."},
                    {"type": "text", "text": "Answer."},
                ],
            },
            {"type": "result", "session_id": "s1", "is_error": False},
        )
        to = ClaudeCodeAgentAdapter().fetch_results(raw)
        assert "First thought." in to.thinking_text
        assert "Second thought." in to.thinking_text

    def test_thinking_block_with_text_key_fallback(self):
        raw = _ndjson(
            {
                "type": "assistant",
                "content": [
                    {"type": "thinking", "text": "Thinking via text key"},
                    {"type": "text", "text": "Answer."},
                ],
            },
            {"type": "result", "session_id": "s1", "is_error": False},
        )
        to = ClaudeCodeAgentAdapter().fetch_results(raw)
        assert to.thinking_text == "Thinking via text key"


# ── Multi-turn session continuity ──


class TestMultiTurn:
    def test_session_persists_across_turns(self):
        agent = ClaudeCodeAgentAdapter()
        agent.setup(_config())
        assert agent._session_id is None

        raw1 = _ndjson(
            {"type": "assistant", "content": [{"type": "text", "text": "Turn 1"}]},
            {"type": "result", "session_id": "sess-1", "is_error": False},
        )
        agent.fetch_results(raw1)
        assert agent._session_id == "sess-1"

        raw2 = _ndjson(
            {"type": "assistant", "content": [{"type": "text", "text": "Turn 2"}]},
            {"type": "result", "session_id": "sess-1", "is_error": False},
        )
        to2 = agent.fetch_results(raw2)
        assert agent._session_id == "sess-1"
        assert to2.reply_text == "Turn 2"

    def test_session_updates_on_new_id(self):
        agent = ClaudeCodeAgentAdapter()
        agent.setup(_config())
        raw1 = _ndjson({"type": "result", "session_id": "old", "is_error": False})
        agent.fetch_results(raw1)
        raw2 = _ndjson({"type": "result", "session_id": "new", "is_error": False})
        agent.fetch_results(raw2)
        assert agent._session_id == "new"


# ── Session management ──


class TestSessionManagement:
    def test_setup_clears_session(self):
        agent = ClaudeCodeAgentAdapter()
        agent._session_id = "old-session"
        agent.setup(_config())
        assert agent._session_id is None

    def test_teardown_clears_session(self):
        agent = ClaudeCodeAgentAdapter()
        agent._session_id = "session-xyz"
        agent.teardown()
        assert agent._session_id is None


# ── Metadata ──


class TestMetadata:
    def test_metadata_with_session(self):
        agent = ClaudeCodeAgentAdapter()
        agent._session_id = "session-123"
        assert agent.metadata() == {"session_id": "session-123"}

    def test_metadata_without_session(self):
        agent = ClaudeCodeAgentAdapter()
        assert agent.metadata() is None


# ── Group lifecycle ──


class TestGroupLifecycle:
    def test_setup_group_noop(self):
        agent = ClaudeCodeAgentAdapter()
        result = agent.setup_group(GroupConfig(agent="claude-code"), Path("/tmp"))
        assert result is None

    def test_group_setup_summary_none(self):
        agent = ClaudeCodeAgentAdapter()
        assert agent.group_setup_summary(None) is None

    def test_teardown_group_noop(self):
        agent = ClaudeCodeAgentAdapter()
        agent.teardown_group(None)


# ── Interface methods (Gap 9) ──


class TestInterfaceMethods:
    def test_cli_options_is_empty(self):
        # Parameterless agent: policy choices like ``model`` flow
        # through scenario flags, not ``-X``. Declaring ``cli_options`` here
        # would let ``_resolve_env_var_defaults`` inject ``model=`` from the
        # environment and crash ``create_agent``.
        assert ClaudeCodeAgentAdapter.cli_options() == []

    def test_display_info_with_version(self):
        with patch("shutil.which", return_value="/usr/local/bin/claude"):
            mock_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="1.2.3\n", stderr="")
            with patch("subprocess.run", return_value=mock_result):
                info = ClaudeCodeAgentAdapter.display_info()
                assert "1.2.3" in info

    def test_display_info_without_claude(self):
        with patch("shutil.which", return_value=None):
            info = ClaudeCodeAgentAdapter.display_info()
            assert "not found" in info

    def test_health_check_noop(self):
        agent = ClaudeCodeAgentAdapter()
        agent.health_check()

    def test_scoring_strategy_returns_default(self):
        agent = ClaudeCodeAgentAdapter()
        strategy = agent.scoring_strategy()
        assert strategy is not None
        assert len(strategy.dimensions) > 0


# ── Constructor ──


class TestConstructor:
    def test_default_session_none(self):
        agent = ClaudeCodeAgentAdapter()
        assert agent._session_id is None
