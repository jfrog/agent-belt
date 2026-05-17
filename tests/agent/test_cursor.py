# (c) JFrog Ltd. (2026)

"""Tests for CursorAgentAdapter - stream-json parsing, subprocess execution,
auth detection, error handling, multi-turn via --resume, full lifecycle."""

from __future__ import annotations

import json
import subprocess
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from belt.agent.base import AgentNotAvailableError
from belt.agent.cursor import CursorAgentAdapter, _extract_cursor_tool
from belt.entities import AgentConfig, GroupConfig
from belt.parser.ndjson import parse_ndjson


def _mock_popen(stdout_text: str, returncode: int = 0, stderr_text: str = ""):
    """Create a mock Popen that yields stdout lines and has expected attributes."""
    mock = MagicMock()
    mock.stdout = StringIO(stdout_text)
    mock.stderr = StringIO(stderr_text)
    mock.returncode = returncode
    mock.wait = MagicMock()
    mock.pid = 99999
    return mock


@pytest.fixture
def agent() -> CursorAgentAdapter:
    a = CursorAgentAdapter()
    a._cli_path = "/usr/local/bin/cursor-agent"
    return a


@pytest.fixture
def config() -> AgentConfig:
    return AgentConfig(
        group_config=GroupConfig(agent="cursor"),
        scenario_name="test",
    )


# ── check_available ──


class TestCheckAvailable:
    """The check_available contract is binary-only - no model calls, no
    auth-status string parsing. Tests guard against regression to the old
    fragile probe."""

    def test_available_when_cursor_agent_on_path(self):
        """Standalone CLI install - `curl https://cursor.com/install | bash` puts
        `cursor-agent` in `~/.local/bin`. No subprocess auth probe runs."""
        with patch(
            "belt.agent.cursor.resolve_binary",
            return_value="/home/user/.local/bin/cursor-agent",
        ):
            with patch("subprocess.run") as mock_run:
                CursorAgentAdapter.check_available()
            mock_run.assert_not_called()

    def test_available_when_cursor_ide_binary_on_path(self):
        """IDE bundle - `cursor` binary with `agent` subcommand. Still OK."""
        with patch(
            "belt.agent.cursor.resolve_binary",
            return_value="/usr/local/bin/cursor",
        ):
            CursorAgentAdapter.check_available()

    def test_not_available_when_missing(self):
        with patch("belt.agent.cursor.resolve_binary", return_value=None):
            with pytest.raises(AgentNotAvailableError, match="cursor CLI not found"):
                CursorAgentAdapter.check_available()

    def test_install_hint_points_to_official_installer(self):
        """The error message should guide the user to the official installer
        (not the IDE) - that's the path that works in CI."""
        with patch("belt.agent.cursor.resolve_binary", return_value=None):
            try:
                CursorAgentAdapter.check_available()
            except AgentNotAvailableError as e:
                assert "cursor.com/install" in e.suggestion
                assert "$HOME/.local/bin" in e.suggestion

    def test_no_subprocess_call_during_check(self):
        """`agent status` was string-matched and produced false negatives when
        authenticated via CURSOR_API_KEY. The new contract forbids ALL subprocess
        calls during check_available."""
        with patch(
            "belt.agent.cursor.resolve_binary",
            return_value="/home/user/.local/bin/cursor-agent",
        ):
            with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
                CursorAgentAdapter.check_available()
            mock_run.assert_not_called()
            mock_popen.assert_not_called()


# ── _resolve_cli + _build_cmd ──


class TestBinaryResolution:
    def test_resolve_prefers_specific_names_over_generic_agent(self):
        """`cursor-agent` and `cursor` are unambiguous; `agent` is a generic
        name that can collide with unrelated tools, so it is checked last.

        The modern installer creates `~/.local/bin/agent` as the primary
        symlink and `~/.local/bin/cursor-agent` as a legacy alias to the
        same binary, so on a fresh install all three candidates resolve to
        the same place; the ordering matters only when collisions exist."""
        from belt.agent.cursor import _BINARY_CANDIDATES

        assert _BINARY_CANDIDATES == ("cursor-agent", "cursor", "agent")

    def test_resolve_searches_local_bin(self):
        """The official installer drops the binary in ~/.local/bin even when
        that's not yet on $PATH (user hasn't sourced rc file yet)."""
        from belt.agent.cursor import _EXTRA_PATHS

        assert any(".local/bin" in str(p) for p in _EXTRA_PATHS)

    def test_build_cmd_for_cursor_agent_drops_subcommand(self):
        cmd = CursorAgentAdapter._build_cmd("/home/user/.local/bin/cursor-agent", "-p", "hello")
        assert cmd == ["/home/user/.local/bin/cursor-agent", "-p", "hello"]

    def test_build_cmd_for_modern_agent_symlink_drops_subcommand(self):
        """`~/.local/bin/agent` is the modern primary symlink to the same
        standalone binary as `cursor-agent` -- invoked positionally, no
        `agent` subcommand to insert."""
        cmd = CursorAgentAdapter._build_cmd("/home/user/.local/bin/agent", "-p", "hello")
        assert cmd == ["/home/user/.local/bin/agent", "-p", "hello"]

    def test_build_cmd_for_cursor_ide_inserts_agent_subcommand(self):
        cmd = CursorAgentAdapter._build_cmd("/usr/local/bin/cursor", "-p", "hello")
        assert cmd == ["/usr/local/bin/cursor", "agent", "-p", "hello"]


class TestAuthSignals:
    def test_credential_env_declared(self):
        assert "CURSOR_API_KEY" in CursorAgentAdapter.CREDENTIAL_ENV

    def test_signals_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CURSOR_API_KEY", "xxx")
        monkeypatch.setattr(CursorAgentAdapter, "CREDENTIAL_PATHS", (tmp_path / "nonexistent",))
        signals = CursorAgentAdapter.auth_signals()
        assert "env CURSOR_API_KEY" in signals
        assert not any("stored login" in s for s in signals)

    def test_signals_stored_login(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CURSOR_API_KEY", raising=False)
        cred_dir = tmp_path / ".cursor"
        cred_dir.mkdir()
        monkeypatch.setattr(CursorAgentAdapter, "CREDENTIAL_PATHS", (cred_dir,))
        signals = CursorAgentAdapter.auth_signals()
        assert any("stored login" in s for s in signals)
        assert not any("env" in s for s in signals)

    def test_signals_both(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CURSOR_API_KEY", "xxx")
        cred_dir = tmp_path / ".cursor"
        cred_dir.mkdir()
        monkeypatch.setattr(CursorAgentAdapter, "CREDENTIAL_PATHS", (cred_dir,))
        signals = CursorAgentAdapter.auth_signals()
        assert len(signals) == 2
        assert signals[0].startswith("env ")  # env var listed first
        assert "stored login" in signals[1]

    def test_signals_neither(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CURSOR_API_KEY", raising=False)
        monkeypatch.setattr(CursorAgentAdapter, "CREDENTIAL_PATHS", (tmp_path / "absent",))
        assert CursorAgentAdapter.auth_signals() == []


# ── execute ──


class TestExecute:
    def test_successful_execution_with_cursor_agent(self, agent: CursorAgentAdapter):
        """Standalone CLI form - `cursor-agent -p ...` (no `agent` subcommand)."""
        agent._cli_path = "/home/user/.local/bin/cursor-agent"
        events = [
            json.dumps({"type": "text", "text": "The answer is 2."}),
            json.dumps({"type": "result", "chat_id": "chat-123"}),
        ]
        mock = _mock_popen("\n".join(events) + "\n")
        with patch("subprocess.Popen", return_value=mock) as mock_popen:
            _raw = agent.execute("What is 1+1?", [])
            args = mock_popen.call_args[0][0]
            assert args[0] == "/home/user/.local/bin/cursor-agent"
            assert args[1] == "-p"
            assert "agent" not in args[:2]
            assert "--output-format" in args
            assert "stream-json" in args
            assert args[-1] == "What is 1+1?"

    def test_successful_execution_with_cursor_ide(self, agent: CursorAgentAdapter):
        """IDE bundle form - `cursor agent -p ...` (subcommand inserted)."""
        agent._cli_path = "/usr/local/bin/cursor"
        events = [json.dumps({"type": "text", "text": "ok"})]
        mock = _mock_popen("\n".join(events) + "\n")
        with patch("subprocess.Popen", return_value=mock) as mock_popen:
            agent.execute("hi", [])
            args = mock_popen.call_args[0][0]
            assert args[:3] == ["/usr/local/bin/cursor", "agent", "-p"]

    def test_resume_on_second_turn(self, agent: CursorAgentAdapter):
        agent._chat_id = "chat-abc"
        mock = _mock_popen("{}\n")
        with patch("subprocess.Popen", return_value=mock) as mock_popen:
            agent.execute("follow up", [])
            args = mock_popen.call_args[0][0]
            assert "--resume" in args
            idx = args.index("--resume")
            assert args[idx + 1] == "chat-abc"

    def test_nonzero_exit_appends_stderr(self, agent: CursorAgentAdapter):
        mock = _mock_popen("", returncode=1, stderr_text="rate limit exceeded")
        with patch("subprocess.Popen", return_value=mock):
            raw = agent.execute("fail", [])
            assert "rate limit exceeded" in raw

    def test_timeout_raises(self, agent: CursorAgentAdapter):
        mock = _mock_popen("")
        mock.wait.side_effect = subprocess.TimeoutExpired("cursor", 300)
        mock.kill = MagicMock()
        with patch("subprocess.Popen", return_value=mock):
            with pytest.raises(subprocess.TimeoutExpired):
                agent.execute("slow prompt", [])

    def test_passes_extra_flags(self, agent: CursorAgentAdapter):
        mock = _mock_popen("ok\n")
        with patch("subprocess.Popen", return_value=mock) as mock_popen:
            agent.execute("test", ["--model", "gpt-5"])
            args = mock_popen.call_args[0][0]
            assert "--model" in args
            assert "gpt-5" in args


# ── fetch_results (NDJSON) ──


class TestFetchResults:
    def test_parses_text_event(self, agent: CursorAgentAdapter):
        events = json.dumps({"type": "text", "text": "Hello!"})
        to = agent.fetch_results(events)
        assert to.reply_text == "Hello!"
        assert to.has_reply is True

    def test_parses_message_event(self, agent: CursorAgentAdapter):
        events = json.dumps({"type": "message", "content": "Hi there"})
        to = agent.fetch_results(events)
        assert "Hi there" in to.reply_text
        assert to.has_reply is True

    def test_parses_tool_use_event(self, agent: CursorAgentAdapter):
        events = "\n".join(
            [
                json.dumps({"type": "tool_use", "name": "read_file", "id": "t1", "input": {"path": "/tmp"}}),
                json.dumps({"type": "text", "text": "Done."}),
            ]
        )
        to = agent.fetch_results(events)
        assert len(to.tool_calls) == 1
        assert to.tool_calls[0].name == "read_file"
        assert to.has_reply is True

    def test_parses_result_event_with_chat_id(self, agent: CursorAgentAdapter):
        events = "\n".join(
            [
                json.dumps({"type": "text", "text": "ok"}),
                json.dumps({"type": "result", "chat_id": "chat-789", "duration_ms": 1500}),
            ]
        )
        to = agent.fetch_results(events)
        assert agent._chat_id == "chat-789"
        assert to.timing is not None
        assert to.timing.total == 1.5

    def test_empty_output(self, agent: CursorAgentAdapter):
        to = agent.fetch_results("")
        assert to.reply_text == ""
        assert to.has_reply is False

    def test_concatenates_thinking_delta_events(self, agent: CursorAgentAdapter):
        """Cursor reasoning arrives as ``{"type":"thinking","subtype":"delta","text":...}``
        deltas; ``thinking_text`` is the concatenation in arrival order."""
        events = "\n".join(
            [
                json.dumps({"type": "thinking", "subtype": "delta", "text": "To compute the "}),
                json.dumps({"type": "thinking", "subtype": "delta", "text": "10th Fibonacci, "}),
                json.dumps({"type": "thinking", "subtype": "delta", "text": "I trace F(n)..."}),
                json.dumps({"type": "text", "text": "55"}),
            ]
        )
        to = agent.fetch_results(events)
        assert to.thinking_text == "To compute the 10th Fibonacci, I trace F(n)..."
        assert to.reply_text == "55"
        assert to.has_reply is True

    def test_thinking_text_is_none_when_no_thinking_events(self, agent: CursorAgentAdapter):
        """No thinking deltas means ``thinking_text`` stays ``None`` (not ``""``)
        so ``has_thinking`` rule checks distinguish "not emitted" from "empty trace"."""
        events = json.dumps({"type": "text", "text": "OK"})
        to = agent.fetch_results(events)
        assert to.thinking_text is None

    def test_thinking_events_ignored_without_text_field(self, agent: CursorAgentAdapter):
        """A ``thinking`` event with no usable text contributes nothing; the
        adapter must not concatenate ``None`` or non-string payloads."""
        events = "\n".join(
            [
                json.dumps({"type": "thinking", "subtype": "started"}),
                json.dumps({"type": "thinking", "subtype": "delta", "text": ""}),
                json.dumps({"type": "thinking", "subtype": "delta", "text": None}),
                json.dumps({"type": "text", "text": "answer"}),
            ]
        )
        to = agent.fetch_results(events)
        assert to.thinking_text is None
        assert to.reply_text == "answer"

    def test_supported_output_fields_declares_thinking_text(self):
        """The adapter contract: declaring ``thinking_text`` unlocks the
        ``has_thinking`` rule check in scoring strategies that gate on
        adapter capability."""
        assert "thinking_text" in CursorAgentAdapter.supported_output_fields()

    def test_result_is_error(self, agent: CursorAgentAdapter):
        events = "\n".join(
            [
                json.dumps({"type": "text", "text": "oops"}),
                json.dumps({"type": "result", "is_error": True}),
            ]
        )
        to = agent.fetch_results(events)
        assert to.has_error is True

    def test_multiline_reply(self, agent: CursorAgentAdapter):
        events = "\n".join(
            [
                json.dumps({"type": "text", "text": "Line 1"}),
                json.dumps({"type": "text", "text": "Line 2"}),
            ]
        )
        to = agent.fetch_results(events)
        assert "Line 1" in to.reply_text
        assert "Line 2" in to.reply_text

    def test_timing_from_execute_fallback(self, agent: CursorAgentAdapter):
        agent._last_duration = 3.5
        to = agent.fetch_results(json.dumps({"type": "text", "text": "ok"}))
        assert to.timing is not None
        assert to.timing.total == 3.5

    def test_parses_cursor_mcp_tool_call(self, agent: CursorAgentAdapter):
        """Cursor wraps MCP tools in mcpToolCall with toolName nested inside."""
        events = "\n".join(
            [
                json.dumps(
                    {
                        "type": "tool_call",
                        "subtype": "started",
                        "call_id": "tc-1",
                        "tool_call": {
                            "mcpToolCall": {
                                "args": {
                                    "toolName": "get_demo_packages_information",
                                    "args": {"queryType": "search_packages"},
                                    "providerIdentifier": "demo-mcp",
                                }
                            }
                        },
                    }
                ),
                json.dumps({"type": "tool_call", "subtype": "completed", "call_id": "tc-1", "tool_call": {}}),
                json.dumps({"type": "text", "text": "Found packages."}),
            ]
        )
        to = agent.fetch_results(events)
        assert len(to.tool_calls) == 1
        assert to.tool_calls[0].name == "get_demo_packages_information"
        assert to.tool_calls[0].call_id == "tc-1"
        assert to.tool_calls[0].args == {"queryType": "search_packages"}

    def test_parses_cursor_read_tool_call(self, agent: CursorAgentAdapter):
        events = json.dumps(
            {
                "type": "tool_call",
                "subtype": "started",
                "call_id": "tc-2",
                "tool_call": {"readToolCall": {"args": {"path": "/tmp/file.txt"}}},
            }
        )
        to = agent.fetch_results(events)
        assert len(to.tool_calls) == 1
        assert to.tool_calls[0].name == "Read"
        assert to.tool_calls[0].args == {"path": "/tmp/file.txt"}

    def test_parses_cursor_shell_tool_call(self, agent: CursorAgentAdapter):
        events = json.dumps(
            {
                "type": "tool_call",
                "subtype": "started",
                "call_id": "tc-3",
                "tool_call": {"shellToolCall": {"args": {"command": "ls -la"}}},
            }
        )
        to = agent.fetch_results(events)
        assert len(to.tool_calls) == 1
        assert to.tool_calls[0].name == "Shell"

    def test_parses_cursor_grep_tool_call(self, agent: CursorAgentAdapter):
        events = json.dumps(
            {
                "type": "tool_call",
                "subtype": "started",
                "call_id": "tc-4",
                "tool_call": {"grepToolCall": {"args": {"pattern": "foo"}}},
            }
        )
        to = agent.fetch_results(events)
        assert len(to.tool_calls) == 1
        assert to.tool_calls[0].name == "Grep"

    def test_skips_completed_duplicates(self, agent: CursorAgentAdapter):
        """Only 'started' events should produce tool calls, not both started+completed."""
        events = "\n".join(
            [
                json.dumps(
                    {
                        "type": "tool_call",
                        "subtype": "started",
                        "call_id": "tc-5",
                        "tool_call": {"shellToolCall": {"args": {"command": "echo hi"}}},
                    }
                ),
                json.dumps(
                    {
                        "type": "tool_call",
                        "subtype": "completed",
                        "call_id": "tc-5",
                        "tool_call": {"shellToolCall": {"args": {"command": "echo hi"}, "result": {"success": {}}}},
                    }
                ),
            ]
        )
        to = agent.fetch_results(events)
        assert len(to.tool_calls) == 1

    def test_completed_attaches_result_to_tool_call(self, agent: CursorAgentAdapter):
        """Completed events attach the result payload to the matching ToolCall."""
        events = "\n".join(
            [
                json.dumps(
                    {
                        "type": "tool_call",
                        "subtype": "started",
                        "call_id": "tc-r1",
                        "tool_call": {"readToolCall": {"args": {"path": "/tmp/f.txt"}}},
                    }
                ),
                json.dumps(
                    {
                        "type": "tool_call",
                        "subtype": "completed",
                        "call_id": "tc-r1",
                        "tool_call": {"readToolCall": {"args": {"path": "/tmp/f.txt"}, "result": {"content": "hello"}}},
                    }
                ),
            ]
        )
        to = agent.fetch_results(events)
        assert len(to.tool_calls) == 1
        assert to.tool_calls[0].result == {"content": "hello"}

    def test_mcp_completed_attaches_result(self, agent: CursorAgentAdapter):
        """MCP completed events attach the result to the matching ToolCall."""
        events = "\n".join(
            [
                json.dumps(
                    {
                        "type": "tool_call",
                        "subtype": "started",
                        "call_id": "tc-m1",
                        "tool_call": {
                            "mcpToolCall": {
                                "args": {
                                    "toolName": "get_demo_information",
                                    "args": {},
                                    "providerIdentifier": "demo-mcp",
                                }
                            }
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "tool_call",
                        "subtype": "completed",
                        "call_id": "tc-m1",
                        "tool_call": {
                            "mcpToolCall": {
                                "args": {
                                    "toolName": "get_demo_information",
                                    "args": {},
                                    "providerIdentifier": "demo-mcp",
                                },
                                "result": {"content": [{"type": "text", "text": "Demo MCP server response"}]},
                            }
                        },
                    }
                ),
            ]
        )
        to = agent.fetch_results(events)
        assert len(to.tool_calls) == 1
        assert to.tool_calls[0].name == "get_demo_information"
        assert to.tool_calls[0].result == {"content": [{"type": "text", "text": "Demo MCP server response"}]}

    def test_result_none_when_no_completed_event(self, agent: CursorAgentAdapter):
        """ToolCall.result stays None when only a started event exists."""
        events = json.dumps(
            {
                "type": "tool_call",
                "subtype": "started",
                "call_id": "tc-no-result",
                "tool_call": {"shellToolCall": {"args": {"command": "ls"}}},
            }
        )
        to = agent.fetch_results(events)
        assert to.tool_calls[0].result is None

    def test_completed_without_matching_started_is_ignored(self, agent: CursorAgentAdapter):
        """A completed event with no matching started call_id is silently ignored."""
        events = "\n".join(
            [
                json.dumps(
                    {
                        "type": "tool_call",
                        "subtype": "started",
                        "call_id": "tc-a",
                        "tool_call": {"shellToolCall": {"args": {"command": "echo"}}},
                    }
                ),
                json.dumps(
                    {
                        "type": "tool_call",
                        "subtype": "completed",
                        "call_id": "tc-orphan",
                        "tool_call": {"shellToolCall": {"args": {"command": "echo"}, "result": {"success": {}}}},
                    }
                ),
            ]
        )
        to = agent.fetch_results(events)
        assert len(to.tool_calls) == 1
        assert to.tool_calls[0].result is None

    def test_mixed_cursor_tool_types(self, agent: CursorAgentAdapter):
        """Real scenario: MCP + Read + Shell + Grep tool calls in one turn."""
        events = "\n".join(
            [
                json.dumps(
                    {
                        "type": "tool_call",
                        "subtype": "started",
                        "call_id": "tc-a",
                        "tool_call": {
                            "mcpToolCall": {
                                "args": {
                                    "toolName": "get_releases_info_from_demo",
                                    "args": {},
                                    "providerIdentifier": "demo-mcp",
                                }
                            }
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "tool_call",
                        "subtype": "started",
                        "call_id": "tc-b",
                        "tool_call": {"readToolCall": {"args": {"path": "/tmp/out.txt"}}},
                    }
                ),
                json.dumps(
                    {
                        "type": "tool_call",
                        "subtype": "started",
                        "call_id": "tc-c",
                        "tool_call": {"shellToolCall": {"args": {"command": "head -5 /tmp/out.txt"}}},
                    }
                ),
                json.dumps(
                    {
                        "type": "tool_call",
                        "subtype": "started",
                        "call_id": "tc-d",
                        "tool_call": {"grepToolCall": {"args": {"pattern": "version"}}},
                    }
                ),
                json.dumps({"type": "text", "text": "Done."}),
                json.dumps({"type": "result", "chat_id": "chat-mixed", "duration_ms": 5000}),
            ]
        )
        to = agent.fetch_results(events)
        names = [tc.name for tc in to.tool_calls]
        assert names == ["get_releases_info_from_demo", "Read", "Shell", "Grep"]
        assert agent._chat_id == "chat-mixed"


# ── _extract_cursor_tool ──


class TestExtractCursorTool:
    def test_mcp_tool_call(self):
        name, args = _extract_cursor_tool(
            {"mcpToolCall": {"args": {"toolName": "my_tool", "args": {"key": "val"}, "providerIdentifier": "server"}}}
        )
        assert name == "my_tool"
        assert args == {"key": "val"}

    def test_read_tool_call(self):
        name, args = _extract_cursor_tool({"readToolCall": {"args": {"path": "/x"}}})
        assert name == "Read"
        assert args == {"path": "/x"}

    def test_unknown_structure(self):
        name, args = _extract_cursor_tool({"unknownThing": {}})
        assert name == ""
        assert args == {}


# ── Lifecycle ──


class TestLifecycle:
    def test_setup_resets_chat_id(self, agent: CursorAgentAdapter, config: AgentConfig):
        agent._chat_id = "old"
        agent.setup(config)
        assert agent._chat_id is None

    def test_setup_resolves_and_caches_cli_path(self, config: AgentConfig):
        """setup() should resolve the binary once so subsequent execute() calls
        don't pay the resolution cost on every turn."""
        a = CursorAgentAdapter()
        with patch(
            "belt.agent.cursor.resolve_binary",
            return_value="/home/u/.local/bin/cursor-agent",
        ):
            a.setup(config)
        assert a._cli_path == "/home/u/.local/bin/cursor-agent"

    def test_execute_falls_back_to_resolution_if_setup_was_skipped(self):
        """Some test/runtime paths instantiate without calling setup();
        execute() should still work via lazy resolution."""
        a = CursorAgentAdapter()
        a._cli_path = None
        mock = _mock_popen("{}\n")
        with patch(
            "belt.agent.cursor.resolve_binary",
            return_value="/usr/local/bin/cursor-agent",
        ):
            with patch("subprocess.Popen", return_value=mock) as mock_popen:
                a.execute("hi", [])
            assert mock_popen.call_args[0][0][0] == "/usr/local/bin/cursor-agent"

    def test_execute_falls_back_to_canonical_name_if_unresolved(self):
        """If the binary can't be resolved (e.g. test that mocks subprocess.Popen
        without installing cursor), execute() falls back to the canonical name
        ``cursor-agent`` rather than raising. check_available() is the gate that
        determines whether the binary actually exists."""
        a = CursorAgentAdapter()
        a._cli_path = None
        mock = _mock_popen("{}\n")
        with patch("belt.agent.cursor.resolve_binary", return_value=None):
            with patch("subprocess.Popen", return_value=mock) as mock_popen:
                a.execute("hi", [])
            assert mock_popen.call_args[0][0][0] == "cursor-agent"

    def test_cli_path_persists_across_turns(self, config: AgentConfig):
        """Multi-turn scenarios reuse the same instance - cached _cli_path
        must persist across execute() calls without re-resolving."""
        a = CursorAgentAdapter()
        with patch(
            "belt.agent.cursor.resolve_binary",
            return_value="/home/u/.local/bin/cursor-agent",
        ) as mock_resolve:
            a.setup(config)
            assert mock_resolve.call_count == 1

            mock = _mock_popen("{}\n")
            with patch("subprocess.Popen", return_value=mock):
                a.execute("turn 1", [])
                a.execute("turn 2", [])
            assert mock_resolve.call_count == 1, "should not re-resolve on subsequent turns"

    def test_teardown_resets_chat_id(self, agent: CursorAgentAdapter):
        agent._chat_id = "active"
        agent.teardown()
        assert agent._chat_id is None

    def test_metadata_with_chat_id(self, agent: CursorAgentAdapter):
        agent._chat_id = "chat-999"
        meta = agent.metadata()
        assert meta == {"chat_id": "chat-999"}

    def test_metadata_without_chat_id(self, agent: CursorAgentAdapter):
        assert agent.metadata() is None

    def test_group_setup_returns_none(self, agent: CursorAgentAdapter):
        gc = GroupConfig(agent="cursor")
        assert agent.setup_group(gc, Path("/tmp")) is None


# ── Parsers ──


class TestParsers:
    def testparse_ndjson_mixed(self):
        raw = 'not json\n{"type": "text"}\n\n{"type": "result"}\n'
        events = parse_ndjson(raw)
        assert len(events) == 2


# ── Interface methods ──


class TestInterfaceMethods:
    def test_cli_options_empty(self):
        assert CursorAgentAdapter.cli_options() == []

    def test_display_info_from_about(self):
        result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="Cursor CLI 2026.03\nclaude-4.6-opus\nuser@test.com\n", stderr=""
        )
        with patch(
            "belt.agent.cursor.resolve_binary",
            return_value="/home/user/.local/bin/cursor-agent",
        ):
            with patch("subprocess.run", return_value=result):
                info = CursorAgentAdapter.display_info()
        assert "Cursor CLI 2026.03" in info
        assert "claude-4.6-opus" in info

    def test_display_info_uses_about_subcommand_for_ide(self):
        """For the IDE binary, `about` must be invoked as `cursor agent about`."""
        result = subprocess.CompletedProcess(args=[], returncode=0, stdout="v1\n", stderr="")
        with patch(
            "belt.agent.cursor.resolve_binary",
            return_value="/usr/local/bin/cursor",
        ):
            with patch("subprocess.run", return_value=result) as mock_run:
                CursorAgentAdapter.display_info()
            cmd = mock_run.call_args[0][0]
            assert cmd[:3] == ["/usr/local/bin/cursor", "agent", "about"]

    def test_display_info_fallback_when_binary_missing(self):
        with patch("belt.agent.cursor.resolve_binary", return_value=None):
            assert "not found" in CursorAgentAdapter.display_info()

    def test_display_info_fallback_on_subprocess_failure(self):
        with patch(
            "belt.agent.cursor.resolve_binary",
            return_value="/home/user/.local/bin/cursor-agent",
        ):
            with patch("subprocess.run", side_effect=FileNotFoundError):
                info = CursorAgentAdapter.display_info()
        assert "CursorAgentAdapter" in info

    def test_health_check_noop(self):
        CursorAgentAdapter().health_check()

    def test_scoring_strategy(self):
        strategy = CursorAgentAdapter().scoring_strategy()
        assert strategy is not None


class TestHeadlessSafety:
    """Cursor hangs without --force/--approve-mcps in headless mode.

    Unlike Claude Code (auto-approves in -p mode), Cursor prompts for
    tool approval even when headless.  The agent always injects these.
    """

    def test_force_and_approve_mcps_always_present(self):
        agent = CursorAgentAdapter()
        mock = _mock_popen("{}\n")
        with patch("subprocess.Popen", return_value=mock) as mock_popen:
            agent.execute("test", [])
            args = mock_popen.call_args[0][0]
            assert "--force" in args
            assert "--approve-mcps" in args

    def test_flags_from_scenario_appended_after(self):
        agent = CursorAgentAdapter()
        mock = _mock_popen("{}\n")
        with patch("subprocess.Popen", return_value=mock) as mock_popen:
            agent.execute("test", ["--model", "gpt-5", "--workspace", "/my/repo"])
            args = mock_popen.call_args[0][0]
            assert "--model" in args
            assert args[args.index("--model") + 1] == "gpt-5"
            assert "--workspace" in args
            assert args[args.index("--workspace") + 1] == "/my/repo"
