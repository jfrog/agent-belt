# (c) JFrog Ltd. (2026)

"""Tests for OpenCodeAgentAdapter - NDJSON parsing, Popen streaming execution,
multi-turn session continuity, cost aggregation, error handling,
timing metrics, tool call extraction, full lifecycle."""

from __future__ import annotations

import json
import subprocess
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from belt.agent.base import AgentNotAvailableError
from belt.agent.opencode import OpenCodeAgentAdapter
from belt.entities import AgentConfig, GroupConfig


@pytest.fixture
def agent() -> OpenCodeAgentAdapter:
    return OpenCodeAgentAdapter()


@pytest.fixture
def config() -> AgentConfig:
    return AgentConfig(
        group_config=GroupConfig(agent="opencode"),
        scenario_name="test",
    )


def _mock_popen(stdout_text: str, returncode: int = 0, stderr_text: str = "") -> MagicMock:
    """Create a mock Popen object with given stdout/stderr."""
    proc = MagicMock()
    proc.stdout = StringIO(stdout_text)
    proc.stderr = StringIO(stderr_text)
    proc.returncode = returncode
    proc.wait = MagicMock(return_value=returncode)
    proc.pid = 99999
    return proc


# ── check_available ──


class TestCheckAvailable:
    def test_not_available_when_missing(self):
        with patch("shutil.which", return_value=None):
            with pytest.raises(AgentNotAvailableError, match="opencode CLI not found"):
                OpenCodeAgentAdapter.check_available()

    def test_available_when_present(self):
        with patch("shutil.which", return_value="/usr/local/bin/opencode"):
            OpenCodeAgentAdapter.check_available()


class TestAuthSignals:
    def test_no_credential_env_declared(self):
        # OpenCode is multi-provider - declaring a single env var would create
        # noisy false positives (e.g. user has OPENAI_API_KEY for unrelated tools
        # but configured opencode for Anthropic). Path-only is the safe signal.
        assert OpenCodeAgentAdapter.CREDENTIAL_ENV == ()

    def test_declares_opencode_auth_path(self):
        paths = [str(p) for p in OpenCodeAgentAdapter.CREDENTIAL_PATHS]
        assert any("opencode/auth.json" in p for p in paths)

    def test_path_signal_detected(self, monkeypatch, tmp_path):
        from belt.agent.base import _detect_auth_signals

        auth = tmp_path / ".local" / "share" / "opencode" / "auth.json"
        auth.parent.mkdir(parents=True)
        auth.write_text("{}")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        sigs = _detect_auth_signals((), (Path.home() / ".local" / "share" / "opencode" / "auth.json",))
        assert any("stored login" in s for s in sigs)


# ── execute ──


class TestExecute:
    def test_successful_execution(self, agent: OpenCodeAgentAdapter):
        event = json.dumps({"type": "text", "part": {"text": "hello"}})
        proc = _mock_popen(event + "\n")
        with patch("subprocess.Popen", return_value=proc) as mock_popen:
            raw = agent.execute("What is 1+1?", [])
            args = mock_popen.call_args[0][0]
            assert args[:4] == ["opencode", "run", "--format", "json"]
            assert args[-1] == "What is 1+1?"
            assert event in raw

    def test_passes_session_on_second_turn(self, agent: OpenCodeAgentAdapter):
        agent._session_id = "ses_abc123"
        proc = _mock_popen("{}\n")
        with patch("subprocess.Popen", return_value=proc) as mock_popen:
            agent.execute("follow up", [])
            args = mock_popen.call_args[0][0]
            assert "--session" in args
            idx = args.index("--session")
            assert args[idx + 1] == "ses_abc123"
            assert "--continue" in args

    def test_passes_model_flag(self, agent: OpenCodeAgentAdapter):
        agent._model = "anthropic/claude-sonnet-4-20250514"
        proc = _mock_popen("ok\n")
        with patch("subprocess.Popen", return_value=proc) as mock_popen:
            agent.execute("test", [])
            args = mock_popen.call_args[0][0]
            assert "--model" in args
            idx = args.index("--model")
            assert args[idx + 1] == "anthropic/claude-sonnet-4-20250514"

    def test_nonzero_exit_appends_stderr(self, agent: OpenCodeAgentAdapter):
        proc = _mock_popen("", returncode=1, stderr_text="provider_error")
        with patch("subprocess.Popen", return_value=proc):
            raw = agent.execute("fail", [])
            assert "provider_error" in raw

    def test_passes_extra_flags(self, agent: OpenCodeAgentAdapter):
        proc = _mock_popen("ok\n")
        with patch("subprocess.Popen", return_value=proc) as mock_popen:
            agent.execute("test", ["--agent", "plan"])
            args = mock_popen.call_args[0][0]
            assert "--agent" in args
            assert "plan" in args

    def test_timing_metrics_captured(self, agent: OpenCodeAgentAdapter):
        lines = (
            json.dumps({"type": "text", "part": {"text": "hello"}})
            + "\n"
            + json.dumps({"type": "step_finish", "part": {"cost": 0.001}})
            + "\n"
        )
        proc = _mock_popen(lines)
        with patch("subprocess.Popen", return_value=proc):
            agent.execute("test", [])
            assert agent._ttfe is not None
            assert agent._ttlt is not None


# ── fetch_results ──


class TestFetchResults:
    def test_parses_text_event(self, agent: OpenCodeAgentAdapter):
        event = json.dumps({"type": "text", "part": {"text": "The answer is 42."}})
        to = agent.fetch_results(event)
        assert to.reply_text == "The answer is 42."
        assert to.has_reply is True

    def test_parses_multiple_text_events(self, agent: OpenCodeAgentAdapter):
        events = "\n".join(
            [
                json.dumps({"type": "text", "part": {"text": "First part."}}),
                json.dumps({"type": "text", "part": {"text": "Second part."}}),
            ]
        )
        to = agent.fetch_results(events)
        assert "First part." in to.reply_text
        assert "Second part." in to.reply_text

    def test_parses_tool_use_event(self, agent: OpenCodeAgentAdapter):
        event = json.dumps(
            {
                "type": "tool_use",
                "part": {
                    "tool": "bash",
                    "callID": "r9bQWsNLvOrJGIOz",
                    "state": {
                        "status": "completed",
                        "input": {"command": "echo hello", "description": "Print hello"},
                        "output": "hello\n",
                    },
                },
            }
        )
        to = agent.fetch_results(event)
        assert len(to.tool_calls) == 1
        assert to.tool_calls[0].name == "bash"
        assert to.tool_calls[0].call_id == "r9bQWsNLvOrJGIOz"
        assert to.tool_calls[0].args == {"command": "echo hello", "description": "Print hello"}
        assert to.tool_sequence == ["bash"]

    def test_tool_use_captures_result(self, agent: OpenCodeAgentAdapter):
        event = json.dumps(
            {
                "type": "tool_use",
                "part": {
                    "tool": "read",
                    "callID": "abc",
                    "state": {"status": "completed", "input": {"path": "foo.py"}, "output": "content"},
                },
            }
        )
        to = agent.fetch_results(event)
        assert to.tool_calls[0].result == {"output": "content"}

    def test_parses_step_start_session_id(self, agent: OpenCodeAgentAdapter):
        event = json.dumps(
            {
                "type": "step_start",
                "sessionID": "ses_494719016ffe85dkDMj0FPRbHK",
                "part": {"type": "step-start"},
            }
        )
        agent.fetch_results(event)
        assert agent._session_id == "ses_494719016ffe85dkDMj0FPRbHK"

    def test_parses_step_finish_cost(self, agent: OpenCodeAgentAdapter):
        events = "\n".join(
            [
                json.dumps({"type": "step_finish", "part": {"cost": 0.003, "reason": "tool-calls"}}),
                json.dumps({"type": "text", "part": {"text": "done"}}),
                json.dumps({"type": "step_finish", "part": {"cost": 0.001, "reason": "stop"}}),
            ]
        )
        to = agent.fetch_results(events)
        assert to.cost_usd == pytest.approx(0.004)
        assert to.llm_turn_count == 2

    def test_parses_error_event(self, agent: OpenCodeAgentAdapter):
        # Vendor-specific labels (OpenCode's ``"APIError"``) are
        # normalised to the canonical taxonomy. ``"Rate limit
        # exceeded"`` in the data.message text matches the rate-limit
        # pattern, so the framework surfaces ``RATE_LIMITED`` regardless
        # of the vendor name.
        from belt.entities import RATE_LIMITED

        event = json.dumps(
            {
                "type": "error",
                "error": {"name": "APIError", "data": {"message": "Rate limit exceeded"}},
            }
        )
        to = agent.fetch_results(event)
        assert to.has_error is True
        assert to.error_type == RATE_LIMITED

    def test_parses_error_event_string(self, agent: OpenCodeAgentAdapter):
        # Strings that don't match any pattern fall back to ``UNKNOWN``
        # so the cross-phase contract's token set stays closed.
        from belt.entities import UNKNOWN

        event = json.dumps({"type": "error", "error": "something went wrong"})
        to = agent.fetch_results(event)
        assert to.has_error is True
        assert to.error_type == UNKNOWN

    def test_empty_output(self, agent: OpenCodeAgentAdapter):
        to = agent.fetch_results("")
        assert to.reply_text == ""
        assert to.has_reply is False
        assert to.tool_sequence == []
        assert to.cost_usd is None

    def test_raw_cli_preserved(self, agent: OpenCodeAgentAdapter):
        to = agent.fetch_results("some output")
        assert to.raw_cli == "some output"

    def test_multiple_tool_calls_build_sequence(self, agent: OpenCodeAgentAdapter):
        events = "\n".join(
            [
                json.dumps({"type": "tool_use", "part": {"tool": "read", "callID": "t1", "state": {"input": {}}}}),
                json.dumps({"type": "tool_use", "part": {"tool": "write", "callID": "t2", "state": {"input": {}}}}),
                json.dumps({"type": "tool_use", "part": {"tool": "bash", "callID": "t3", "state": {"input": {}}}}),
            ]
        )
        to = agent.fetch_results(events)
        assert to.tool_sequence == ["read", "write", "bash"]
        assert len(to.tool_calls) == 3

    def test_no_cost_when_step_finish_has_none(self, agent: OpenCodeAgentAdapter):
        event = json.dumps({"type": "step_finish", "part": {"reason": "stop"}})
        to = agent.fetch_results(event)
        assert to.cost_usd is None

    def test_zero_cost_is_reported(self, agent: OpenCodeAgentAdapter):
        event = json.dumps({"type": "step_finish", "part": {"cost": 0, "reason": "stop"}})
        to = agent.fetch_results(event)
        assert to.cost_usd == 0.0

    def test_full_conversation_flow(self, agent: OpenCodeAgentAdapter):
        """Simulates a realistic opencode run --format json output."""
        events = "\n".join(
            [
                json.dumps(
                    {
                        "type": "step_start",
                        "timestamp": 1767036059338,
                        "sessionID": "ses_test123",
                        "part": {"type": "step-start", "snapshot": "abc123"},
                    }
                ),
                json.dumps(
                    {
                        "type": "tool_use",
                        "timestamp": 1767036061199,
                        "sessionID": "ses_test123",
                        "part": {
                            "tool": "bash",
                            "callID": "call1",
                            "state": {
                                "status": "completed",
                                "input": {"command": "echo hello"},
                                "output": "hello\n",
                                "metadata": {"exit": 0},
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "step_finish",
                        "timestamp": 1767036061205,
                        "sessionID": "ses_test123",
                        "part": {
                            "type": "step-finish",
                            "reason": "tool-calls",
                            "cost": 0.002,
                            "tokens": {"input": 500, "output": 50},
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "text",
                        "timestamp": 1767036064268,
                        "sessionID": "ses_test123",
                        "part": {"type": "text", "text": "The command output was: hello"},
                    }
                ),
                json.dumps(
                    {
                        "type": "step_finish",
                        "timestamp": 1767036064273,
                        "sessionID": "ses_test123",
                        "part": {
                            "type": "step-finish",
                            "reason": "stop",
                            "cost": 0.001,
                            "tokens": {"input": 671, "output": 8},
                        },
                    }
                ),
            ]
        )
        to = agent.fetch_results(events)
        assert agent._session_id == "ses_test123"
        assert to.reply_text == "The command output was: hello"
        assert to.has_reply is True
        assert to.has_error is False
        assert len(to.tool_calls) == 1
        assert to.tool_calls[0].name == "bash"
        assert to.tool_sequence == ["bash"]
        assert to.cost_usd == pytest.approx(0.003)
        assert to.llm_turn_count == 2


# ── Lifecycle ──


class TestLifecycle:
    def test_setup_resets_state(self, agent: OpenCodeAgentAdapter, config: AgentConfig):
        agent._session_id = "old"
        agent._ttfe = 1.0
        agent._ttft = 2.0
        agent._ttlt = 3.0
        agent.setup(config)
        assert agent._session_id is None
        assert agent._ttfe is None
        assert agent._ttft is None
        assert agent._ttlt is None

    def test_teardown_resets_session_id(self, agent: OpenCodeAgentAdapter):
        agent._session_id = "active"
        agent.teardown()
        assert agent._session_id is None

    def test_metadata_with_session_id(self, agent: OpenCodeAgentAdapter):
        agent._session_id = "ses_abc"
        assert agent.metadata() == {"session_id": "ses_abc"}

    def test_metadata_without(self, agent: OpenCodeAgentAdapter):
        assert agent.metadata() is None

    def test_group_setup_returns_none(self, agent: OpenCodeAgentAdapter):
        gc = GroupConfig(agent="opencode")
        assert agent.setup_group(gc, Path("/tmp")) is None


# ── Interface methods ──


class TestInterfaceMethods:
    def test_cli_options_has_model(self):
        options = OpenCodeAgentAdapter.cli_options()
        assert len(options) == 1
        assert options[0].name == "model"
        assert options[0].env_var == "OPENCODE_DEFAULT_MODEL"

    def test_display_info_with_version(self):
        with patch("shutil.which", return_value="/usr/local/bin/opencode"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="0.2.5\n", stderr="")
                info = OpenCodeAgentAdapter.display_info()
                assert "opencode 0.2.5" in info

    def test_display_info_version_failure(self):
        with patch("shutil.which", return_value="/usr/local/bin/opencode"):
            with patch("subprocess.run", side_effect=Exception("fail")):
                info = OpenCodeAgentAdapter.display_info()
                assert "unknown" in info

    def test_display_info_not_installed(self):
        with patch("shutil.which", return_value=None):
            info = OpenCodeAgentAdapter.display_info()
            assert "not found" in info

    def test_supported_output_fields(self):
        fields = OpenCodeAgentAdapter.supported_output_fields()
        assert "tool_sequence" in fields
        assert "llm_turn_count" in fields

    def test_health_check_noop(self):
        OpenCodeAgentAdapter().health_check()

    def test_scoring_strategy(self):
        strategy = OpenCodeAgentAdapter().scoring_strategy()
        assert strategy is not None


# ── parse_stream_event ──


class TestParseStreamEvent:
    def test_tool_use_renders_name_and_args(self):
        event = {
            "type": "tool_use",
            "part": {
                "tool": "bash",
                "callID": "abc",
                "state": {"input": {"command": "ls -la"}},
            },
        }
        result = OpenCodeAgentAdapter.parse_stream_event(event)
        assert result is not None
        icon, summary = result
        assert icon == "🔧"
        assert "bash" in summary
        assert "ls -la" in summary

    def test_text_renders_content(self):
        event = {"type": "text", "part": {"text": "Hello world"}}
        result = OpenCodeAgentAdapter.parse_stream_event(event)
        assert result is not None
        assert result == ("💬", "Hello world")

    def test_step_start_suppressed(self):
        event = {"type": "step_start", "part": {"type": "step-start"}}
        result = OpenCodeAgentAdapter.parse_stream_event(event)
        assert result == ("", "")

    def test_step_finish_stop_shows_done(self):
        event = {"type": "step_finish", "part": {"reason": "stop", "cost": 0.003}}
        result = OpenCodeAgentAdapter.parse_stream_event(event)
        assert result is not None
        assert result[0] == "✅"
        assert "$0.0030" in result[1]

    def test_step_finish_tool_calls_suppressed(self):
        event = {"type": "step_finish", "part": {"reason": "tool-calls"}}
        result = OpenCodeAgentAdapter.parse_stream_event(event)
        assert result == ("", "")

    def test_error_renders(self):
        event = {
            "type": "error",
            "error": {"name": "APIError", "data": {"message": "Rate limit"}},
        }
        result = OpenCodeAgentAdapter.parse_stream_event(event)
        assert result is not None
        assert result[0] == "❌"
        assert "APIError" in result[1]

    def test_unknown_event_returns_none(self):
        event = {"type": "unknown_type", "part": {}}
        assert OpenCodeAgentAdapter.parse_stream_event(event) is None


class TestConstructor:
    def test_model_stored(self):
        a = OpenCodeAgentAdapter(model="openai/gpt-4.1")
        assert a._model == "openai/gpt-4.1"

    def test_model_passed_to_command(self):
        a = OpenCodeAgentAdapter(model="openai/gpt-4.1")
        proc = _mock_popen("ok\n")
        with patch("subprocess.Popen", return_value=proc) as mock_popen:
            a.execute("test", [])
            args = mock_popen.call_args[0][0]
            assert "--model" in args
            idx = args.index("--model")
            assert args[idx + 1] == "openai/gpt-4.1"
