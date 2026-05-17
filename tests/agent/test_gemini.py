# (c) JFrog Ltd. (2026)

"""Tests for GeminiAgentAdapter - stream-json output parsing, multi-turn sessions,
structured tool calls, auth detection, timing, error handling, full lifecycle."""

from __future__ import annotations

import json
import subprocess
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from belt.agent.base import AgentNotAvailableError
from belt.agent.gemini import GeminiAgentAdapter
from belt.entities import AgentConfig, GroupConfig


@pytest.fixture
def agent() -> GeminiAgentAdapter:
    return GeminiAgentAdapter()


@pytest.fixture
def config() -> AgentConfig:
    return AgentConfig(
        group_config=GroupConfig(agent="gemini"),
        scenario_name="test",
    )


def _ndjson(*events: dict) -> str:
    return "\n".join(json.dumps(e) for e in events) + "\n"


# ── check_available ──


class TestCheckAvailable:
    """The check_available contract is binary-only - no model calls. Tests
    guard against regression to the old `gemini -p ping` probe that consumed
    credits and made doctor slow on the happy path."""

    def test_available_when_on_path(self):
        with patch("shutil.which", return_value="/usr/bin/gemini"):
            with patch("subprocess.run") as mock_run:
                GeminiAgentAdapter.check_available()
            mock_run.assert_not_called()

    def test_not_available_when_missing(self):
        with patch("shutil.which", return_value=None):
            with pytest.raises(AgentNotAvailableError, match="gemini CLI not found"):
                GeminiAgentAdapter.check_available()

    def test_no_subprocess_call_during_check(self):
        """The previous implementation invoked `gemini -p ping` (60s timeout,
        consumed model credits). The new contract forbids it."""
        with patch("shutil.which", return_value="/usr/bin/gemini"):
            with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
                GeminiAgentAdapter.check_available()
            mock_run.assert_not_called()
            mock_popen.assert_not_called()


class TestAuthSignals:
    def test_credential_env_includes_gemini_and_google_keys(self):
        assert "GEMINI_API_KEY" in GeminiAgentAdapter.CREDENTIAL_ENV
        assert "GOOGLE_API_KEY" in GeminiAgentAdapter.CREDENTIAL_ENV

    def test_credential_path_points_to_settings_json(self):
        assert any(".gemini" in str(p) for p in GeminiAgentAdapter.CREDENTIAL_PATHS)

    def test_signals_env_var(self, tmp_path, monkeypatch):
        for v in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "xxx")
        monkeypatch.setattr(GeminiAgentAdapter, "CREDENTIAL_PATHS", (tmp_path / "no-such",))
        assert "env GEMINI_API_KEY" in GeminiAgentAdapter.auth_signals()


# ── execute ──


class TestExecute:
    def _mock_popen(self, stdout_data: str, returncode: int = 0, stderr_data: str = ""):
        mock_proc = MagicMock()
        mock_proc.stdout = iter(stdout_data.splitlines(keepends=True))
        mock_proc.stderr = StringIO(stderr_data)
        mock_proc.returncode = returncode
        mock_proc.wait.return_value = None
        mock_proc.pid = 99999
        return mock_proc

    def test_uses_stream_json(self, agent: GeminiAgentAdapter):
        ndjson = _ndjson(
            {"type": "init", "session_id": "s1"},
            {"type": "result", "status": "success", "stats": {}},
        )
        mock_proc = self._mock_popen(ndjson)
        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            agent.execute("hello", [])
            args = mock_popen.call_args[0][0]
            assert "--output-format" in args
            assert "stream-json" in args

    def test_passes_extra_flags(self, agent: GeminiAgentAdapter):
        ndjson = _ndjson({"type": "init"}, {"type": "result", "status": "success", "stats": {}})
        mock_proc = self._mock_popen(ndjson)
        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            agent.execute("test", ["--model", "gemini-2.5-pro"])
            args = mock_popen.call_args[0][0]
            assert "--model" in args
            assert "gemini-2.5-pro" in args

    def test_resume_with_session_id(self, agent: GeminiAgentAdapter):
        agent._session_id = "prev-session"
        ndjson = _ndjson({"type": "init"}, {"type": "result", "status": "success", "stats": {}})
        mock_proc = self._mock_popen(ndjson)
        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            agent.execute("follow up", [])
            args = mock_popen.call_args[0][0]
            assert "--resume" in args
            assert "prev-session" in args

    def test_no_resume_on_first_turn(self, agent: GeminiAgentAdapter):
        ndjson = _ndjson({"type": "init"}, {"type": "result", "status": "success", "stats": {}})
        mock_proc = self._mock_popen(ndjson)
        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            agent.execute("first", [])
            args = mock_popen.call_args[0][0]
            assert "--resume" not in args


# ── fetch_results ──


class TestFetchResults:
    def test_parses_text_reply(self, agent: GeminiAgentAdapter):
        raw = _ndjson(
            {"type": "init", "session_id": "s1"},
            {"type": "message", "role": "model", "content": "Hello, world!"},
            {"type": "result", "status": "success", "stats": {"duration_ms": 500}},
        )
        to = agent.fetch_results(raw)
        assert to.reply_text == "Hello, world!"
        assert to.has_reply is True
        assert to.has_error is False

    def test_parses_structured_content_blocks(self, agent: GeminiAgentAdapter):
        raw = _ndjson(
            {"type": "init", "session_id": "s1"},
            {
                "type": "message",
                "role": "model",
                "content": [
                    {"type": "text", "text": "I'll read the file."},
                    {"type": "functionCall", "name": "read_file", "args": {"path": "main.py"}},
                ],
            },
            {"type": "result", "status": "success", "stats": {}},
        )
        to = agent.fetch_results(raw)
        assert "read the file" in to.reply_text
        assert len(to.tool_calls) == 1
        assert to.tool_calls[0].name == "read_file"
        assert to.tool_calls[0].args == {"path": "main.py"}
        assert to.tool_sequence == ["read_file"]

    def test_parses_tool_call_events(self, agent: GeminiAgentAdapter):
        raw = _ndjson(
            {"type": "init", "session_id": "s1"},
            {"type": "tool_call", "name": "write_file", "id": "tc1", "args": {"path": "out.txt"}},
            {"type": "result", "status": "success", "stats": {}},
        )
        to = agent.fetch_results(raw)
        assert len(to.tool_calls) == 1
        assert to.tool_calls[0].name == "write_file"
        assert to.tool_calls[0].call_id == "tc1"
        assert to.tool_sequence == ["write_file"]

    def test_captures_session_id(self, agent: GeminiAgentAdapter):
        raw = _ndjson(
            {"type": "init", "session_id": "abc-123"},
            {"type": "result", "status": "success", "stats": {}},
        )
        agent.fetch_results(raw)
        assert agent._session_id == "abc-123"

    def test_multi_turn_session_persists(self, agent: GeminiAgentAdapter):
        turn1 = _ndjson(
            {"type": "init", "session_id": "session-1"},
            {"type": "message", "role": "model", "content": "Turn 1"},
            {"type": "result", "status": "success", "stats": {}},
        )
        agent.fetch_results(turn1)
        assert agent._session_id == "session-1"

        turn2 = _ndjson(
            {"type": "init", "session_id": "session-1"},
            {"type": "message", "role": "model", "content": "Turn 2"},
            {"type": "result", "status": "success", "stats": {}},
        )
        to2 = agent.fetch_results(turn2)
        assert to2.reply_text == "Turn 2"

    def test_error_result(self, agent: GeminiAgentAdapter):
        # Vendor-specific tokens like ``"QuotaError"`` are normalised
        # to the canonical taxonomy in :mod:`belt.entities`.
        # ``"Rate limited"`` in the error message text matches the
        # rate-limit pattern, so the framework surfaces
        # ``RATE_LIMITED`` regardless of the upstream label.
        from belt.entities import RATE_LIMITED

        raw = _ndjson(
            {"type": "init", "session_id": "s1"},
            {
                "type": "result",
                "status": "error",
                "error": {"type": "QuotaError", "message": "Rate limited"},
                "stats": {"duration_ms": 100},
            },
        )
        to = agent.fetch_results(raw)
        assert to.has_error is True
        assert to.error_type == RATE_LIMITED

    def test_error_string_format(self, agent: GeminiAgentAdapter):
        # Vendor tokens that don't match any pattern (and whose
        # surrounding text doesn't either) fall back to the canonical
        # ``UNKNOWN`` sentinel so consumers can rely on the token set.
        from belt.entities import UNKNOWN

        raw = _ndjson(
            {"type": "init"},
            {"type": "result", "status": "error", "error": "Something failed", "stats": {}},
        )
        to = agent.fetch_results(raw)
        assert to.has_error is True
        assert to.error_type == UNKNOWN

    def test_timing_from_result_stats(self, agent: GeminiAgentAdapter):
        agent._ttlt = 2.5
        agent._ttfe = 0.1
        agent._ttft = 0.3
        raw = _ndjson(
            {"type": "init"},
            {"type": "result", "status": "success", "stats": {"duration_ms": 2000}},
        )
        to = agent.fetch_results(raw)
        assert to.timing is not None
        assert to.timing.total == 2.5
        assert to.timing.ttfe == 0.1
        assert to.timing.ttft == 0.3

    def test_timing_fallback_to_api(self, agent: GeminiAgentAdapter):
        agent._ttlt = None
        raw = _ndjson(
            {"type": "init"},
            {"type": "result", "status": "success", "stats": {"duration_ms": 3000}},
        )
        to = agent.fetch_results(raw)
        assert to.timing is not None
        assert to.timing.total == 3.0

    def test_empty_output(self, agent: GeminiAgentAdapter):
        to = agent.fetch_results("")
        assert to.reply_text == ""
        assert to.has_reply is False
        assert to.has_error is None

    def test_raw_cli_preserved(self, agent: GeminiAgentAdapter):
        raw = _ndjson({"type": "init"}, {"type": "result", "status": "success", "stats": {}})
        to = agent.fetch_results(raw)
        assert to.raw_cli == raw

    def test_ignores_user_messages(self, agent: GeminiAgentAdapter):
        raw = _ndjson(
            {"type": "init"},
            {"type": "message", "role": "user", "content": "user input"},
            {"type": "message", "role": "model", "content": "model reply"},
            {"type": "result", "status": "success", "stats": {}},
        )
        to = agent.fetch_results(raw)
        assert to.reply_text == "model reply"

    def test_parses_assistant_role_reply(self, agent: GeminiAgentAdapter):
        """Current Gemini CLI emits ``role: "assistant"``.

        The parser accepts both ``"model"`` (legacy) and ``"assistant"`` (current).
        Without this, ``has_reply`` was false even on successful runs and rule
        scoring failed for every Gemini scenario."""
        raw = _ndjson(
            {"type": "init", "session_id": "s1"},
            {"type": "message", "role": "assistant", "content": "Paris", "delta": True},
            {"type": "result", "status": "success", "stats": {"duration_ms": 100}},
        )
        to = agent.fetch_results(raw)
        assert to.reply_text == "Paris"
        assert to.has_reply is True

    def test_captures_all_assistant_delta_messages(self, agent: GeminiAgentAdapter):
        """Streaming Gemini CLI emits multiple ``delta=True`` chunks per turn,
        and each must contribute to the final ``reply_text``. Pre-fix, all chunks
        were silently dropped because role was ``"assistant"``, not ``"model"``,
        leaving ``reply_text`` empty.

        Parts are joined with newlines (existing behavior - orthogonal to
        the role-name parsing fix), so we assert each chunk's substring is
        present rather than an exact concatenation."""
        raw = _ndjson(
            {"type": "init", "session_id": "s1"},
            {"type": "message", "role": "assistant", "content": "This function is an", "delta": True},
            {"type": "message", "role": "assistant", "content": " implementation of", "delta": True},
            {"type": "message", "role": "assistant", "content": " binary search.", "delta": True},
            {"type": "result", "status": "success", "stats": {}},
        )
        to = agent.fetch_results(raw)
        assert to.has_reply is True
        assert "This function is an" in to.reply_text
        assert "implementation of" in to.reply_text
        assert "binary search." in to.reply_text

    def test_assistant_role_with_structured_content_blocks(self, agent: GeminiAgentAdapter):
        """Assistant role + structured content (text + functionCall) must be
        parsed identically to the legacy model role."""
        raw = _ndjson(
            {"type": "init", "session_id": "s1"},
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Reading the file."},
                    {"type": "functionCall", "name": "read_file", "args": {"path": "main.py"}},
                ],
            },
            {"type": "result", "status": "success", "stats": {}},
        )
        to = agent.fetch_results(raw)
        assert "Reading the file" in to.reply_text
        assert to.tool_sequence == ["read_file"]
        assert to.tool_calls[0].args == {"path": "main.py"}

    def test_mixed_model_and_assistant_roles_both_captured(self, agent: GeminiAgentAdapter):
        """Defensive: if a single transcript mixes both role values
        (e.g. CLI version transition mid-session), both contribute."""
        raw = _ndjson(
            {"type": "init", "session_id": "s1"},
            {"type": "message", "role": "model", "content": "First "},
            {"type": "message", "role": "assistant", "content": "second."},
            {"type": "result", "status": "success", "stats": {}},
        )
        to = agent.fetch_results(raw)
        assert "First" in to.reply_text
        assert "second." in to.reply_text
        assert to.has_reply is True

    def test_multiple_tool_calls_ordered(self, agent: GeminiAgentAdapter):
        raw = _ndjson(
            {"type": "init"},
            {"type": "tool_call", "name": "read_file", "id": "1", "args": {}},
            {"type": "tool_call", "name": "edit_file", "id": "2", "args": {}},
            {"type": "tool_call", "name": "run_tests", "id": "3", "args": {}},
            {"type": "result", "status": "success", "stats": {}},
        )
        to = agent.fetch_results(raw)
        assert to.tool_sequence == ["read_file", "edit_file", "run_tests"]
        assert len(to.tool_calls) == 3


# ── Lifecycle ──


class TestLifecycle:
    def test_setup_resets_state(self, agent: GeminiAgentAdapter, config: AgentConfig):
        agent._session_id = "old"
        agent.setup(config)
        assert agent._session_id is None

    def test_teardown_clears_session(self, agent: GeminiAgentAdapter):
        agent._session_id = "something"
        agent.teardown()
        assert agent._session_id is None

    def test_metadata_with_session(self, agent: GeminiAgentAdapter):
        agent._session_id = "s1"
        meta = agent.metadata()
        assert meta == {"session_id": "s1"}

    def test_metadata_without_session(self, agent: GeminiAgentAdapter):
        assert agent.metadata() is None

    def test_group_setup_returns_none(self, agent: GeminiAgentAdapter):
        gc = GroupConfig(agent="gemini")
        assert agent.setup_group(gc, Path("/tmp")) is None

    def test_group_setup_summary_none(self, agent: GeminiAgentAdapter):
        assert agent.group_setup_summary(None) is None

    def test_teardown_group_noop(self, agent: GeminiAgentAdapter):
        agent.teardown_group(None)


# ── Interface methods ──


class TestInterfaceMethods:
    def test_cli_options_is_empty(self):
        # Parameterless agent: scenarios pin a model via ``flags``
        # (e.g. ``--model gemini-2.5-pro``). See ``test_passes_extra_flags``.
        assert GeminiAgentAdapter.cli_options() == []

    def test_display_info_with_gemini(self):
        with patch("shutil.which", return_value="/usr/bin/gemini"):
            result = subprocess.CompletedProcess(args=[], returncode=0, stdout="1.2.3\n", stderr="")
            with patch("subprocess.run", return_value=result):
                info = GeminiAgentAdapter.display_info()
                assert "1.2.3" in info

    def test_display_info_without_gemini(self):
        with patch("shutil.which", return_value=None):
            info = GeminiAgentAdapter.display_info()
            assert "not found" in info

    def test_health_check_noop(self):
        GeminiAgentAdapter().health_check()

    def test_scoring_strategy(self):
        strategy = GeminiAgentAdapter().scoring_strategy()
        assert strategy is not None
        assert len(strategy.dimensions) > 0


class TestConstructor:
    def test_initial_state(self):
        a = GeminiAgentAdapter()
        assert a._session_id is None
        assert a._ttfe is None
