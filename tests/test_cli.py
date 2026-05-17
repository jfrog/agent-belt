# (c) JFrog Ltd. (2026)

"""Tests for the unified belt CLI entry point."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from belt.cli import main


class TestCliEntryPoint:
    def test_no_command_returns_0(self, capsys):
        with patch("sys.argv", ["belt"]):
            result = main()
        assert result == 0
        out = capsys.readouterr().out
        assert "belt" in out.lower() or "usage" in out.lower()

    def test_version(self, capsys):
        with patch("sys.argv", ["belt", "--version"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "belt" in out

    def test_run_delegates(self):
        """Verify the run subcommand delegates to commands.run.main."""
        with patch("sys.argv", ["belt", "run", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_score_delegates(self):
        with patch("sys.argv", ["belt", "score", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_aggregate_delegates(self):
        with patch("sys.argv", ["belt", "aggregate", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_compare_delegates(self):
        with patch("sys.argv", ["belt", "compare", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_eval_delegates(self):
        """Verify the eval subcommand delegates to commands.eval.main."""
        with patch("sys.argv", ["belt", "eval", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_agent_list(self, capsys):
        with patch("sys.argv", ["belt", "agent", "list"]):
            result = main()
        assert result == 0
        # Agent table is UI, routed to stderr by ``eprint`` (belt._ui).
        # Stdout is kept clean so ``belt agent list | jq``-style pipelines
        # can be added without retrofitting later.
        err = capsys.readouterr().err
        assert "claude-code" in err

    def test_agent_info_known(self, capsys):
        with patch("sys.argv", ["belt", "agent", "info", "claude-code"]):
            result = main()
        assert result == 0
        err = capsys.readouterr().err
        assert "ClaudeCodeAgentAdapter" in err

    def test_agent_info_unknown(self, capsys):
        with patch("sys.argv", ["belt", "agent", "info", "nonexistent"]):
            result = main()
        assert result == 1
        err = capsys.readouterr().err
        assert "Unknown agent" in err or "Error" in err

    def test_agent_no_subcommand(self, capsys):
        with patch("sys.argv", ["belt", "agent"]):
            result = main()
        assert result == 0

    def test_doctor_delegates(self, capsys):
        with patch("sys.argv", ["belt", "doctor", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_quickstart_delegates(self, capsys):
        with patch("sys.argv", ["belt", "quickstart", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
