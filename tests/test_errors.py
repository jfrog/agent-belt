# (c) JFrog Ltd. (2026)

"""Tests for error hierarchy and CLI exception boundary."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from belt.errors import AgentExecutionError, BeltError, ConfigError, ScenarioError, ScorerError


class TestErrorHierarchy:
    def test_all_inherit_from_base(self):
        for cls in (ConfigError, AgentExecutionError, ScorerError, ScenarioError):
            assert issubclass(cls, BeltError)

    def test_base_inherits_from_exception(self):
        assert issubclass(BeltError, Exception)

    def test_catch_by_base(self):
        with pytest.raises(BeltError):
            raise ConfigError("bad config")


class TestCliBoundary:
    def test_known_error_returns_1(self, capsys):
        from belt.cli import main

        with patch("belt.cli._dispatch", side_effect=ConfigError("bad config")):
            result = main()
        assert result == 1
        err = capsys.readouterr().err
        assert "bad config" in err

    def test_unexpected_error_returns_1(self, capsys):
        from belt.cli import main

        with patch("belt.cli._dispatch", side_effect=KeyError("missing")):
            result = main()
        assert result == 1
        err = capsys.readouterr().err
        assert "KeyError" in err
        assert "BELT_DEBUG" in err

    def test_keyboard_interrupt_returns_130(self, capsys):
        from belt.cli import main

        with patch("belt.cli._dispatch", side_effect=KeyboardInterrupt):
            result = main()
        assert result == 130

    def test_debug_mode_shows_traceback(self, capsys):
        from belt.cli import main

        with patch("belt.cli._dispatch", side_effect=ValueError("boom")):
            with patch.dict("os.environ", {"BELT_DEBUG": "1"}):
                result = main()
        assert result == 1
        err = capsys.readouterr().err
        assert "Traceback" in err

    def test_normal_mode_no_traceback(self, capsys):
        from belt.cli import main

        with patch("belt.cli._dispatch", side_effect=ValueError("boom")):
            with patch.dict("os.environ", {}, clear=False):
                import os

                os.environ.pop("BELT_DEBUG", None)
                result = main()
        assert result == 1
        err = capsys.readouterr().err
        assert "Traceback" not in err
