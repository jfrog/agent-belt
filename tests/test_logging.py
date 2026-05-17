# (c) JFrog Ltd. (2026)

"""Tests for the terminal-log level resolver.

Three input channels feed into the terminal handler level: ``-v`` /
``-vv`` from the CLI, ``BELT_LOG_LEVEL`` from the environment, and the
default. Precedence and case-insensitive parsing are pinned here so a
later refactor cannot quietly let an invalid env var silence the run.
"""

from __future__ import annotations

import pytest

from belt import envvars
from belt._logging import _VALID_LEVELS, configure_terminal_logging, resolve_terminal_level


@pytest.fixture(autouse=True)
def _clear_log_level_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(envvars.LOG_LEVEL, raising=False)


class TestResolveTerminalLevelDefaults:
    def test_no_flags_no_env_returns_warning(self) -> None:
        assert resolve_terminal_level(0) == "WARNING"


class TestResolveTerminalLevelVerboseFlag:
    """``-v`` accumulation maps to INFO/DEBUG and clamps above ``-vv``."""

    def test_v_returns_info(self) -> None:
        assert resolve_terminal_level(1) == "INFO"

    def test_vv_returns_debug(self) -> None:
        assert resolve_terminal_level(2) == "DEBUG"

    def test_vvv_clamps_to_debug(self) -> None:
        # ``-v`` accumulation beyond ``-vv`` must not raise - clamp to
        # the most-verbose supported level instead.
        assert resolve_terminal_level(5) == "DEBUG"

    def test_verbose_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Explicit ``-v`` always wins over the env var, mirroring the CLI
        # convention that command-line flags trump shell exports.
        monkeypatch.setenv(envvars.LOG_LEVEL, "ERROR")
        assert resolve_terminal_level(1) == "INFO"


class TestResolveTerminalLevelEnv:
    """``BELT_LOG_LEVEL`` is honoured when ``-v`` was not passed."""

    @pytest.mark.parametrize("level", sorted(_VALID_LEVELS))
    def test_every_valid_level_round_trips(self, level: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(envvars.LOG_LEVEL, level)
        assert resolve_terminal_level(0) == level

    def test_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(envvars.LOG_LEVEL, "debug")
        assert resolve_terminal_level(0) == "DEBUG"

    def test_invalid_level_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A typo on a shared host must not silence the run by collapsing
        # to ``CRITICAL`` (or raising). Falling back to ``WARNING`` keeps
        # the user-visible behaviour predictable.
        monkeypatch.setenv(envvars.LOG_LEVEL, "VERBOSE")
        assert resolve_terminal_level(0) == "WARNING"

    def test_empty_env_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(envvars.LOG_LEVEL, "")
        assert resolve_terminal_level(0) == "WARNING"


class TestConfigureTerminalLogging:
    """The convenience wrapper returns the resolved level so the caller
    (e.g. the eval post-run banner) can record it without re-deriving."""

    def test_returns_resolved_level(self) -> None:
        assert configure_terminal_logging(0) == "WARNING"
        assert configure_terminal_logging(1) == "INFO"
        assert configure_terminal_logging(2) == "DEBUG"
