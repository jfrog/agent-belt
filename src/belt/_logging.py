# (c) JFrog Ltd. (2026)

"""Two-level logging: lean terminal handler + always-on file handler.

`belt` borrows the Inspect AI / Promptfoo split: the terminal is a
scoreboard, the on-disk run is the canonical artifact. The terminal
handler defaults to `WARNING` so the user sees clean progress + a
results panel; the file handler attached by ``commands/run.py`` always
runs at ``DEBUG`` so ``<run_dir>/eval.log`` carries full forensics
regardless of how quiet the terminal is.

The terminal level is set once at CLI dispatch and honoured by every
subcommand that imports ``loguru.logger``. Override precedence (highest
wins): the ``-v`` / ``-vv`` flag, then ``BELT_LOG_LEVEL``, then the
default ``WARNING``.
"""

from __future__ import annotations

import sys
from typing import Final

from loguru import logger

from belt import envvars

# Mapping from ``-v`` count to loguru level. ``0`` is the default (the
# user passed no ``-v`` and set no env var); ``1`` is ``INFO`` (today's
# wall-of-text inline), ``2`` is ``DEBUG`` (trace-level diagnostics).
# Counts beyond ``2`` clamp to ``DEBUG`` rather than raising, matching
# ``pytest``-style ``-v`` accumulation semantics.
_VERBOSE_LEVEL_MAP: Final[dict[int, str]] = {
    0: "WARNING",
    1: "INFO",
    2: "DEBUG",
}

# Loguru level names accepted from ``BELT_LOG_LEVEL``. Listed alphabetically
# for predictable error messages.
_VALID_LEVELS: Final[frozenset[str]] = frozenset({"CRITICAL", "DEBUG", "ERROR", "INFO", "SUCCESS", "TRACE", "WARNING"})


def resolve_terminal_level(verbose_count: int = 0) -> str:
    """Return the loguru level the terminal handler should run at.

    ``verbose_count`` from ``-v`` (``1``) / ``-vv`` (``2``) takes priority
    over ``BELT_LOG_LEVEL``; bare invocation (``0``) defers to the env
    var, then the default. Invalid env values are ignored so a typo on a
    shared host cannot silence the run.
    """
    if verbose_count > 0:
        return _VERBOSE_LEVEL_MAP.get(verbose_count, "DEBUG")
    env_raw = envvars.get_str(envvars.LOG_LEVEL).strip().upper()
    if env_raw and env_raw in _VALID_LEVELS:
        return env_raw
    return _VERBOSE_LEVEL_MAP[0]


def configure_terminal_logging(verbose_count: int = 0) -> str:
    """Replace loguru's default sink with one tied to the resolved level.

    Returns the level name that was applied so callers can record it
    (e.g. for the post-run banner) without re-deriving it.
    """
    level = resolve_terminal_level(verbose_count)
    logger.remove()
    logger.add(sys.stderr, level=level, format=_TERMINAL_FORMAT)
    return level


# Compact format: ``HH:MM:SS LEVEL message``. No module path or line
# number on the terminal handler; the file handler keeps the default
# loguru format with the full diagnostic chain.
_TERMINAL_FORMAT: Final[str] = "<dim>{time:HH:mm:ss}</dim> <level>{level: <7}</level> {message}"
