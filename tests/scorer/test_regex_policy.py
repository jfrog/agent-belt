# (c) JFrog Ltd. (2026)

"""Tests for the user-supplied regex policy module.

This module is the single source of truth for compiling scenario
regexes (``reply_pattern`` on :class:`TurnExpectation`,
``tool_result_pattern`` on the same). The policy is small but the
contract is load-bearing: a typo in a user regex must surface as a
loud parse-time error, never a silent-``False`` at score time. We
pin the contract directly here AND re-pin it via the validator
integration tests in ``tests/parser/test_scenario.py``.
"""

from __future__ import annotations

import re

import pytest

from belt._regex_policy import USER_REGEX_FLAGS, compile_user_regex


def test_compile_user_regex_returns_compiled_pattern() -> None:
    compiled = compile_user_regex(r"^ORDER\|")
    assert isinstance(compiled, re.Pattern)
    assert compiled.search("ORDER|42|...") is not None


def test_compile_user_regex_default_flags_are_ignorecase_only() -> None:
    """Default flag set is exactly ``re.IGNORECASE`` - no ``re.MULTILINE``.

    Documented contract: ``^`` / ``$`` anchor the full string by default;
    authors opt into per-line matching with the inline ``(?m)`` flag.
    Pinning the flags constant catches any future caller that "helpfully"
    adds ``re.MULTILINE`` and silently loosens the assertion.
    """
    assert USER_REGEX_FLAGS == re.IGNORECASE
    compiled = compile_user_regex(r"anything")
    assert compiled.flags & re.IGNORECASE
    assert not (compiled.flags & re.MULTILINE)


def test_compile_user_regex_is_case_insensitive() -> None:
    compiled = compile_user_regex(r"belt")
    assert compiled.search("BELT") is not None
    assert compiled.search("Belt") is not None


def test_compile_user_regex_inline_multiline_flag_opt_in() -> None:
    """Authors who want per-line ``^``/``$`` opt in with ``(?m)``.

    Without the inline flag, ``^foo$`` matches the full reply; with
    ``(?m)`` it matches any individual line. This locks the documented
    "no per-line magic by default" contract.
    """
    multi = "first line\nfoo\nthird"
    assert compile_user_regex(r"^foo$").search(multi) is None
    assert compile_user_regex(r"(?m)^foo$").search(multi) is not None


def test_compile_user_regex_raises_value_error_on_bad_regex() -> None:
    with pytest.raises(ValueError) as exc_info:
        compile_user_regex("[unclosed")
    msg = str(exc_info.value)
    assert "[unclosed" in msg
    assert "invalid regex" in msg


def test_compile_user_regex_value_error_chains_re_error() -> None:
    """The original ``re.error`` is preserved as ``__cause__``.

    Lets debugging tooling inspect the underlying parser error
    (offset, message) without re-running the compile.
    """
    with pytest.raises(ValueError) as exc_info:
        compile_user_regex("(?P<bad")
    assert isinstance(exc_info.value.__cause__, re.error)
