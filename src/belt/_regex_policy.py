# (c) JFrog Ltd. (2026)

"""Single source of truth for user-supplied scenario regexes.

Every regex an author writes in a scenario JSON (today: ``reply_pattern``
on :class:`belt.scenario.TurnExpectation` and the values of
``tool_result_pattern``) compiles through :func:`compile_user_regex`.
Centralising the compile step gives the schema one place to define:

* the default flag set (``re.IGNORECASE`` only - ``^`` and ``$`` anchor
  the full reply unless the author opts into per-line matching with the
  inline ``(?m)`` flag),
* the error contract (raise ``ValueError`` so the parse-time validator
  can aggregate every malformed entry into one report; never silently
  return ``False`` and ship a false-green to CI).

Both compile sites (the :func:`pydantic.field_validator` on
:class:`belt.scenario.TurnExpectation` and the runtime match in
:mod:`belt.scorer.rules`) consume the same compiled
:class:`re.Pattern`. Direct ``re.compile`` / ``re.search`` on values
sourced from :class:`belt.scenario.Scenario` /
:class:`belt.scenario.TurnExpectation` is forbidden anywhere else and
gated by ``tests/test_no_inline_user_regex.py``.

Lives at the top level (alongside :mod:`belt._safe`,
:mod:`belt._sanitize`, :mod:`belt._redact`) because the schema layer
and the scorer layer both consume it - putting it under either side
would invert the natural dependency direction. Underscore-prefixed
because it is an internal helper, not part of the published surface;
plugins that want regex assertions extend
:class:`belt.scenario.TurnExpectation` (via the ``extra='allow'``
plugin-extension surface), not this module.
"""

from __future__ import annotations

import re

USER_REGEX_FLAGS: int = re.IGNORECASE
"""Default flags applied to every user-supplied scenario regex.

Single-line semantics by default (``^`` / ``$`` anchor the full string).
Authors who need per-line matching opt in explicitly with the inline
``(?m)`` flag inside their pattern."""


def compile_user_regex(pattern: str) -> re.Pattern[str]:
    """Compile a user-supplied scenario regex with the canonical flag set.

    Raises :class:`ValueError` (with the offending pattern in the
    message) if ``pattern`` is not a valid Python regex. Callers in the
    Pydantic validator catch the per-entry ``ValueError`` and aggregate
    them into one error report so authors see every bad regex at once.
    """
    try:
        return re.compile(pattern, USER_REGEX_FLAGS)
    except re.error as exc:
        raise ValueError(f"invalid regex {pattern!r}: {exc}") from exc


__all__ = ["USER_REGEX_FLAGS", "compile_user_regex"]
