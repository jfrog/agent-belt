# (c) JFrog Ltd. (2026)

"""Shared helper functions for rule-based checks."""

from __future__ import annotations

import json
import re
from typing import Any

from belt.entities import ToolCall


def tool_name_in_cli(tool: str, raw_cli: str) -> bool:
    """Check if a tool name appears as a whole word in raw CLI output."""
    return bool(re.search(r"\b" + re.escape(tool) + r"\b", raw_cli))


def is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    """Check if needle appears as a subsequence (not necessarily contiguous) in haystack."""
    it = iter(haystack)
    return all(item in it for item in needle)


def args_match(actual: dict, expected: dict) -> bool:
    """Check if all expected key-value pairs are present in actual args (substring match on values)."""
    for key, val in expected.items():
        actual_val = actual.get(key)
        if actual_val is None:
            return False
        if str(val).lower() not in str(actual_val).lower():
            return False
    return True


# Practical upper bound for real coding-agent tool returns. A pathological
# dict like ``{"a": {"a": {...}}}`` 10K deep would otherwise consume Python
# recursion budget here; cap and fall back to a one-shot ``json.dumps`` of
# the subtree. Cycles are not possible by construction (``tc.result`` is
# JSON-parsed and JSON has no references).
_MAX_RESULT_DEPTH = 10


def _iter_result_strings(tc: ToolCall) -> list[str]:
    """Render a tool call's result as a list of strings for substring/regex matching.

    Walks the result recursively and yields every string leaf verbatim so
    author-supplied regexes do not need to escape JSON quote encoding. This
    matters for nested structured returns like the cursor ``Read`` tool's
    ``{"success": {"content": 'name = "agent-belt"'}}`` shape: yielding the
    inner ``'name = "agent-belt"'`` string raw lets ``"agent-belt"`` in the
    regex match a literal ``"`` instead of the JSON-encoded ``\\"``.

    Non-string scalars (numbers, bools, None) are JSON-encoded so authors can
    still reach into structured returns. Empty when ``result`` was not
    captured. Mirrors the convention used by ``_any_arg_contains``.
    """
    if tc.result is None:
        return []
    out: list[str] = []

    def _walk(node: Any, depth: int) -> None:
        if depth >= _MAX_RESULT_DEPTH:
            try:
                out.append(json.dumps(node, ensure_ascii=False, default=str))
            except (TypeError, ValueError):
                out.append(str(node))
            return
        if isinstance(node, str):
            out.append(node)
        elif isinstance(node, dict):
            for v in node.values():
                _walk(v, depth + 1)
        elif isinstance(node, (list, tuple)):
            for v in node:
                _walk(v, depth + 1)
        else:
            try:
                out.append(json.dumps(node, ensure_ascii=False, default=str))
            except (TypeError, ValueError):
                out.append(str(node))

    _walk(tc.result, 0)
    return out


def result_contains(tc: ToolCall, substring: str) -> bool:
    """Check if any top-level value in a tool call's result contains the substring (case-insensitive)."""
    needle = substring.lower()
    return any(needle in text.lower() for text in _iter_result_strings(tc))


def result_matches(tc: ToolCall, compiled: re.Pattern[str]) -> bool:
    """Check if any top-level value in a tool call's result matches ``compiled``.

    The caller passes a pre-compiled :class:`re.Pattern` (cached on the
    :class:`belt.scenario.TurnExpectation` after parse-time validation
    in :mod:`belt._regex_policy`). Centralising the compile step there
    means a malformed regex aborts scenario load with one aggregated
    error report - never a silent ``False`` at score time.
    """
    return any(compiled.search(text) is not None for text in _iter_result_strings(tc))


_SKILL_PATH_RE = re.compile(r"skills/([^/]+)")


def skill_invoked(skill_name: str, tool_calls: list[ToolCall], raw_cli: str) -> tuple[bool, str]:
    """Detect whether a specific skill was invoked, regardless of agent.

    Detection tiers (checked in order):
      1. Claude Code: ``Skill`` tool with args["skill"] matching
      2. Path match: any tool arg value containing ``skills/<name>``
      3. CLI fallback: raw CLI text containing ``skills/<name>``

    Returns (found, detail_string).
    """
    # Tier 1: first-class Skill tool (Claude Code)
    for tc in tool_calls:
        if tc.name == "Skill" and _arg_value_matches(tc.args, "skill", skill_name):
            return True, "via Skill tool"

    # Tier 2: path-based detection in any tool arg
    # Covers Cursor (Read), Gemini (read_file), Codex (shell), etc.
    pattern = f"skills/{skill_name}"
    for tc in tool_calls:
        if _any_arg_contains(tc.args, pattern):
            return True, f"via {tc.name} tool args"

    # Tier 3: raw CLI fallback
    if pattern in raw_cli:
        return True, "via cli fallback"

    return False, ""


def _arg_value_matches(args: dict[str, Any], key: str, expected: str) -> bool:
    """Check if args[key] matches the expected name (case-insensitive).

    Accepts two shapes Claude Code emits for skill names:
      1. Direct match: ``args["skill"] == "<expected>"`` for top-level skills
         (``.claude/skills/<name>/SKILL.md``).
      2. Plugin-prefixed match: ``args["skill"] == "<plugin>:<expected>"`` for
         plugin-bundled skills (``.claude/plugins/<plugin>/skills/<name>/SKILL.md``).
         Claude prefixes plugin-bundled skill invocations with the plugin
         namespace so they can't collide with top-level skills.

    Matching the prefixed shape keeps ``skill_invoked: ["processing-watch"]``
    a Tier-1 (first-class Skill-tool) match for plugin skills, rather than
    falling through to the weaker Tier-3 (raw-CLI banner) fallback.
    """
    val = args.get(key)
    if val is None:
        return False
    val_lower = str(val).lower()
    expected_lower = expected.lower()
    if val_lower == expected_lower:
        return True
    # Plugin-prefixed: "<plugin>:<expected>"
    return ":" in val_lower and val_lower.split(":", 1)[1] == expected_lower


def _any_arg_contains(args: dict[str, Any], substring: str) -> bool:
    """Check if any arg value (stringified, including nested) contains substring."""
    needle = substring.lower()
    for val in args.values():
        text = json.dumps(val).lower() if not isinstance(val, str) else val.lower()
        if needle in text:
            return True
    return False
