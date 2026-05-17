# (c) JFrog Ltd. (2026)

"""Unit tests for rule-based scorer helpers (``result_contains``, ``result_matches``).

Higher-level integration with ``check_trajectory`` lives in
``test_trajectory.py``; this file pins the helper contract directly so a
behavior change shows up here first.
"""

from __future__ import annotations

from belt._regex_policy import compile_user_regex
from belt.entities import ToolCall
from belt.scorer.rules.helpers import result_contains, result_matches


def _tc(result: dict | None = None) -> ToolCall:
    return ToolCall(name="Read", call_id="c1", args={}, result=result)


# -- result_contains --------------------------------------------------------


def test_result_contains_happy_path() -> None:
    assert result_contains(_tc({"output": 'name = "belt"'}), "belt") is True


def test_result_contains_none_result_returns_false() -> None:
    assert result_contains(_tc(None), "anything") is False


def test_result_contains_empty_dict_returns_false_when_substring_nonempty() -> None:
    assert result_contains(_tc({}), "belt") is False


def test_result_contains_case_insensitive() -> None:
    assert result_contains(_tc({"output": "BELT"}), "belt") is True
    assert result_contains(_tc({"output": "belt"}), "BELT") is True


def test_result_contains_nested_dict() -> None:
    # Nested string LEAVES are flattened and matched verbatim.
    # ``"belt"`` lives at depth-3 and must reach the matcher without
    # escape-encoding.
    nested = {"data": {"meta": {"project": "belt"}, "rows": [1, 2, 3]}}
    assert result_contains(_tc(nested), "belt") is True


def test_result_contains_skips_dict_keys() -> None:
    """Recursive flatten yields VALUES only - keys are structural, not content.

    Authors who need key-shape matching should use the structured assertions
    (``tool_args_contain``, ``args_match``) or check the JSON-encoded full
    result via a tool that exposes raw bytes; mixing key/value matching here
    was the pre-fix behaviour that this contract rules out.
    """
    nested = {"data": {"meta": {"project": "belt"}, "rows": [1, 2, 3]}}
    assert result_contains(_tc(nested), "project") is False
    assert result_contains(_tc(nested), "meta") is False


def test_result_contains_missing_substring_returns_false() -> None:
    assert result_contains(_tc({"output": "something else"}), "belt") is False


def test_result_contains_handles_non_serializable_values() -> None:
    """Non-JSON-serializable values fall back to ``str()`` rather than raising."""

    class Custom:
        def __str__(self) -> str:
            return "belt-fallback"

    # ``json.dumps`` with ``default=str`` handles this; the helper must not raise.
    assert result_contains(_tc({"obj": Custom()}), "belt-fallback") is True


# -- result_matches ---------------------------------------------------------
#
# ``result_matches`` takes a pre-compiled :class:`re.Pattern`. Validation
# happens at scenario-load via ``belt._regex_policy.compile_user_regex``
# and the ``TurnExpectation`` validators, so the patterns reaching this
# helper are always well-formed.


def test_result_matches_happy_path() -> None:
    assert result_matches(_tc({"output": 'name = "belt"'}), compile_user_regex(r'name\s*=\s*"belt"')) is True


def test_result_matches_none_result_returns_false() -> None:
    assert result_matches(_tc(None), compile_user_regex(r".*")) is False


def test_result_matches_case_insensitive() -> None:
    """The compiled pattern carries ``re.IGNORECASE`` from the policy module."""
    assert result_matches(_tc({"output": "Belt"}), compile_user_regex(r"belt")) is True


def test_result_matches_nested_string_leaf_verbatim() -> None:
    """Cursor-shaped ``{"success": {"content": ...}}`` regression guard.

    Prior to the leaf-flattening contract, ``_iter_result_strings`` json.dumps'd the whole
    sub-dict, turning the inner ``"`` into ``\\"`` and silently breaking
    author regexes like ``r'name = "agent-belt"'``. The leaf must reach
    the matcher unescaped.
    """
    nested = {"success": {"content": 'name = "agent-belt"'}}
    assert result_matches(_tc(nested), compile_user_regex(r'name\s*=\s*"agent-belt"')) is True


def test_result_matches_does_not_see_dict_keys() -> None:
    """Recursive flatten emits leaf STRING values, not the structural keys."""
    nested = {"meta": {"project": "belt"}}
    assert result_matches(_tc(nested), compile_user_regex(r'"project"\s*:')) is False
    # The leaf value itself is still reachable verbatim.
    assert result_matches(_tc(nested), compile_user_regex(r"^belt$")) is True


def test_result_matches_list_of_strings() -> None:
    """Lists/tuples are walked: each string leaf is yielded individually."""
    payload = {"lines": ["alpha", 'name = "belt"', "omega"]}
    assert result_matches(_tc(payload), compile_user_regex(r'name\s*=\s*"belt"')) is True


def test_result_matches_depth_cap_falls_back_to_json() -> None:
    """Beyond ``_MAX_RESULT_DEPTH`` the subtree is json-dumped so we never recurse forever.

    This protects against pathological ``{"a": {"a": ...}}`` payloads
    without losing visibility of the content - the regex can still match
    the JSON-encoded fallback string.
    """
    from belt.scorer.rules.helpers import _MAX_RESULT_DEPTH

    # Build a chain of dicts deeper than the cap; the innermost leaf
    # ``"belt"`` sits past the recursion bound.
    deep: object = "belt"
    for _ in range(_MAX_RESULT_DEPTH + 5):
        deep = {"x": deep}
    # The fallback json.dumps emits ``{"x": ...}`` so ``belt`` is still
    # findable as a substring of the encoded blob.
    assert result_matches(_tc({"root": deep}), compile_user_regex(r"belt")) is True


def test_result_matches_no_match_returns_false() -> None:
    assert result_matches(_tc({"output": "hello world"}), compile_user_regex(r"^bye")) is False


def test_result_matches_uses_search_not_fullmatch() -> None:
    """Anchoring is the author's responsibility; ``re.search`` is the contract."""
    assert result_matches(_tc({"output": "prefix belt suffix"}), compile_user_regex(r"belt")) is True
