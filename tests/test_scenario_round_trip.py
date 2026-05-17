# (c) JFrog Ltd. (2026)

"""Round-trip tests for ``TurnExpectation`` regex caching.

The compiled-regex caches on :class:`belt.scenario.TurnExpectation`
(populated by ``model_post_init`` after the field validators run) live
on private attributes so that:

1. The on-disk JSON contract stays the source pattern strings - never
   :class:`re.Pattern` objects, which would be (a) unserialisable and
   (b) leaky implementation detail in any persisted scenario file.
2. ``model_dump`` -> ``model_validate`` round-trips losslessly: the
   re-parsed instance recompiles the cache from the dumped strings.
3. The two compile paths (the field validator and the cache populator)
   share one source of truth in :mod:`belt._regex_policy`, so the cached
   :class:`re.Pattern` flags are guaranteed to match what runtime
   matching uses.

This file pins those invariants directly; without them, a future change
that "helpfully" exposes the cache as a public field would silently
break every persisted scenario JSON in the repo.
"""

from __future__ import annotations

import re

from belt.scenario import TurnExpectation


def test_turn_expectation_round_trips_reply_pattern_strings() -> None:
    original = TurnExpectation(reply_pattern=[r"^ORDER\|", r"\|shipped\|"])
    dumped = original.model_dump_json()
    re_parsed = TurnExpectation.model_validate_json(dumped)
    assert re_parsed.reply_pattern == original.reply_pattern


def test_turn_expectation_round_trips_tool_result_pattern_dict() -> None:
    original = TurnExpectation(tool_result_pattern={"Read": r"^OK$", "Write": r"\d+"})
    dumped = original.model_dump_json()
    re_parsed = TurnExpectation.model_validate_json(dumped)
    assert re_parsed.tool_result_pattern == original.tool_result_pattern


def test_dumped_json_does_not_contain_compiled_pattern_objects() -> None:
    """The on-disk shape is the source pattern strings, full stop.

    Compiled :class:`re.Pattern` objects living on private attrs must
    not leak into ``model_dump``/``model_dump_json``. A leak here would
    (a) make the JSON unserialisable in many cases and (b) couple the
    on-disk contract to CPython's regex implementation - exactly the
    drift this design exists to prevent.
    """
    te = TurnExpectation(
        reply_pattern=[r"^ORDER\|"],
        tool_result_pattern={"echo": r"^OK$"},
    )
    dumped = te.model_dump()
    assert "_compiled_reply_patterns" not in dumped
    assert "_compiled_tool_patterns" not in dumped
    json_str = te.model_dump_json()
    assert "re.compile" not in json_str
    assert "Pattern" not in json_str
    assert "_compiled" not in json_str


def test_re_parsed_instance_recompiles_cache() -> None:
    """A round-tripped instance has live, populated caches - not empty defaults."""
    original = TurnExpectation(
        reply_pattern=[r"^ORDER\|"],
        tool_result_pattern={"echo": r"^OK$"},
    )
    re_parsed = TurnExpectation.model_validate_json(original.model_dump_json())
    assert len(re_parsed._compiled_reply_patterns) == 1
    assert isinstance(re_parsed._compiled_reply_patterns[0], re.Pattern)
    assert re_parsed._compiled_reply_patterns[0].search("ORDER|42") is not None
    assert "echo" in re_parsed._compiled_tool_patterns
    assert re_parsed._compiled_tool_patterns["echo"].search("OK") is not None


def test_default_construction_has_empty_caches() -> None:
    te = TurnExpectation()
    assert te._compiled_reply_patterns == []
    assert te._compiled_tool_patterns == {}
