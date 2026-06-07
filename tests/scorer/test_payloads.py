# (c) JFrog Ltd. (2026)

"""Tests for the scorer payload contract.

Targets the cross-cutting guarantees of ``belt.scorer.payloads``:

1. Built-in payloads round-trip through JSON without losing
   ``schema_version`` (the typed-dispatch contract).
2. ``ScenarioScore`` validation hard-fails on missing or unregistered
   ``schema_version`` - no silent fallback to a "guess" walker.
3. ``register_payload_type`` reads the version off the payload class
   itself, so the only literal version string in the codebase is the
   ``Literal[...]`` default declared by each payload.
4. ``iter_dimension_feedback`` dispatches to the registered iterator
   for built-ins and plugin-registered shapes, and raises on
   unregistered shapes.

These guarantees are what makes ``ScenarioScore.scores`` a typed
contract instead of an ad-hoc dict; if any of them regress, downstream
exporters start producing fabricated rows.
"""

from __future__ import annotations

from typing import Iterator, Literal

import pytest
from pydantic import BaseModel

from belt.entities import ScenarioScore
from belt.scorer.payloads import (
    _PAYLOAD_REGISTRY,
    CheckEntry,
    DimensionFeedback,
    LLMDimensionVerdict,
    LLMPayload,
    RulesPayload,
    _payload_version,
    iter_dimension_feedback,
    level_to_score,
    register_payload_type,
    registered_payload_types,
)


def _passing_score() -> ScenarioScore:
    return ScenarioScore(
        scenario_name="s",
        group="g",
        overall_pass=True,
        scores={
            "rules": RulesPayload(
                passed=True,
                checks=[
                    CheckEntry(dimension="execution", check="no_errors", passed=True),
                    CheckEntry(dimension="response", check="has_reply", passed=True),
                ],
            ),
            "llm": LLMPayload(
                overall_pass=True,
                dimensions={
                    "correctness": LLMDimensionVerdict(score="high", reasoning="solid"),
                },
            ),
        },
    )


class TestRoundTrip:
    def test_builtin_payloads_round_trip_through_json(self) -> None:
        original = _passing_score()
        rebuilt = ScenarioScore.model_validate_json(original.model_dump_json())

        assert isinstance(rebuilt.scores["rules"], RulesPayload)
        assert isinstance(rebuilt.scores["llm"], LLMPayload)
        assert rebuilt.scores["rules"].schema_version == "rules.v1"
        assert rebuilt.scores["llm"].schema_version == "llm.v1"

    def test_serialised_json_carries_schema_version_for_each_scorer(self) -> None:
        # ``SerializeAsAny`` on the value type is what makes the
        # concrete payload's ``schema_version`` survive serialisation;
        # if someone accidentally drops it, downstream readers cannot
        # re-dispatch and the contract collapses.
        payload = _passing_score().model_dump(mode="json")
        assert payload["scores"]["rules"]["schema_version"] == "rules.v1"
        assert payload["scores"]["llm"]["schema_version"] == "llm.v1"


class TestHardFailOnUnknownSchema:
    def test_missing_schema_version_is_rejected_at_parse_time(self) -> None:
        with pytest.raises(Exception, match="missing required 'schema_version'"):
            ScenarioScore(
                scenario_name="s",
                group="g",
                overall_pass=False,
                scores={"rules": {"checks": []}},
            )

    def test_unregistered_schema_version_is_rejected_at_parse_time(self) -> None:
        with pytest.raises(Exception, match="schema_version 'rules.v999' is not registered"):
            ScenarioScore(
                scenario_name="s",
                group="g",
                overall_pass=False,
                scores={"rules": {"schema_version": "rules.v999", "checks": []}},
            )

    def test_iterator_rejects_unregistered_payload(self) -> None:
        # Construct a score with a payload whose ``schema_version`` is
        # not registered; iteration must raise rather than fall back to
        # a generic walker that fabricates dimensions.
        score = ScenarioScore(scenario_name="s", group="g", overall_pass=True)
        # Bypass validation to seed an unregistered shape so we test the
        # iterator's own defence (validators are exercised separately).
        score.__dict__["scores"] = {"unknown": {"schema_version": "unknown.v1"}}
        with pytest.raises(ValueError, match="not registered"):
            iter_dimension_feedback(score)


class TestSingleSourceOfTruth:
    def test_payload_version_reads_off_field_default(self) -> None:
        assert _payload_version(RulesPayload) == "rules.v1"
        assert _payload_version(LLMPayload) == "llm.v1"

    def test_registry_keys_match_class_defaults(self) -> None:
        # The registry key for each built-in is the class's own
        # ``schema_version`` default. This is the invariant that lets
        # the codebase declare each version string exactly once (in the
        # ``Literal[...] = "..."`` line on the payload class).
        assert _PAYLOAD_REGISTRY["rules.v1"][0] is RulesPayload
        assert _PAYLOAD_REGISTRY["llm.v1"][0] is LLMPayload

    def test_registered_payload_types_lists_builtins(self) -> None:
        assert "rules.v1" in registered_payload_types()
        assert "llm.v1" in registered_payload_types()


class TestPluginRegistration:
    def test_plugin_payload_round_trips_and_iterates(self) -> None:
        class _PluginPayload(BaseModel):
            schema_version: Literal["plugin.v1"] = "plugin.v1"
            label: str
            score_value: float

        def _iter(scorer_name: str, payload: BaseModel) -> Iterator[DimensionFeedback]:
            assert isinstance(payload, _PluginPayload)
            yield DimensionFeedback(
                scorer_name=scorer_name,
                dimension=payload.label,
                score=payload.score_value,
                comment="plugin-iter",
                raw=payload.model_dump(mode="json"),
            )

        register_payload_type(_PluginPayload, _iter)
        try:
            score = ScenarioScore(
                scenario_name="s",
                group="g",
                overall_pass=True,
                scores={"plugin": _PluginPayload(label="quality", score_value=0.9)},
            )

            rebuilt = ScenarioScore.model_validate_json(score.model_dump_json())
            assert isinstance(rebuilt.scores["plugin"], _PluginPayload)

            feedback = iter_dimension_feedback(rebuilt)
            assert any(f.scorer_name == "plugin" and f.dimension == "quality" for f in feedback)
        finally:
            _PAYLOAD_REGISTRY.pop("plugin.v1", None)


class TestLevelToScore:
    @pytest.mark.parametrize(
        "level, expected",
        [
            ("high", 1.0),
            ("medium", 0.5),
            ("low", 0.0),
            ("pass", 1.0),
            ("fail", 0.0),
            # ``inconclusive`` deliberately maps to ``None`` so consumers
            # can distinguish "no numeric verdict" from "scored low".
            ("inconclusive", None),
            ("bogus", None),
            ("", None),
        ],
    )
    def test_level_mapping(self, level: str, expected: float | None) -> None:
        assert level_to_score(level) == expected
