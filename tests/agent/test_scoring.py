# (c) JFrog Ltd. (2026)

"""Tests for agent.scoring - ScoringStrategy, schema generation, and built-in strategies."""

from __future__ import annotations

from belt.agent.scoring import DimensionDef, ScoringStrategy, default_scoring_strategy


def _simple_strategy() -> ScoringStrategy:
    return ScoringStrategy(
        dimensions=[
            DimensionDef(
                name="quality",
                description="Was it good?",
                high="excellent",
                medium="ok",
                low="bad",
            ),
            DimensionDef(
                name="speed",
                description="Was it fast?",
                high="blazing",
                medium="acceptable",
                low="slow",
                evidence_hints="response time, latency",
            ),
        ],
        agent_context="Test agent context.",
    )


class TestScoringStrategy:
    def test_dimension_names(self) -> None:
        s = _simple_strategy()
        assert s.dimension_names == ["quality", "speed"]

    def test_build_schema_structure(self) -> None:
        schema = _simple_strategy().build_schema()
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert "quality" in schema["properties"]
        assert "speed" in schema["properties"]
        assert "overall_pass" in schema["properties"]
        assert set(schema["required"]) == {"quality", "speed", "overall_pass"}

    def test_build_schema_has_per_dimension_verdict_defs(self) -> None:
        schema = _simple_strategy().build_schema()
        # Each dimension owns its own verdict def so the verdict enum
        # is constrained to that dimension's allowed values - a binary
        # dimension could not return "medium" even if the model tries.
        assert "quality_Verdict" in schema["$defs"]
        assert "speed_Verdict" in schema["$defs"]
        assert schema["$defs"]["quality_Verdict"]["properties"]["score"]["enum"] == [
            "low",
            "medium",
            "high",
        ]

    def test_build_schema_strict_compliant(self) -> None:
        """Schema must satisfy OpenAI strict:true requirements."""
        schema = _simple_strategy().build_schema()
        assert schema["additionalProperties"] is False
        for def_name, def_body in schema["$defs"].items():
            assert def_body["additionalProperties"] is False, def_name
            assert set(def_body["required"]) == {"reasoning", "score"}, def_name

    def test_build_dimensions_prompt(self) -> None:
        prompt = _simple_strategy().build_dimensions_prompt()
        assert "## Quality" in prompt
        assert "## Speed" in prompt
        assert "1. high - excellent" in prompt
        assert "Look for: response time, latency" in prompt

    def test_no_evidence_hints_omitted(self) -> None:
        prompt = _simple_strategy().build_dimensions_prompt()
        quality_section = prompt.split("## Speed")[0]
        assert "Look for:" not in quality_section


class TestVerdictScales:
    """Coverage for the four ``kind`` x ``allow_inconclusive`` combinations.

    The tests exercise the prompt rendering and the JSON schema since
    those are the two surfaces the LLM judge actually sees - any other
    layer that reads the dimension flows through them.
    """

    @staticmethod
    def _strategy(*dims: DimensionDef) -> ScoringStrategy:
        return ScoringStrategy(dimensions=list(dims), agent_context="ctx")

    def test_ternary_default_is_unchanged(self) -> None:
        d = DimensionDef(name="quality", description="how good")
        assert d.kind == "ternary"
        assert d.allow_inconclusive is False
        assert d.allowed_score_values == ["low", "medium", "high"]

    def test_ternary_with_inconclusive_extends_scale(self) -> None:
        d = DimensionDef(name="quality", description="how good", allow_inconclusive=True)
        assert d.allowed_score_values == ["low", "medium", "high", "inconclusive"]
        prompt = self._strategy(d).build_dimensions_prompt()
        assert "1. high" in prompt
        assert "inconclusive" in prompt

    def test_ternary_inconclusive_uses_position_four_not_three(self) -> None:
        # Pins the rubric numbering for ternary + allow_inconclusive:
        # the inconclusive line is "4. ...", not "3. ...". A hardcoded
        # "3." here collides with "3. low" and produces a malformed
        # rubric that smaller judges parse as ambiguous.
        d = DimensionDef(name="quality", description="how good", allow_inconclusive=True)
        prompt = self._strategy(d).build_dimensions_prompt()
        assert "4. inconclusive" in prompt
        assert "3. inconclusive" not in prompt
        # And the rubric must list each verdict exactly once.
        assert prompt.count("3. low") == 1

    def test_binary_inconclusive_uses_position_three(self) -> None:
        d = DimensionDef(name="safety", description="x", kind="binary", allow_inconclusive=True)
        prompt = self._strategy(d).build_dimensions_prompt()
        assert "3. inconclusive" in prompt
        assert "4. inconclusive" not in prompt

    def test_binary_kind_uses_pass_fail_scale(self) -> None:
        d = DimensionDef(
            name="no_raw_tokens",
            description="agent did not leak tokens",
            kind="binary",
            pass_="no tokens visible",
            fail="any token visible",
        )
        assert d.allowed_score_values == ["pass", "fail"]
        schema = self._strategy(d).build_schema()
        assert schema["$defs"]["no_raw_tokens_Verdict"]["properties"]["score"]["enum"] == [
            "pass",
            "fail",
        ]
        prompt = self._strategy(d).build_dimensions_prompt()
        assert "1. pass - no tokens visible" in prompt
        assert "2. fail - any token visible" in prompt
        assert "high" not in prompt
        assert "medium" not in prompt

    def test_binary_with_inconclusive_extends_scale(self) -> None:
        d = DimensionDef(
            name="no_raw_tokens",
            description="agent did not leak tokens",
            kind="binary",
            allow_inconclusive=True,
        )
        assert d.allowed_score_values == ["pass", "fail", "inconclusive"]
        schema = self._strategy(d).build_schema()
        assert schema["$defs"]["no_raw_tokens_Verdict"]["properties"]["score"]["enum"] == [
            "pass",
            "fail",
            "inconclusive",
        ]

    def test_invalid_kind_raises_at_construction(self) -> None:
        import pytest

        from belt.errors import ConfigError

        # ``ConfigError`` (a ``BeltError`` subtype) is used so the
        # top-level CLI boundary in ``cli.py`` renders this as a clean
        # one-liner rather than a raw Python traceback (Principle 7).
        with pytest.raises(ConfigError, match="kind"):
            DimensionDef(name="bad", description="x", kind="quaternary")  # type: ignore[arg-type]

    def test_from_config_dict_accepts_pass_alias(self) -> None:
        d = DimensionDef.from_config_dict(
            {
                "name": "no_raw_tokens",
                "description": "binary check",
                "kind": "binary",
                "pass": "no token",
                "fail": "any token",
            }
        )
        assert d.kind == "binary"
        assert d.pass_ == "no token"
        assert d.fail == "any token"

    def test_from_config_dict_rejects_unknown_keys(self) -> None:
        import pytest

        from belt.errors import ConfigError

        with pytest.raises(ConfigError, match="unknown field"):
            DimensionDef.from_config_dict({"name": "x", "description": "x", "stale_typo_field": "oops"})

    def test_mixed_strategy_renders_distinct_per_dimension_schemas(self) -> None:
        # The schema must keep each dimension's verdict enum independent
        # so a binary dimension does not silently accept ternary verdicts.
        strat = self._strategy(
            DimensionDef(name="quality", description="how good"),
            DimensionDef(name="quality_check", description="check", kind="binary"),
            DimensionDef(
                name="quality_with_unknown",
                description="check",
                allow_inconclusive=True,
            ),
        )
        schema = strat.build_schema()
        assert schema["$defs"]["quality_Verdict"]["properties"]["score"]["enum"] == [
            "low",
            "medium",
            "high",
        ]
        assert schema["$defs"]["quality_check_Verdict"]["properties"]["score"]["enum"] == [
            "pass",
            "fail",
        ]
        assert schema["$defs"]["quality_with_unknown_Verdict"]["properties"]["score"]["enum"] == [
            "low",
            "medium",
            "high",
            "inconclusive",
        ]


class TestBuiltInStrategies:
    def test_default_has_4_dimensions(self) -> None:
        s = default_scoring_strategy()
        assert len(s.dimensions) == 4
        assert "execution" in s.dimension_names
        assert "review" not in s.dimension_names

    def test_default_has_generic_context(self) -> None:
        s = default_scoring_strategy()
        assert "CLI-based AI agent" in s.agent_context
