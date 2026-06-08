# (c) JFrog Ltd. (2026)

"""Pydantic schema for ``--scorer-config`` YAML files.

A scorer-config YAML lets users declare multiple LLM judges (optionally
voting via :class:`belt.scorer.llm.consensus.ConsensusScorer`) and the
per-judge knobs that drive each:

.. code-block:: yaml

    judges:
      sonnet:
        model: anthropic/claude-sonnet-4-5
        resolution: turn
        evidence_scope: cumulative
        dimensions:
          - name: correctness
            kind: binary
        max_retries: 5
      gpt:
        model: openai/gpt-5.4-mini
        resolution: turn
    consensus: majority

Promoted from ad-hoc ``dict`` parsing in
:func:`belt.scorer.pipeline.load_scorer_config` to a typed Pydantic
model so:

* Typos in field names (``resolutoin``, ``evdience_scope``) are
  rejected with a precise error instead of silently ignored.
* Caps (``dimensions <= 50``, name pattern) ride on the model and
  cannot drift from the runtime checks.
* Plugins consuming the parsed shape get a typed contract
  (:class:`ScorerConfigFile`, :class:`JudgeDef`) rather than untyped
  ``dict[str, Any]``.

The literal regex below is intentionally inlined (not imported from
``belt.scenario``): the constant there is module-private. If a future
refactor promotes a single name-policy module to ``belt._safe.py``, the
import update is a one-line diff.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

from belt.scorer.entities import EvidenceScope, Resolution

_SAFE_JUDGE_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,63}$"
"""Conservative ASCII identifier for judge keys.

Mirrors ``_SAFE_NAME_PATTERN`` in :mod:`belt.scenario` for scenario
names: judge names appear in CLI panels, GitHub markdown summaries,
JUnit testcase attributes, and result.json keys, so the same
markup-injection envelope applies."""

_RESERVED_JUDGE_NAMES: frozenset[str] = frozenset({"rules", "llm"})
"""Names a per-judge entry may NOT take.

``rules`` clashes with the built-in rule-based scorer's key in
``ScenarioScore.scores``. ``llm`` is the conventional key for the
single-judge / consensus scorer. A user-declared judge taking
either of these names would silently overwrite the built-in
contribution; the model-validator below rejects it loudly."""


class JudgeDef(BaseModel):
    """One judge entry inside ``judges:`` of a scorer-config YAML."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=_SAFE_JUDGE_NAME_PATTERN, max_length=64)
    model: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0, le=1)
    seed: Optional[int] = None
    max_tokens: Optional[int] = Field(default=None, ge=1)
    # Bounded: a runaway ``max_retries=1_000_000`` against a 429-prone
    # backend would amplify cost without bound. Caps the per-call retry
    # budget at 20 which exceeds the SOTA exponential-backoff default
    # by 4x.
    max_retries: int = Field(default=5, ge=0, le=20)
    # Mirrors :attr:`belt.scorer.entities.JudgeConfig.max_prompt_chars`
    # min/max bounds so a configured value rounds-trips through this
    # schema without surprises.
    max_prompt_chars: Optional[int] = Field(default=None, ge=1000, le=2_000_000)
    cost_per_prompt_token: Optional[float] = Field(default=None, ge=0)
    cost_per_completion_token: Optional[float] = Field(default=None, ge=0)
    # Parsed via :func:`belt.scorer.scenario_map.parse_dimension_defs`
    # which accepts both string shorthand and dict shape, hence
    # ``list[Any]`` rather than a stricter union.
    dimensions: Optional[list[Any]] = Field(default=None, max_length=50)
    system_preamble: Optional[str] = Field(default=None, max_length=10_000)
    # When True, declared ``dimensions`` extend the framework's default
    # generic dimensions rather than replacing them.
    extend_defaults: bool = False
    resolution: Resolution = "scenario"
    evidence_scope: EvidenceScope = "isolated"

    @model_validator(mode="after")
    def _check_reserved(self) -> "JudgeDef":
        if self.name in _RESERVED_JUDGE_NAMES:
            raise ValueError(
                f"judge name {self.name!r} is reserved (clashes with the built-in "
                f"{self.name!r} scorer key in ``ScenarioScore.scores``). "
                "Rename the judge."
            )
        return self


class _JudgesMap(RootModel[dict[str, dict[str, Any]]]):
    """Round-trip type for the inner ``judges:`` mapping.

    Keeps the YAML round-trip declarative while letting the outer
    :class:`ScorerConfigFile` flatten the mapping into ``list[JudgeDef]``
    with the YAML key threaded through as ``JudgeDef.name``.
    """


class ScorerConfigFile(BaseModel):
    """Top-level shape of a ``--scorer-config`` YAML file."""

    model_config = ConfigDict(extra="forbid")

    judges: dict[str, dict[str, Any]]
    consensus: Optional[Literal["majority", "unanimous", "any"]] = None

    def to_judge_defs(self) -> list[JudgeDef]:
        """Flatten ``{name: cfg}`` into ``[JudgeDef(name=name, **cfg)]``.

        Performed lazily (not in ``__init__``) so ``ScorerConfigFile``
        instances stay cheap to construct in tests and round-trip
        through ``model_validate(model_dump())`` cleanly.
        """
        if not self.judges:
            raise ValueError("scorer-config file must declare at least one judge under ``judges:``")
        out: list[JudgeDef] = []
        for raw_name, cfg in self.judges.items():
            if not isinstance(cfg, dict):
                raise ValueError(f"judge entry {raw_name!r} must be a mapping, got {type(cfg).__name__}")
            cfg_name = cfg.get("name")
            if cfg_name is not None and cfg_name != raw_name:
                raise ValueError(
                    f"judge entry {raw_name!r} declares conflicting inner ``name: {cfg_name!r}``; "
                    "the YAML key is the canonical name - drop the inner ``name`` field."
                )
            out.append(JudgeDef(name=raw_name, **{k: v for k, v in cfg.items() if k != "name"}))
        return out


__all__ = ["JudgeDef", "ScorerConfigFile", "_SAFE_JUDGE_NAME_PATTERN", "_RESERVED_JUDGE_NAMES"]
