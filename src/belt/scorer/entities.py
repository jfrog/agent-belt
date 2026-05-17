# (c) JFrog Ltd. (2026)

"""Scorer-only entities: judge verdicts and configs.

The on-disk shape of ``ScenarioScore.scores[name]`` lives in
``belt.scorer.payloads`` (the canonical contract). This module
keeps only the scorer-internal types - the *adversarial input contract*
the LLM is asked to produce (:class:`JudgeVerdict`) and the *backend
configuration* knobs (:class:`JudgeConfig`). Keeping them separate from
:class:`belt.scorer.payloads.LLMPayload` lets the on-disk schema
evolve independently of the LLM-facing persona.

Cross-phase entities (:class:`belt.entities.ScenarioScore`) live
in ``belt.entities`` because they are consumed by the aggregator.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from belt.scorer.payloads import ScorerPayload


class ScoreLevel(str, Enum):
    """Verdict tokens the LLM judge may return for a single dimension.

    Two scales coexist so authors can pick the one that matches the
    rubric: ``LOW``/``MEDIUM``/``HIGH`` for graded subjective questions,
    ``PASS``/``FAIL`` for binary checks. ``INCONCLUSIVE`` is orthogonal
    to the scale and indicates the judge could not produce a verdict
    from the available evidence; it counts as a failure in the headline
    but is reported separately so reviewers can distinguish "agent did
    it wrong" from "evidence missing".

    Per-dimension validity is enforced by the JSON schema rendered in
    :func:`belt.agent.scoring.ScoringStrategy.build_schema`; this enum
    is the union of every value any dimension may emit.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


TERNARY_LEVELS: tuple[ScoreLevel, ...] = (ScoreLevel.LOW, ScoreLevel.MEDIUM, ScoreLevel.HIGH)
"""Verdicts a ``kind="ternary"`` dimension may emit (excluding inconclusive)."""

BINARY_LEVELS: tuple[ScoreLevel, ...] = (ScoreLevel.PASS, ScoreLevel.FAIL)
"""Verdicts a ``kind="binary"`` dimension may emit (excluding inconclusive)."""

ALL_VERDICT_TOKENS: tuple[str, ...] = tuple(level.value for level in ScoreLevel)
"""Every verdict token any dimension may emit, in enum-declaration order.

Used as histogram bucket keys and as the ``--llm-fail-on`` allowed-token
set so the CLI validator, the aggregator, and the JSON schema all read
the same list (Design Principle 9)."""

DEFAULT_LLM_FAIL_ON: tuple[str, ...] = ("low", "fail", "inconclusive")
"""Ordered default for ``--llm-fail-on``: every verdict below pass on
either scale plus ``inconclusive``. Order is preserved so the CLI help
text, the argparse default string, and the threshold gating set never
drift from one another."""

DEFAULT_LLM_FAIL_ON_STR: str = ",".join(DEFAULT_LLM_FAIL_ON)
"""Comma-joined form of :data:`DEFAULT_LLM_FAIL_ON` for argparse
defaults and equality checks in ``commands/eval.py``."""

DEFAULT_FAIL_LEVELS: frozenset[str] = frozenset(DEFAULT_LLM_FAIL_ON)
"""Verdict tokens that count as failure in the default headline pass-rate.

``inconclusive`` is included so the judge has no incentive to hedge -
hedging costs the agent a pass, exactly like a real failing verdict
would. Override with ``--llm-fail-on`` if you want to treat
inconclusive as informational only."""

DOWNGRADE_VERDICTS: tuple[str, ...] = (*DEFAULT_LLM_FAIL_ON, "medium")
"""Verdicts that warrant surfacing the judge's reasoning in failure
detail blocks. Extends :data:`DEFAULT_LLM_FAIL_ON` with ``medium`` so
ternary near-misses still appear in the explanation pane even though
they don't fail the headline."""

DOWNGRADE_VERDICT_SET: frozenset[str] = frozenset(DOWNGRADE_VERDICTS)
"""Set form of :data:`DOWNGRADE_VERDICTS` for membership checks in
renderers and exporters."""

JUDGE_ERROR_TYPES: tuple[str, ...] = ("rate_limited", "timeout", "auth_failed", "other")
"""Classification tokens for transient LLM-judge infrastructure failures.

Mirrors the agent-side :data:`belt.entities.ENVIRONMENTAL_ERROR_TYPES`
taxonomy but lives here because it is recorded on the LLM payload, not
on :class:`belt.entities.TurnOutput`. Backends map their provider-specific
exception shapes onto these tokens via
:meth:`belt.scorer.llm.backend.BaseJudgeBackend.classify_error`.

* ``rate_limited`` - HTTP 429 after retries exhausted, ``quota exceeded``.
* ``timeout`` - request hit the configured client timeout (default 120s).
* ``auth_failed`` - HTTP 401/403/404 (recorded only for surfaced
  scenarios; this token is also raised as a fatal :class:`ScorerError`
  to abort the run, but post-mortem readers see the same token).
* ``other`` - any other ``httpx`` error or unexpected exception.
"""

JUDGE_ERROR_TYPE_SET: frozenset[str] = frozenset(JUDGE_ERROR_TYPES)
"""Set form of :data:`JUDGE_ERROR_TYPES` for membership checks."""


class DimensionScore(BaseModel):
    reasoning: str
    score: ScoreLevel


class JudgeVerdict(BaseModel):
    """Dynamic LLM judge verdict - dimensions vary by agent.

    Fixed field: ``overall_pass``.  All other top-level keys are
    ``DimensionScore`` dicts, accepted via ``extra="allow"``. This is
    the *prompt-output contract* (what the LLM is told to produce);
    the on-disk shape is :class:`belt.scorer.payloads.LLMPayload`,
    constructed from a ``JudgeVerdict`` by the LLM scorer.
    """

    model_config = {"extra": "allow"}

    overall_pass: bool

    @property
    def dimension_scores(self) -> dict[str, DimensionScore]:
        """Return only the DimensionScore entries (excludes overall_pass).

        Built from ``model_extra`` because ``JudgeVerdict`` declares
        ``extra="allow"`` and dimension keys vary per agent, so they all
        land in the extras bucket rather than as named fields.
        """
        scores: dict[str, DimensionScore] = {}
        for key, val in (self.model_extra or {}).items():
            if isinstance(val, dict) and "score" in val:
                scores[key] = DimensionScore(**val)
        return scores


class JudgeConfig(BaseModel):
    """LLM judge configuration."""

    model: str = Field(
        ...,
        description=(
            "Model spec. **Required** - no default. Prefix routes to provider: "
            "``openai/...``, ``azure/...``, ``anthropic/...``, ``ollama/...``. A bare "
            "model name (no prefix) requires ``BELT_LLM_PROVIDER`` or an explicit "
            "``provider=`` to resolve. There is no built-in default: silently routing "
            "Azure/Anthropic/Ollama users to OpenAI hides genuine misconfiguration "
            "(401s, cost-surprise, plaintext-credential exfil via custom base URLs). "
            "Set ``--scorer-arg model=...``, ``BELT_LLM_MODEL``, or "
            "``llm.model`` in ``belt.yaml``."
        ),
    )
    temperature: float = Field(default=0.0, ge=0, le=1)
    seed: int = Field(default=2008, description="Sampling seed (OpenAI/Azure only; ignored by Anthropic)")
    max_tokens: int = Field(default=4096)
    max_prompt_chars: int = Field(
        default=100_000,
        description=(
            "Character budget for the dynamic message (user content). "
            "Sections are truncated in priority order (CLI output first, scenario JSON last) "
            "when the rendered message exceeds this limit. ~4 chars ≈ 1 token."
        ),
    )
    provider: Optional[str] = Field(default=None, description="LLM provider override (openai, azure, anthropic)")
    cost_per_prompt_token: Optional[float] = Field(
        default=None, description="Override: USD per prompt token (e.g. 0.000002 for $2/1M)"
    )
    cost_per_completion_token: Optional[float] = Field(
        default=None, description="Override: USD per completion token (e.g. 0.000008 for $8/1M)"
    )


class ScorerResult(BaseModel):
    """Output from any scorer - keyed by scorer name in ``ScenarioScore.scores``.

    ``data`` is a typed payload registered via
    :func:`belt.scorer.payloads.register_payload_type`. The
    registry-backed validator on :attr:`ScenarioScore.scores`
    dispatches on ``schema_version`` so consumers reach into typed
    attributes without re-validating.
    """

    passed: bool
    data: ScorerPayload
