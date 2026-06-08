# (c) JFrog Ltd. (2026)

"""Canonical contract for scorer output: typed payloads + uniform iteration.

This module is the single source of truth for "what does a scorer write
into ``ScenarioScore.scores[name]``, and how do consumers read it
uniformly".

Two complementary surfaces live here, intentionally:

* **Typed payloads** (``RulesPayload``, ``LLMPayload``,
  ``LLMDimensionVerdict``, ``CheckEntry``) - Pydantic models that
  describe each scorer's on-disk shape with a ``schema_version``
  discriminator. ``ScenarioScore.scores`` is typed as
  ``dict[str, ScorerPayload]`` so consumers reach into typed fields
  (``payload.checks``, ``payload.dimensions``, ``payload.usage``)
  instead of stringly-typed dict lookups.
* **Uniform iteration** (``DimensionFeedback`` +
  ``iter_dimension_feedback``) - one function any consumer (built-in
  exporter, aggregator stat builder, plugin) calls to get one normalised
  ``DimensionFeedback`` per ``(scorer, dimension)``, regardless of which
  scorer produced the payload.

Third-party scorers participate via :func:`register_payload_type`: they
declare their ``schema_version`` plus a typed payload class plus the
iterator that converts their payload to ``DimensionFeedback`` rows. Any
already-shipped consumer immediately handles them without modification.

Why one module instead of splitting types and iteration:
  the payload class and its per-dimension iterator are the same concern
  ("the contract for this scorer's output"). Splitting them creates a
  seam that invites drift between the type and the walker - exactly the
  drift this design exists to remove. Mirrors the in-repo pattern in
  ``_safe.py`` (untrusted-input handling) and ``_public_api.py`` (public
  surface): one module = one concern = no place to drift.

Failure model: hard-fail, never guess. Missing or unregistered
``schema_version`` raises ``ValueError`` at parse time. There is no
silent fallback - a wrong schema written to disk surfaces as a loud
exception, not as fabricated dimensions.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator, Literal, Optional

from pydantic import BaseModel, Field, SerializeAsAny

# -----------------------------------------------------------------------------
# Typed payloads
# -----------------------------------------------------------------------------


class CheckEntry(BaseModel):
    """One deterministic check produced by the rule-based scorer.

    ``passed`` is tri-state: ``True`` (pass), ``False`` (fail),
    ``None`` (skipped - e.g. preflight aborted before this check ran).
    Treating skipped as ``None`` rather than ``False`` lets consumers
    distinguish "scenario did not run to completion" from "scenario ran
    and check failed".
    """

    dimension: str
    check: str
    passed: Optional[bool]
    details: str = ""
    turn_idx: Optional[int] = None


class RulesPayload(BaseModel):
    """On-disk shape of the rule-based scorer's contribution to ``scores``.

    Dimensions are not first-class here - they are tagged on each
    individual :class:`CheckEntry` via :attr:`CheckEntry.dimension`,
    which keeps the rules scorer's per-check granularity intact for
    consumers that need it (``stats.py``, ``compute_partial_score``).
    Per-dimension aggregation for cross-scorer iteration is synthesised
    by :func:`iter_dimension_feedback`.
    """

    schema_version: Literal["rules.v1"] = "rules.v1"
    checks: list[CheckEntry] = Field(default_factory=list)
    passed: bool = True
    has_error: Optional[bool] = None

    model_config = {"extra": "allow"}


class LLMDimensionVerdict(BaseModel):
    """One LLM-judged dimension (verdict + reasoning).

    ``score`` is the categorical level the judge assigned. The literal
    union covers every verdict any per-dimension scale may emit; per-
    dimension validity is enforced by the JSON schema the strategy
    renders for the LLM, not by this model. Consumers that want a
    numeric score should call :func:`level_to_score` rather than
    reinventing the mapping.
    """

    score: Literal["high", "medium", "low", "pass", "fail", "inconclusive"]
    reasoning: str = ""


class UsageStats(BaseModel):
    """Token accounting for an LLM-backed scorer call.

    Fields are optional so backends that don't report a particular
    counter (e.g. Anthropic's cache-read tokens) can omit it without
    forcing zeros into the on-disk JSON. Aggregator code accepts both
    OpenAI (``prompt_tokens``) and Anthropic (``input_tokens``) names
    via the canonical fields below; backends translate at the edge.
    """

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    model_config = {"extra": "allow"}


class ConsensusMeta(BaseModel):
    """Provenance for a consensus-merged LLM payload.

    Populated only when more than one judge voted (multi-judge
    scoring). Single-judge runs leave this as ``None`` so the on-disk
    JSON stays small.
    """

    strategy: str
    judges: list[str]
    shared_dimensions: list[str] = Field(default_factory=list)
    disagreements: list[Any] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class LLMPayload(BaseModel):
    """On-disk shape of an LLM-judge scorer's contribution to ``scores``.

    The producer-side ``JudgeVerdict`` (in ``scorer/entities.py``) is
    the *adversarial input* contract the LLM is asked to satisfy;
    ``LLMPayload`` is the *canonical on-disk* contract, intentionally
    distinct so structural changes to the on-disk shape don't force
    every LLM judge persona to be rewritten in lockstep.

    ``judge_errored`` / ``judge_error_type`` mark a non-verdict payload:
    the judge backend failed for infrastructure reasons (rate-limit,
    timeout, network) and produced no real verdict. The fields are set
    at the :class:`belt.scorer.llm.backend.BaseJudgeBackend` boundary so
    every backend classifies the same exception shapes into the same
    tokens. A verdict-less payload always has ``overall_pass=False`` and
    ``dimensions={}``; the aggregator's task-quality split partitions
    these scenarios into a "judge environment failures" axis so they do
    not contaminate the headline pass-rate either as false greens (rules
    passed, judge silently vanished) or false reds (rules failed and
    judge silently vanished, attributed to task quality).
    """

    schema_version: Literal["llm.v1"] = "llm.v1"
    overall_pass: bool
    dimensions: dict[str, LLMDimensionVerdict] = Field(default_factory=dict)
    usage: Optional[UsageStats] = None
    consensus_meta: Optional[ConsensusMeta] = None
    individual_verdicts: Optional[dict[str, Any]] = None
    judge_errored: bool = False
    judge_error_type: Optional[Literal["rate_limited", "timeout", "auth_failed", "other", "all_turns_skipped"]] = None

    model_config = {"extra": "allow"}


class TurnVerdict(BaseModel):
    """One per-turn LLM verdict inside a :class:`PerTurnLLMPayload`.

    Field shape mirrors :class:`LLMPayload` so consumers walking turns
    via the registered iterator see uniform per-dimension rows. The
    ``judge_errored`` flag is per-turn: a single turn may have errored
    (rate-limited, timeout) without invalidating the verdicts produced
    by sibling turns. The payload-level :attr:`PerTurnLLMPayload.judge_errored`
    is the OR of every turn's flag.
    """

    turn_idx: int
    dimensions: dict[str, LLMDimensionVerdict] = Field(default_factory=dict)
    usage: Optional[UsageStats] = None
    judge_errored: bool = False
    judge_error_type: Optional[Literal["rate_limited", "timeout", "auth_failed", "other", "all_turns_skipped"]] = None

    model_config = {"extra": "allow"}


class PerTurnLLMPayload(BaseModel):
    """On-disk shape of a *per-turn* LLM-judge scorer's contribution to ``scores``.

    A judge configured with ``resolution="turn"`` produces one
    :class:`TurnVerdict` per scenario turn (see
    :class:`belt.scorer.llm.scorer.LLMScorer._score_per_turn`).
    Payload-level fields aggregate across turns so callers that only
    care about scenario-level signal (``judge_errored``, ``usage``,
    ``overall_pass``) read uniformly between scenario- and per-turn
    payloads.

    Per-turn iteration via the registered payload iterator
    (:func:`_iter_per_turn_llm`) emits one :class:`DimensionFeedback`
    per ``(scorer, dimension)`` using worst-of-turns rollup so the
    aggregator's existing per-dimension code path keeps working with no
    public API change to :class:`DimensionFeedback`. The per-turn
    detail is preserved verbatim in the iterator row's ``raw["per_turn"]``
    for consumers that need it (per-turn detail in JUnit failure
    bodies, CSV sidecar, markdown drill-down, ``belt view``).

    ``overall_pass`` is the AND across turns AND dimensions, with an
    explicit ``any(t.dimensions ...)`` guard so an all-skipped scenario
    (every turn carries ``skip:true`` for this judge) cannot vacuously
    pass via ``all([])``. The runtime ``all_skipped`` taint rule in
    ``LLMScorer._score_per_turn`` AND the preflight check in
    ``validate_per_turn_judges_against_scenarios`` are belt-and-braces:
    preflight catches the static case, the taint rule catches the
    dynamic edge case (e.g. mixed skip-and-error across turns).
    """

    schema_version: Literal["per_turn_llm.v1"] = "per_turn_llm.v1"
    overall_pass: bool
    turns: list[TurnVerdict] = Field(default_factory=list)
    usage: Optional[UsageStats] = None
    consensus_meta: Optional[ConsensusMeta] = None
    individual_verdicts: Optional[dict[str, Any]] = None
    judge_errored: bool = False
    judge_error_type: Optional[Literal["rate_limited", "timeout", "auth_failed", "other", "all_turns_skipped"]] = None

    model_config = {"extra": "allow"}


# -----------------------------------------------------------------------------
# Registry: schema_version -> (payload class, iterator)
# -----------------------------------------------------------------------------


class DimensionFeedback(BaseModel):
    """One row's worth of "scorer X scored dimension Y" feedback.

    Yielded by :func:`iter_dimension_feedback` regardless of the
    underlying payload shape. Consumers (markdown / csv / junit
    exporters, terminal renderer) iterate this stream instead of
    forking on scorer name + payload shape.

    ``score`` is normalised to the ``0.0 - 1.0`` range or ``None``
    when no numeric score is available (e.g. a rules dimension whose
    only checks were skipped). ``raw`` carries the source sub-payload
    for consumers that need detail beyond the summary.
    """

    scorer_name: str
    dimension: str
    score: Optional[float]
    comment: str = ""
    raw: dict[str, Any] = Field(default_factory=dict)


PayloadIterator = Callable[[str, BaseModel], Iterable[DimensionFeedback]]
"""Signature plugins implement to participate in :func:`iter_dimension_feedback`.

Receives the scorer name (the key under which the payload is stored
in ``ScenarioScore.scores``) and the validated payload instance.
Yields one :class:`DimensionFeedback` per scored dimension."""


_PAYLOAD_REGISTRY: dict[str, tuple[type[BaseModel], PayloadIterator]] = {}


def _payload_version(payload_cls: type[BaseModel]) -> str:
    """Read ``schema_version`` off a payload class via its field default.

    Single source of truth: the only place a literal version string
    lives is the ``Literal[...] = "..."`` declaration on the payload
    class itself. Everywhere else (registration, discrimination,
    public listing) reads back through this helper so renaming a
    version touches exactly one line of code.
    """
    field = payload_cls.model_fields.get("schema_version")
    if field is None or field.default is None:
        raise TypeError(
            f"{payload_cls.__name__} is not a valid payload class: it must declare "
            "``schema_version: Literal['<name>.v<int>'] = '<name>.v<int>'``"
        )
    return str(field.default)


def register_payload_type(
    payload_cls: type[BaseModel],
    iterator: PayloadIterator,
) -> None:
    """Register a scorer payload class + its per-dimension iterator.

    The ``schema_version`` is read from ``payload_cls`` so the
    registry, the on-disk discriminator, and the
    ``Literal[...]`` declaration cannot drift apart. After
    registration, ``iter_dimension_feedback`` validates incoming dicts
    bearing this ``schema_version`` against ``payload_cls`` and
    dispatches to ``iterator`` for per-dimension rows. Plugins
    typically register at module import time so the contract is in
    place before any reader walks a ``ScenarioScore``.

    Re-registering the same ``schema_version`` overwrites the prior
    entry; tests and plugins occasionally rely on this to swap an
    iterator in-place.
    """
    version = _payload_version(payload_cls)
    _PAYLOAD_REGISTRY[version] = (payload_cls, iterator)


def registered_payload_types() -> list[str]:
    """Return the sorted list of registered ``schema_version`` values.

    Built-ins (``rules.v1``, ``llm.v1``) are always present once
    :mod:`belt.scorer.payloads` is imported; any additional
    entries come from third-party :func:`register_payload_type` calls.
    """
    return sorted(_PAYLOAD_REGISTRY.keys())


# -----------------------------------------------------------------------------
# Validator + type alias for ScenarioScore.scores
# -----------------------------------------------------------------------------


def _validate_scores_dict(value: Any) -> Any:
    """Validate the ``ScenarioScore.scores`` map against the registry.

    Each entry must carry a ``schema_version`` whose value is
    registered via :func:`register_payload_type`. Missing or unknown
    versions raise a ``ValueError`` so the failure surfaces at parse
    time rather than as a guessed dimension stream further downstream.
    Already-validated payload instances pass through untouched.
    """
    if not isinstance(value, dict):
        return value
    validated: dict[str, BaseModel] = {}
    for scorer_name, payload in value.items():
        if isinstance(payload, BaseModel):
            validated[scorer_name] = payload
            continue
        if not isinstance(payload, dict):
            raise ValueError(f"scores['{scorer_name}']: expected dict or BaseModel, " f"got {type(payload).__name__}")
        version = payload.get("schema_version")
        if version is None:
            raise ValueError(
                f"scores['{scorer_name}']: missing required 'schema_version'. "
                f"Registered: {registered_payload_types()}"
            )
        entry = _PAYLOAD_REGISTRY.get(version)
        if entry is None:
            raise ValueError(
                f"scores['{scorer_name}']: schema_version {version!r} is not "
                f"registered. Plugins must call belt.register_payload_type "
                f"before reading score.json. Registered: {registered_payload_types()}"
            )
        payload_cls, _ = entry
        validated[scorer_name] = payload_cls.model_validate(payload)
    return validated


ScorerPayload = SerializeAsAny[BaseModel]
"""Typed value of one entry in ``ScenarioScore.scores``.

``SerializeAsAny`` instructs Pydantic to use the *runtime* class's
serialization schema rather than the declared :class:`BaseModel`
base, so concrete payload classes
(:class:`RulesPayload`, :class:`LLMPayload`, plugin-registered
subclasses) round-trip every field, including ``schema_version``.
Validation routes through :func:`_validate_scores_dict` (wired into
:class:`belt.entities.ScenarioScore` as a ``field_validator``);
anything not registered via :func:`register_payload_type` is rejected
at parse time."""


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


_LEVEL_TO_SCORE: dict[str, float] = {
    "high": 1.0,
    "medium": 0.5,
    "low": 0.0,
    "pass": 1.0,
    "fail": 0.0,
}


def level_to_score(level: str) -> Optional[float]:
    """Map an LLM judge's categorical level to a 0.0-1.0 numeric score.

    Returns ``None`` for ``inconclusive`` (no numeric verdict was
    produced) and for unknown levels, so callers can distinguish "no
    score" from "scored low". Centralised here so the mapping has one
    source of truth shared by built-in iteration and plugin iterators.
    """
    return _LEVEL_TO_SCORE.get(level)


# -----------------------------------------------------------------------------
# Built-in iterators
# -----------------------------------------------------------------------------


def _iter_rules(scorer_name: str, payload: BaseModel) -> Iterator[DimensionFeedback]:
    """Synthesise per-dimension feedback from a flat ``checks`` list.

    Rules scorers don't emit dimensions at the top level - they emit a
    flat array of :class:`CheckEntry` with ``dimension`` tagged per
    check. Group those by dimension, score as
    ``passed_count / runnable_count`` (skipped checks excluded from
    the denominator), and surface the failing-check details in the
    ``comment`` so the markdown report shows what went wrong without
    expanding ``raw``.
    """
    assert isinstance(payload, RulesPayload)
    by_dim: dict[str, list[CheckEntry]] = {}
    for check in payload.checks:
        by_dim.setdefault(check.dimension, []).append(check)
    for dim_name in sorted(by_dim):
        dim_checks = by_dim[dim_name]
        passed = [c for c in dim_checks if c.passed is True]
        failed = [c for c in dim_checks if c.passed is False]
        skipped = [c for c in dim_checks if c.passed is None]
        runnable = len(passed) + len(failed)
        score = (len(passed) / runnable) if runnable else None
        yield DimensionFeedback(
            scorer_name=scorer_name,
            dimension=dim_name,
            score=score,
            comment=_format_rules_comment(failed, skipped),
            raw={
                "passed": len(passed),
                "failed": len(failed),
                "skipped": len(skipped),
                "total": len(dim_checks),
                "checks": [c.model_dump(mode="json") for c in dim_checks],
            },
        )


def _format_rules_comment(failed: list[CheckEntry], skipped: list[CheckEntry]) -> str:
    """Build the human-readable comment for a synthesised rules row."""
    if not failed and not skipped:
        return ""
    if not failed and skipped:
        return f"{len(skipped)} check(s) not evaluated (scenario aborted before rules ran)"
    lines = []
    for c in failed:
        prefix = f"turn {c.turn_idx}: " if c.turn_idx is not None else ""
        suffix = f" - {c.details}" if c.details else ""
        lines.append(f"{prefix}{c.check}{suffix}")
    return "\n".join(lines)


def _iter_llm(scorer_name: str, payload: BaseModel) -> Iterator[DimensionFeedback]:
    """One :class:`DimensionFeedback` per dimension in :class:`LLMPayload`.

    The numeric ``score`` comes from :func:`level_to_score`;
    ``comment`` is the judge's reasoning for that dimension verbatim
    so consumers don't have to re-derive it from ``raw``.
    """
    assert isinstance(payload, LLMPayload)
    for dim_name in sorted(payload.dimensions):
        verdict = payload.dimensions[dim_name]
        yield DimensionFeedback(
            scorer_name=scorer_name,
            dimension=dim_name,
            score=level_to_score(verdict.score),
            comment=verdict.reasoning,
            raw=verdict.model_dump(mode="json"),
        )


# Worst-of-turns rank: used by :func:`_iter_per_turn_llm` to pick the
# per-dimension cell that represents the scenario in cross-judge / cross-
# threshold rollups. Lower rank = worse verdict; ``inconclusive`` ranks
# below every real verdict so a single inconclusive turn pulls the rollup
# down (mirrors :data:`belt.scorer.llm.consensus._LEVEL_RANK` exactly).
_WORST_OF_TURNS_RANK: dict[str, int] = {
    "inconclusive": -1,
    "fail": 0,
    "low": 0,
    "medium": 1,
    "high": 2,
    "pass": 2,
}


def _iter_per_turn_llm(scorer_name: str, payload: BaseModel) -> Iterator[DimensionFeedback]:
    """One :class:`DimensionFeedback` per dimension across all turns.

    Per-turn judging produces ``T × D`` raw cells (T turns × D
    dimensions). The aggregator's existing per-dimension code path
    expects one row per ``(scorer, dimension)``; we collapse the T-axis
    via *worst-of-turns rollup*:

    - ``score`` = worst-ranked turn's numeric score (via
      :func:`level_to_score`).
    - ``comment`` = the worst turn's reasoning prefixed ``[turn N]`` so
      a reader sees which turn dragged the cell down.
    - ``raw["per_turn"]`` = every :class:`TurnVerdict` for this dimension
      in turn order, so consumers that need the full per-turn breakdown
      (CSV sidecar, JUnit failure body, ``belt view``) can read it
      without re-walking the payload.

    Turns with no entry for a dimension (e.g. ``skip:true`` or a
    judge-errored turn) are excluded from the rollup but still surface
    in ``raw["per_turn"]`` so the consumer can distinguish "no verdict"
    from "low verdict".
    """
    assert isinstance(payload, PerTurnLLMPayload)
    # Collect, per dimension, the list of (turn_idx, LLMDimensionVerdict)
    # pairs for turns that actually voted. Dimensions can shift per turn
    # (per-turn override may declare distinct dimensions), so we take the
    # union across turns.
    by_dim: dict[str, list[tuple[int, LLMDimensionVerdict]]] = {}
    full_per_turn: dict[str, list[dict[str, Any]]] = {}
    for turn in payload.turns:
        for dim_name, verdict in turn.dimensions.items():
            by_dim.setdefault(dim_name, []).append((turn.turn_idx, verdict))
            full_per_turn.setdefault(dim_name, []).append(
                {"turn_idx": turn.turn_idx, **verdict.model_dump(mode="json")}
            )
    for dim_name in sorted(by_dim):
        entries = by_dim[dim_name]
        # Worst-of-turns: pick the entry whose rank is lowest. Ties
        # broken by lowest turn_idx so the rollup is deterministic.
        worst_idx, worst_verdict = min(
            entries,
            key=lambda e: (_WORST_OF_TURNS_RANK.get(e[1].score, -1), e[0]),
        )
        comment = f"[turn {worst_idx}] {worst_verdict.reasoning}" if worst_verdict.reasoning else f"[turn {worst_idx}]"
        yield DimensionFeedback(
            scorer_name=scorer_name,
            dimension=dim_name,
            score=level_to_score(worst_verdict.score),
            comment=comment,
            raw={
                "worst_turn_idx": worst_idx,
                "worst_score": worst_verdict.score,
                "per_turn": sorted(full_per_turn[dim_name], key=lambda e: e["turn_idx"]),
            },
        )


# Built-ins registered eagerly so importing this module is sufficient.
# Version strings are read off the payload classes themselves -
# ``Literal[...]`` is the only place a string literal lives.
register_payload_type(RulesPayload, _iter_rules)
register_payload_type(LLMPayload, _iter_llm)
register_payload_type(PerTurnLLMPayload, _iter_per_turn_llm)


# -----------------------------------------------------------------------------
# Public iterator
# -----------------------------------------------------------------------------


def iter_dimension_feedback(score: Any) -> list[DimensionFeedback]:
    """Yield one :class:`DimensionFeedback` per ``(scorer, dimension)`` in ``score``.

    ``score`` is a :class:`belt.entities.ScenarioScore`. The
    return type is a concrete ``list`` (not a generator) so consumers
    can iterate multiple times - exporters frequently need ``len(...)``
    plus a render pass.

    Order is deterministic: scorer keys in their dict-insertion order
    (matches the order the scorer pipeline writes them in), and inside
    each scorer's payload, dimensions sorted alphabetically.

    Raises ``ValueError`` if a payload's ``schema_version`` is not
    registered. There is no silent fallback: an unrecognised shape is
    a bug to be fixed (in the producer or via :func:`register_payload_type`),
    not a condition to paper over.
    """
    out: list[DimensionFeedback] = []
    for scorer_name, payload in score.scores.items():
        out.extend(_iter_payload(scorer_name, payload))
    return out


def _iter_payload(scorer_name: str, payload: Any) -> Iterator[DimensionFeedback]:
    """Dispatch one payload to its registered iterator handler.

    Accepts both validated payload models and plain dicts (the latter
    arrives via ``model_dump()`` round-trips and via callers
    constructing ``ScenarioScore`` from raw JSON). Anything without
    a registered ``schema_version`` is a contract violation and raises
    ``ValueError`` immediately - never guessed at.
    """
    if isinstance(payload, dict):
        version = payload.get("schema_version")
    else:
        version = getattr(payload, "schema_version", None)

    if version is None:
        raise ValueError(
            f"scores['{scorer_name}']: missing required 'schema_version'. " f"Registered: {registered_payload_types()}"
        )

    entry = _PAYLOAD_REGISTRY.get(version)
    if entry is None:
        raise ValueError(
            f"scores['{scorer_name}']: schema_version {version!r} is not "
            f"registered. Plugins must call belt.register_payload_type "
            f"before iterating. Registered: {registered_payload_types()}"
        )

    payload_cls, iterator = entry
    if isinstance(payload, payload_cls):
        yield from iterator(scorer_name, payload)
        return
    if isinstance(payload, BaseModel):
        # Different concrete class with the same schema_version (e.g. a
        # plugin re-registered a built-in version with their own model).
        # Round-trip through dict to materialise the registered class.
        yield from iterator(scorer_name, payload_cls.model_validate(payload.model_dump(mode="json")))
        return
    yield from iterator(scorer_name, payload_cls.model_validate(payload))


def iter_llm_payloads(score: Any) -> Iterator[tuple[str, BaseModel]]:
    """Yield ``(scorer_name, payload)`` entries whose payload is LLM-shaped.

    A keyed lookup like ``score.scores.get("llm")`` misses every per-
    judge payload (multi-judge non-consensus declares its own scorer
    keys) and every per-turn payload. Walking via this helper handles
    all LLM-shaped payloads uniformly:

    - :class:`LLMPayload` (scenario-level, single judge or consensus)
    - :class:`PerTurnLLMPayload` (per-turn judging)
    - any plugin-registered subclass of either above

    Order matches dict-insertion (the scorer pipeline writes them in
    config order).
    """
    for scorer_name, payload in score.scores.items():
        if isinstance(payload, (LLMPayload, PerTurnLLMPayload)):
            yield scorer_name, payload


def iter_llm_verdicts(payload: BaseModel) -> Iterator[tuple[str, str, str]]:
    """Yield ``(dimension, score_token, reasoning)`` from any LLM-shaped payload.

    For :class:`PerTurnLLMPayload` the worst-of-turns rollup is applied
    (via :func:`_iter_per_turn_llm`) so the caller sees one row per
    dimension regardless of resolution. The ``reasoning`` for a per-
    turn row is prefixed ``[turn N]`` so a downstream renderer
    surfaces which turn dragged the rollup down.

    Order: dimensions sorted alphabetically (same as
    :func:`_iter_llm`, :func:`_iter_per_turn_llm`).
    """
    if isinstance(payload, LLMPayload):
        for dim in sorted(payload.dimensions):
            v = payload.dimensions[dim]
            yield dim, v.score, v.reasoning
        return
    if isinstance(payload, PerTurnLLMPayload):
        for fb in _iter_per_turn_llm("__rollup__", payload):
            raw = fb.raw or {}
            yield fb.dimension, str(raw.get("worst_score", "")), fb.comment
        return
    raise TypeError(
        f"iter_llm_verdicts: unsupported payload {type(payload).__name__}; " "expected LLMPayload or PerTurnLLMPayload"
    )


__all__ = [
    "CheckEntry",
    "ConsensusMeta",
    "DimensionFeedback",
    "LLMDimensionVerdict",
    "LLMPayload",
    "PayloadIterator",
    "PerTurnLLMPayload",
    "RulesPayload",
    "ScorerPayload",
    "TurnVerdict",
    "UsageStats",
    "iter_dimension_feedback",
    "iter_llm_payloads",
    "iter_llm_verdicts",
    "level_to_score",
    "register_payload_type",
    "registered_payload_types",
]
