# (c) JFrog Ltd. (2026)

"""Cross-phase data contracts and lazy re-exports.

Entities that cross phase boundaries live here directly:
    ToolCall, TurnTiming, TurnOutput, ScenarioScore, AggregatedResults

Domain-specific entities have canonical homes but are re-exported lazily:
    belt.scenario        - GroupConfig, Scenario, Turn, TurnExpectation, StateExpectation
    belt.runner.entities - AgentConfig, ScenarioResult
    belt.scorer.entities - ScoreLevel, DimensionScore, JudgeVerdict, JudgeConfig,
                                ScorerResult
    belt.scorer.payloads - CheckEntry, RulesPayload, LLMDimensionVerdict,
                                LLMPayload, ScorerPayload, DimensionFeedback,
                                iter_dimension_feedback, register_payload_type

``from belt.entities import X`` continues to work for all entities
listed above. Plugins should import from the public top-level instead
(``from belt import X``).
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from belt.scorer.payloads import ScorerPayload, _validate_scores_dict

# ── Agent runtime-error taxonomy (cross-phase contract) ──
#
# Agent runtime failures - the agent did not really run, as opposed to
# the agent ran and answered wrong - are classified into a small set of
# stable tokens carried in :attr:`TurnOutput.error_type`. The same
# tokens are read by the aggregator (``stats.collect_agent_errors``),
# the doctor (``error_types.remediation_for``), the run-phase footer
# (``progress.RunnerProgress.summary``), and the benchmark card
# (``benchmark_card.entities.AgentErrorsSummary``).
#
# Located here (rather than in ``agent/error_types.py``) because
# ``entities.py`` is the cross-phase contract surface: every consumer of
# ``TurnOutput.error_type`` already imports from this module, so the set
# of valid values belongs next to the field declaration. Adding a new
# token is additive (no schema bump). Renaming a token is a breaking
# schema change.
AUTHENTICATION_FAILED = "authentication_failed"
RATE_LIMITED = "rate_limited"
TIMEOUT = "timeout"
# Distinct from AUTHENTICATION_FAILED: the user is authenticated but
# the requested model is unknown to the provider or not entitled to the
# user's project. The fix is a model swap or an entitlement request, not
# re-authentication. Empirically surfaces on OpenAI-backed adapters
# (codex, opencode/wandb, copilot) as
# ``Project <id> does not have access to model <name>`` or
# ``model_not_found``. Treated as ENVIRONMENTAL because the agent never
# ran - rolling it into ``unknown`` would attribute the failure to the
# agent's own conduct and contaminate task pass-rate metrics.
MODEL_UNAVAILABLE = "model_unavailable"
REFUSED = "refused"
UNKNOWN = "unknown"

ERROR_TYPES: frozenset[str] = frozenset(
    {AUTHENTICATION_FAILED, RATE_LIMITED, TIMEOUT, MODEL_UNAVAILABLE, REFUSED, UNKNOWN}
)

# Partition of ``ERROR_TYPES`` for the headline split.
#
# - ENVIRONMENTAL errors are transient or external. The agent could not
#   complete because of authentication, throttling, a per-turn deadline,
#   or a model entitlement gap. The user fixes their environment, not
#   their scenario.
# - TASK errors are the agent's own behaviour. The agent refused, or
#   failed in a way the framework couldn't classify - the agent's own
#   conduct, not the environment.
#
# Together the two sets MUST partition ``ERROR_TYPES`` exactly: every
# token belongs to exactly one bucket. Adding a new token requires
# placing it in one of the two sets; the partition invariant is
# pinned by a parity test.
ENVIRONMENTAL_ERROR_TYPES: frozenset[str] = frozenset({AUTHENTICATION_FAILED, RATE_LIMITED, TIMEOUT, MODEL_UNAVAILABLE})
TASK_ERROR_TYPES: frozenset[str] = frozenset({REFUSED, UNKNOWN})

# ── Cross-phase entities (used by 2+ phases) ──


class ToolCall(BaseModel):
    name: str
    call_id: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: Optional[dict[str, Any]] = None


class TurnTiming(BaseModel):
    """Timing breakdown for a single turn.

    Universal (all agents):
        total - wall-clock duration in seconds.

    Agent-specific (populated by agents that support streaming timing):
        ttfe - time to first event (assistant starts processing).
        ttft - time to first token (streaming response begins).
        ttlt - time to last token (streaming response ends).
    """

    ttfe: Optional[float] = None
    ttft: Optional[float] = None
    ttlt: Optional[float] = None
    total: Optional[float] = None


class VerifyResult(BaseModel):
    """Outcome of a ``VerifySpec`` command executed by the runner.

    Captured into :attr:`TurnOutput.verify_result` (per-turn) or
    :attr:`TurnOutput.scenario_verify_result` (per-scenario, on the final
    turn) so the rule-based scorer can assert on it without re-executing.
    ``cmd`` is stored so the artifact and every renderer are self-describing
    (a reader sees *what* was run, not just the exit code).
    """

    cmd: list[str] = Field(default_factory=list)
    exit_code: int
    stdout: str = ""
    duration_s: Optional[float] = None


class TurnOutput(BaseModel):
    """Normalized output from a single conversation turn, produced by an agent.

    Fields are divided into two tiers:

    **Universal (all agents should populate):**
        raw_cli, reply_text, tool_calls, has_reply, has_error, timing

    **Agent-specific (optional - agents populate when meaningful, others leave defaults):**
        raw_state, llm_turn_count, error_type, thinking_text, tool_sequence

    Scenario expectations referencing agent-specific fields are only useful for
    agents that populate them.  Checks against default values (empty list, None,
    False) silently pass - this is by design so that shared scenarios don't break
    when run against agents that lack these signals.

    **Plugin extension fields:** ``model_config`` sets ``extra="allow"`` so adapter
    plugins can attach arbitrary additional fields (e.g., a multi-agent framework
    tracking handoff counts, a review-loop framework tracking pending reviews).
    Plugins read their own extras via ``output.model_extra["my_field"]`` from a
    custom scorer registered under the ``belt.scorers`` entry-point group.
    Core scorers and the LLM judge prompt only consume the fields declared here.
    """

    model_config = ConfigDict(extra="allow")

    # ── Schema versioning ──
    schema_version: Optional[str] = None

    # ── Universal fields ──
    raw_cli: str
    reply_text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    has_reply: bool = False
    has_error: Optional[bool] = None
    timing: Optional[TurnTiming] = None

    cost_usd: Optional[float] = None
    error_type: Optional[str] = None

    # ── Workspace state (populated by orchestrator from StateExpectation) ──
    workspace_files: dict[str, Optional[str]] = Field(default_factory=dict)
    git_diff: Optional[str] = Field(default=None, description="Git diff captured from isolated worktree after turn")
    files_modified: list[str] = Field(default_factory=list, description="File paths from git diff --name-only")

    # ── Agent-specific fields (populated when the agent supports them) ──
    raw_state: Optional[str] = None
    llm_turn_count: Optional[int] = None
    thinking_text: Optional[str] = None
    tool_sequence: list[str] = Field(default_factory=list)

    # ── Verify results (populated by the runner when a VerifySpec is set) ──
    # ``verify_result`` is the per-turn ``Turn.verify`` outcome; the per-scenario
    # ``Scenario.verify`` outcome is recorded on the final turn under
    # ``scenario_verify_result``. Distinct fields so a scenario using both on
    # the last turn never collides. See docs/glossary/SECURITY-MODEL.md.
    verify_result: Optional[VerifyResult] = None
    scenario_verify_result: Optional[VerifyResult] = None


class ScenarioScore(BaseModel):
    """Combined score for a scenario.

    Keys in :attr:`scores` match scorer names (e.g. ``"rules"``,
    ``"llm"``). Values are typed payloads from
    :mod:`belt.scorer.payloads`; the registry-backed validator
    dispatches on each payload's ``schema_version`` to construct the
    right concrete class, so consumers reach into typed attributes
    (``payload.checks``, ``payload.dimensions``, ``payload.usage``)
    instead of stringly-typed dict lookups.

    Plugins iterating across scorers should call
    :func:`belt.iter_dimension_feedback` rather than walking
    :attr:`scores` directly - the function handles every registered
    scorer payload (built-in + plugin) uniformly.
    """

    schema_version: Optional[str] = None
    scenario_name: str
    group: str
    # Effective tags = scenario.tags ∪ group_config.default_tags. Populated by
    # the scorer so the aggregator can render tag-aware annotations (e.g. a
    # footnote pointing at ``--tags real-runnable`` when failures are
    # ``dry-run-only`` showcase examples) without re-loading scenario files.
    tags: list[str] = Field(default_factory=list)
    scores: dict[str, ScorerPayload] = Field(default_factory=dict)
    overall_pass: bool
    judge_cost_usd: Optional[float] = None
    scorer_prompt_tokens: int = 0
    scorer_completion_tokens: int = 0

    # ``BeforeValidator`` on the dict consults the runtime payload
    # registry: each entry is rebuilt as the registered concrete class
    # (``RulesPayload``, ``LLMPayload``, or whatever a plugin registered).
    # Missing or unknown ``schema_version`` raises ``ValueError`` here so
    # bad shapes never reach the iterator phase.
    _scores_validator = field_validator("scores", mode="before")(classmethod(lambda cls, v: _validate_scores_dict(v)))


class AggregatedResults(BaseModel):
    """Typed model of the ``results.json`` artifact written by ``belt aggregate``.

    Promoted from a hand-built dict so the export phase has a stable Pydantic
    contract instead of stringly-typed key lookups (Design Principle 2).

    The free-form ``dict`` sub-blocks (``stats``, ``cost_timing``,
    ``reliability``) are produced by ``aggregator/stats.py`` and consumed
    read-only by exporters and external tooling. Promoting any of them to a
    dedicated entity is purely additive and can happen as the shape
    stabilises.
    """

    schema_version: Optional[str] = None
    total: int = 0
    passed: int = 0
    failed: int = 0
    overall_pass: bool = True
    # Count of scenario JSON files dropped by ``parse_and_filter`` because
    # they failed Pydantic validation (typo in a field, malformed JSON,
    # forbidden extra key). Threaded from ``run_meta.json`` so the
    # aggregator's contract stays "the run was meant to be N + M, of
    # which M never made it past the parser". A non-zero value means
    # ``--strict`` would have aborted the run; CI dashboards should
    # surface this prominently. Additive field, default ``0`` - old
    # ``results.json`` files load unchanged.
    scenarios_skipped: int = 0
    stats: dict[str, Any] = Field(default_factory=dict)
    cost_timing: dict[str, Any] = Field(default_factory=dict)
    reliability: Optional[dict[str, Any]] = None
    # First-class record of agent runtime failures (auth, rate-limit,
    # refusal, timeout). Populated by
    # :func:`belt.aggregator.stats.collect_agent_errors`. ``None``
    # when no scenario reported ``has_error`` on any turn. Distinct from
    # rule failures: a scenario with rules passing AND agent errored is a
    # *vacuous pass*, captured under ``vacuous_passes``.
    agent_errors: Optional[dict[str, Any]] = None
    # First-class record of LLM-judge infrastructure failures
    # (rate-limit, timeout, network). Populated by
    # :func:`belt.aggregator.stats.collect_judge_errors`. ``None`` when
    # no scenario's judge errored. Mirrors ``agent_errors`` but for the
    # judge side of the run; the aggregator's task-quality partition
    # subtracts both axes from the headline pass-rate denominator so a
    # judge outage cannot manifest as a false green or false red.
    judge_errors: Optional[dict[str, Any]] = None
    bottom_line: list[str] = Field(default_factory=list)
    thresholds_passed: Optional[bool] = None
    scenarios: list[dict[str, Any]] = Field(default_factory=list)
    # Per-group setup failures captured by the run phase before any
    # scenario produces a turn. Stored as a list (preserves the order
    # groups were rejected in) of ``{"group": name, "scenarios":
    # [names], "error": message}`` records so downstream consumers
    # (``belt view``, ``belt compare``, exporters) can surface the cause
    # at the run-level instead of mis-attributing it to a per-scenario
    # error. Empty list means "no group setup failed" - additive field,
    # old ``results.json`` files load unchanged.
    setup_errors: list[dict[str, Any]] = Field(default_factory=list)


# ── Lazy re-exports ──

_SCENARIO_NAMES = frozenset(
    {"GroupConfig", "Scenario", "StateExpectation", "Turn", "TurnExpectation", "TurnJudgeOverride"}
)
_RUNNER_NAMES = frozenset({"AgentConfig", "ScenarioResult"})
_SCORER_ENTITY_NAMES = frozenset(
    {
        "DimensionScore",
        "EvidenceScope",
        "JudgeConfig",
        "JudgeVerdict",
        "Resolution",
        "ScoreLevel",
        "ScorerResult",
    }
)
_SCORER_PAYLOAD_NAMES = frozenset(
    {
        "CheckEntry",
        "ConsensusMeta",
        "DimensionFeedback",
        "LLMDimensionVerdict",
        "LLMPayload",
        "PerTurnLLMPayload",
        "RulesPayload",
        "TurnVerdict",
        "UsageStats",
        "iter_dimension_feedback",
        "iter_llm_payloads",
        "iter_llm_verdicts",
        "level_to_score",
        "register_payload_type",
        "registered_payload_types",
    }
)


def __getattr__(name: str) -> Any:
    if name in _SCENARIO_NAMES:
        from belt import scenario as _mod

        return getattr(_mod, name)
    if name in _RUNNER_NAMES:
        from belt.runner import entities as _mod

        return getattr(_mod, name)
    if name in _SCORER_ENTITY_NAMES:
        from belt.scorer import entities as _mod

        return getattr(_mod, name)
    if name in _SCORER_PAYLOAD_NAMES:
        from belt.scorer import payloads as _mod

        return getattr(_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [  # noqa: F822
    # cross-phase (defined here)
    "AggregatedResults",
    "ScenarioScore",
    "ToolCall",
    "TurnOutput",
    "TurnTiming",
    # scenario (re-exported via __getattr__)
    "GroupConfig",
    "Scenario",
    "StateExpectation",
    "Turn",
    "TurnExpectation",
    "TurnJudgeOverride",
    # runner (re-exported via __getattr__)
    "AgentConfig",
    "ScenarioResult",
    # scorer entities (re-exported via __getattr__)
    "DimensionScore",
    "EvidenceScope",
    "JudgeConfig",
    "JudgeVerdict",
    "Resolution",
    "ScoreLevel",
    "ScorerResult",
    # scorer payloads (re-exported via __getattr__)
    "CheckEntry",
    "ConsensusMeta",
    "DimensionFeedback",
    "LLMDimensionVerdict",
    "LLMPayload",
    "PerTurnLLMPayload",
    "RulesPayload",
    "TurnVerdict",
    "UsageStats",
    "iter_dimension_feedback",
    "level_to_score",
    "register_payload_type",
    "registered_payload_types",
]
