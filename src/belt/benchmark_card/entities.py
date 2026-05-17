# (c) JFrog Ltd. (2026)

"""Pydantic data contracts for the reproducibility manifest.

These models are the cross-phase contract: the run phase populates them
into ``run_meta.json`` and per-scenario sidecars; the aggregate phase
reads them back, merges them with results, and emits
``benchmark-card.json``. The schema is at v1; adding optional fields with
sensible defaults extends the contract without bumping ``SCHEMA_VERSION``,
while renaming, removing, or changing the type of a field does.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from belt.constants import SCHEMA_VERSION


class BeltProvenance(BaseModel):
    """Identity of the belt install that produced the run."""

    version: str
    install_kind: str  # "wheel" | "editable" | "unknown"
    git_sha: Optional[str] = None
    git_dirty: Optional[bool] = None


class HostProvenance(BaseModel):
    """Operating system and Python runtime that hosted the run."""

    os: str  # "Darwin 25.4.0" | "Linux 6.8.0-1014-azure"
    machine: str  # "arm64" | "x86_64"
    python_version: str  # "3.12.4"
    python_implementation: str  # "CPython"
    package_versions: dict[str, str] = Field(default_factory=dict)


class Invocation(BaseModel):
    """How the user invoked belt, after secret redaction."""

    argv: list[str] = Field(default_factory=list)
    parsed_args: dict[str, Any] = Field(default_factory=dict)
    cwd: str = ""
    env: dict[str, str] = Field(default_factory=dict)


class ScenarioFile(BaseModel):
    """A scenario JSON file content-hashed for reproducibility."""

    relpath: str  # relative to scenarios_root
    sha256: str


class ScenarioSelection(BaseModel):
    """Scenarios that were eligible to run (post-filter)."""

    scenarios_root: str
    selected_groups: list[str] = Field(default_factory=list)
    selected_tags: list[str] = Field(default_factory=list)
    excluded_tags: list[str] = Field(default_factory=list)
    scenario_files: list[ScenarioFile] = Field(default_factory=list)


class FixtureProvenance(BaseModel):
    """Per-group fixture state captured at group-setup time.

    Only meaningful when ``workspace_isolation == "git-worktree"``. For
    plain-directory fixtures, ``tracked == False`` and the SHA fields are
    ``None``.
    """

    group: str
    working_dir: Optional[str] = None
    tracked: bool = False
    git_sha: Optional[str] = None
    git_ref: Optional[str] = None  # the ``workspace_ref`` requested
    auto_initialized: bool = False
    dirty_files: int = 0


class AgentIdentity(BaseModel):
    """Logical agent identity: registry name, adapter class, user-supplied args.

    Distinct from :class:`CliIdentity` (the *binary* the agent invoked)
    so a future change in either dimension does not force the other to
    move. ``args`` is the post-redaction snapshot of every ``-X``
    flag the user passed for this agent.
    """

    name: str  # registry name (e.g. "cursor", "claude-code")
    adapter_class: str  # e.g. "CursorAgentAdapter"
    args: dict[str, str] = Field(default_factory=dict)
    auth_signals: list[str] = Field(default_factory=list)


class CliIdentity(BaseModel):
    """The CLI binary an adapter invoked, captured at ``setup`` time.

    Both fields default to ``None`` so an adapter that does not wrap a
    CLI (e.g. an in-process agent) still produces a card; the same
    defaults apply when an adapter's ``runtime_info`` override declines
    to populate them.
    """

    binary_path: Optional[str] = None
    version: Optional[str] = None


class AgentProvenance(BaseModel):
    """Per-group agent record on the card, sourced from the ``_runtime_info`` sidecar.

    The two-level shape (``agent`` / ``cli``) mirrors the persisted
    sidecar - see :func:`belt.runner.orchestrator._write_runtime_info_sidecar`
    for how the framework projects each adapter's flat
    :meth:`runtime_info` return value into this structure. Adapter
    authors continue to write flat keys; the framework owns the
    persisted layout.
    """

    group: str
    agent: AgentIdentity
    cli: CliIdentity = Field(default_factory=CliIdentity)


class JudgeProvenance(BaseModel):
    """A single LLM judge backend referenced by the scoring configuration."""

    provider: str
    model: str
    base_url: Optional[str] = None  # already redacted to scheme+host
    dimensions: list[str] = Field(default_factory=list)


class ScoringConfig(BaseModel):
    """Resolved scoring configuration for the run."""

    modes: list[str] = Field(default_factory=list)  # ["rules", "llm"]
    consensus: Optional[str] = None  # consensus mode if any
    thresholds: dict[str, Any] = Field(default_factory=dict)
    judges: list[JudgeProvenance] = Field(default_factory=list)


class RuntimeConfig(BaseModel):
    """Runner-level knobs that affect timing, parallelism, and error surfaces."""

    workers: int = 1
    trials: int = 1
    streaming: bool = True
    scenario_delay_s: float = 0.0


class CostTimingSummary(BaseModel):
    """Run-level cost and latency summary, sourced from ``results.json``."""

    agent_cost_usd: Optional[float] = None
    judge_cost_usd: Optional[float] = None
    total_cost_usd: Optional[float] = None
    total_seconds: Optional[float] = None
    mean_seconds: Optional[float] = None


class ScoreSummary(BaseModel):
    """Pass/fail summary, sourced from ``results.json``."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    overall_pass: bool = False
    pass_rate: float = 0.0
    thresholds_passed: Optional[bool] = None


class TaskQualitySplit(BaseModel):
    """Task-quality vs environmental-health partition of one run.

    Mirrors :func:`belt.aggregator.stats.build_task_quality_split`. The
    aggregator's partition has TWO environmental axes (agent vs judge)
    plus a task axis, and the card preserves all three so downstream
    tooling can attribute env failures correctly:

    - ``env_failed_agent``: scenarios where the agent CLI hit
      auth / rate-limit / timeout. Maps to
      :data:`belt.entities.ENVIRONMENTAL_ERROR_TYPES`.
    - ``env_failed_judge``: scenarios where the LLM judge backend hit
      its own infra failure
      (:data:`belt.scorer.entities.JUDGE_ERROR_TYPES`). Agent-axis
      wins on overlap (a scenario whose agent errored cannot also be
      counted as a judge env failure).
    - ``env_failed``: ``env_failed_agent + env_failed_judge``. Carried
      for callers that only want the rolled-up environmental count.
    - ``task_failed``: ``completed - passed``. The agent ran, the
      judge ran, the agent did the wrong thing.

    Present on the card only when at least one scenario was blocked by
    an environmental error. When absent, the run had no environmental
    failures and the single-axis ``ScoreSummary`` already tells the
    right story.

    The ``passed/completed`` ratio is the number a CI dashboard can
    defensibly publish: it excludes scenarios the agent never got to
    attempt because of transient external failures.
    """

    attempted: int = 0
    env_failed: int = 0
    env_failed_agent: int = 0
    env_failed_judge: int = 0
    completed: int = 0
    passed: int = 0
    task_failed: int = 0
    # ``None`` when ``completed == 0`` (every scenario was env-blocked).
    pct: Optional[float] = None


class AgentErrorsSummary(BaseModel):
    """Run-level agent runtime-failure summary.

    Mirrors :func:`belt.aggregator.stats.collect_agent_errors`. A
    populated card with this block tells a downstream consumer that the
    agent itself failed at runtime (auth, rate-limit, refusal) on at
    least one turn, distinct from rule failures.
    """

    scenarios_with_errors: int = 0
    scenarios_total: int = 0
    vacuous_passes: int = 0
    by_error_type: dict[str, int] = Field(default_factory=dict)
    remediation: Optional[str] = None
    # ``None`` when no scenario hit an environmental error type
    # (auth / rate-limit / timeout). When present, downstream tooling
    # should prefer ``task_quality.passed / task_quality.completed`` as
    # the headline pass-rate.
    task_quality: Optional[TaskQualitySplit] = None


class JudgeErrorsSummary(BaseModel):
    """Run-level LLM-judge infrastructure-failure summary.

    Mirrors :func:`belt.aggregator.stats.collect_judge_errors`. The
    judge axis is structurally distinct from the agent axis: a scenario
    can have a clean agent run and still be unscored if the judge
    backend rate-limited mid-run. Surfacing both axes on the card lets
    downstream tooling attribute environmental failures to the right
    layer (provider key vs judge key, agent CLI auth vs judge backend
    auth) instead of collapsing them into a single ``env_failed`` bucket.

    The ``by_error_type`` keys are
    :data:`belt.scorer.entities.JUDGE_ERROR_TYPES` tokens.
    """

    scenarios_with_errors: int = 0
    scenarios_total: int = 0
    by_error_type: dict[str, int] = Field(default_factory=dict)


class BenchmarkCard(BaseModel):
    """Reproducibility manifest for one ``belt eval`` run.

    Every field is either deterministic (timestamps, SHAs, versions) or
    redacted via :mod:`belt._redact` (env vars, agent args, base
    URLs). No field admits raw user-supplied strings without going
    through the redactor.
    """

    schema_version: str = SCHEMA_VERSION
    run_id: str
    started_at: str  # ISO-8601, UTC
    ended_at: str  # ISO-8601, UTC
    belt: BeltProvenance
    host: HostProvenance
    invocation: Invocation
    scenarios: ScenarioSelection
    fixtures: list[FixtureProvenance] = Field(default_factory=list)
    agents: list[AgentProvenance] = Field(default_factory=list)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    cost_timing: CostTimingSummary = Field(default_factory=CostTimingSummary)
    summary: ScoreSummary = Field(default_factory=ScoreSummary)
    # ``None`` when no scenario reported ``has_error`` on any turn. The
    # presence of this field on a card is itself the signal that "agent
    # didn't really run" failures occurred.
    agent_errors: Optional[AgentErrorsSummary] = None
    # ``None`` when every scenario's LLM judge produced a verdict. The
    # presence of this field on a card is itself the signal that "judge
    # didn't really vote" failures occurred. Structurally parallel to
    # ``agent_errors`` so downstream tooling can attribute environmental
    # failures to the right backend.
    judge_errors: Optional[JudgeErrorsSummary] = None
    links: dict[str, str] = Field(default_factory=dict)
