# (c) JFrog Ltd. (2026)

"""Scenario schema entities - scenario authoring and group configuration."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, StringConstraints, field_validator

from belt._regex_policy import compile_user_regex
from belt.constants import TURN_MESSAGE_MAX_CHARS

# Filesystem-safe identifier - no path separators, parent-traversal segments,
# Rich/Markdown control characters, or whitespace. Scenario name and tags both
# end up as path segments (``run_dir/<group>/<scenario>/...``) and as text in
# CLI panels and GitHub markdown summaries; restricting to a conservative
# ASCII set prevents both path traversal and markup-injection vectors at the
# rendering layer. Authors who need spaces should use ``-`` or ``_`` in
# scenario names; descriptive prose belongs in ``description``.
_SAFE_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,255}$"
_SAFE_TAG_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,63}$"

SafeTag = Annotated[str, StringConstraints(pattern=_SAFE_TAG_PATTERN)]


class Resource(BaseModel):
    """A resource installed into the agent workspace before scenarios run.

    ``kind`` selects the install method:

    - ``file`` -- copy ``source`` to ``dest`` (single file or directory).
    - ``archive`` -- extract ``source`` into ``dest`` (``.zip``, ``.tar``,
      ``.tar.gz`` / ``.tgz``, or ``.tar.bz2``).

    ``source`` is a local filesystem path or a ``file://`` / ``https://`` URL.
    ``dest`` is a relative path inside the worktree (path traversal segments
    are rejected). ``version`` is a free-form label captured in the
    per-scenario ``resource_lock.json`` so cross-run comparisons can pin
    "skill X v0.91.0 vs v0.92.0" alongside the source SHA-256.
    """

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(description="Resource kind. One of: ``file``, ``archive``.")
    source: str = Field(description="Local path or URL of the resource.")
    dest: str = Field(description="Destination path relative to the worktree root.")
    name: Optional[str] = Field(
        default=None,
        description="Optional label for the resource. Defaults to the basename of ``dest``.",
    )
    version: Optional[str] = Field(
        default=None,
        description="Optional version label captured in the per-scenario resource lock.",
    )


class TurnExpectation(BaseModel):
    """Deterministic checks for a single conversation turn.

    **Plugin extension keys:** ``model_config`` sets ``extra="allow"`` so adapter
    plugins can declare additional expectation keys in scenario JSON (e.g., a
    multi-agent framework asserting ``max_handoffs`` or a review-loop framework
    asserting ``review_prompted``). Plugins consume these via
    ``expect.model_extra["my_key"]`` from a custom scorer registered under the
    ``belt.scorers`` entry-point group. Built-in scorers ignore unknown keys.
    """

    model_config = ConfigDict(extra="allow")

    no_errors: bool = True
    not_contains: list[str] = Field(default_factory=list)

    tools_invoked: list[str] = Field(default_factory=list)
    tools_invoked_any: list[list[str]] = Field(default_factory=list)
    tools_invoked_in_order: list[str] = Field(default_factory=list)
    only_used_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    tool_args_contain: dict[str, dict[str, Any]] = Field(default_factory=dict)
    tool_result_contains: dict[str, str] = Field(default_factory=dict)
    tool_result_pattern: dict[str, str] = Field(default_factory=dict)
    skills_invoked: list[str] = Field(default_factory=list)

    has_reply: bool = True
    contains: list[str] = Field(default_factory=list)
    reply_pattern: list[str] = Field(
        default_factory=list,
        description=(
            "Strict reply-format assertions as Python regexes. ALL patterns "
            "must match (each via ``re.search``, ``re.IGNORECASE``). Matches "
            "against ``reply_text`` only - unlike ``contains``, there is no "
            "``raw_cli`` fallback, so noise in stderr or the agent-trace "
            "transcript cannot produce a false green. Single-line semantics "
            "by default; opt into per-line matching with the inline ``(?m)`` "
            "flag. Bad regexes are rejected at scenario-load time, not at "
            "scoring time."
        ),
    )

    max_llm_turns: Optional[int] = None
    max_tool_calls: Optional[int] = None
    max_cost_usd: Optional[float] = None

    error_type_is: Optional[str] = None
    has_thinking: Optional[bool] = None

    max_ttfe_seconds: Optional[float] = None
    max_ttft_seconds: Optional[float] = None
    max_ttlt_seconds: Optional[float] = None
    max_total_seconds: Optional[float] = None

    # ── File diff checks (populated when workspace isolation is active) ──
    files_modified_any: list[str] = Field(
        default_factory=list,
        description="At least one of these paths must appear in git diff",
    )
    files_modified_exact: list[str] = Field(
        default_factory=list,
        description="Exact set of modified files expected",
    )
    files_not_modified: list[str] = Field(
        default_factory=list,
        description="These paths must NOT appear in git diff",
    )
    git_diff_contains: list[str] = Field(
        default_factory=list,
        description="Substrings that must appear in the git diff output",
    )

    # Compiled-regex caches populated in ``model_post_init`` after the
    # ``field_validator`` below has confirmed every author-supplied
    # pattern parses cleanly. Stored as ``PrivateAttr`` so they are
    # excluded from ``model_dump`` / ``model_dump_json`` (the on-disk
    # contract is the source pattern strings, not the runtime
    # :class:`re.Pattern` objects). Consumed by
    # :mod:`belt.scorer.rules.helpers` and
    # :mod:`belt.scorer.rules.response` so the runtime never recompiles
    # a user pattern - one compile per scenario load, regardless of how
    # many turns or judges score it.
    _compiled_reply_patterns: list[re.Pattern[str]] = PrivateAttr(default_factory=list)
    _compiled_tool_patterns: dict[str, re.Pattern[str]] = PrivateAttr(default_factory=dict)

    @field_validator("files_modified_any", "files_modified_exact", "files_not_modified")
    @classmethod
    def _reject_directory_paths(cls, v: list[str], info: Any) -> list[str]:
        # The file-diff scorer compares each entry against a flat list of modified
        # file paths via literal equality. A trailing-slash directory path would
        # never match any concrete file and silently pass, hiding regressions.
        # Reject at load time so authors find out before the scenario is sitting
        # in CI giving false-greens.
        bad = [p for p in v if p.endswith("/") or p.endswith("\\")]
        if bad:
            raise ValueError(
                f"{info.field_name}: directory-shaped paths are not supported "
                f"(entries: {bad}). The file-diff scorer compares against a flat "
                f"list of modified file paths via literal equality, so a trailing-"
                f"slash entry would silently pass even when the agent modified "
                f"files inside that directory. Use specific file paths instead "
                f"(e.g. 'src/billing/handler.py' rather than 'src/billing/')."
            )
        return v

    @field_validator("reply_pattern")
    @classmethod
    def _validate_reply_pattern_regexes(cls, v: list[str], info: Any) -> list[str]:
        # Compile every entry through the canonical policy so a typo in a
        # regex surfaces at scenario-load time, not as a silently-failing
        # CheckEntry months later. Aggregate every offender into one
        # error so authors see the full list in one report instead of
        # fix-one-at-a-time.
        errors = []
        for idx, pattern in enumerate(v):
            try:
                compile_user_regex(pattern)
            except ValueError as exc:
                errors.append(f"  [{idx}] {exc}")
        if errors:
            raise ValueError(f"{info.field_name}: invalid regex(es):\n" + "\n".join(errors))
        return v

    @field_validator("tool_result_pattern")
    @classmethod
    def _validate_tool_result_pattern_regexes(cls, v: dict[str, str], info: Any) -> dict[str, str]:
        # Same load-time gate as ``reply_pattern``: every entry compiles
        # through the canonical policy, malformed entries aggregate into
        # one error report. Both surfaces share one contract so a typo
        # never reaches the runtime as a silent fail.
        errors = []
        for tool_name, pattern in v.items():
            try:
                compile_user_regex(pattern)
            except ValueError as exc:
                errors.append(f"  [{tool_name}] {exc}")
        if errors:
            raise ValueError(f"{info.field_name}: invalid regex(es):\n" + "\n".join(errors))
        return v

    def model_post_init(self, __context: Any) -> None:
        # ``field_validator`` above guarantees every entry compiles, so
        # the calls here cannot raise. Cache once per scenario load; the
        # rule scorers consume these private attrs instead of recompiling
        # per turn / per judge.
        self._compiled_reply_patterns = [compile_user_regex(p) for p in self.reply_pattern]
        self._compiled_tool_patterns = {
            tool_name: compile_user_regex(pattern) for tool_name, pattern in self.tool_result_pattern.items()
        }


class StateExpectation(BaseModel):
    """Post-turn workspace filesystem checks.

    Paths are relative to the workspace root (typically cwd where belt runs).
    The orchestrator captures the referenced files after each turn and stores them
    in TurnOutput.workspace_files so the scorer can verify without workspace access.
    """

    model_config = ConfigDict(extra="forbid")

    files_exist: list[str] = Field(default_factory=list)
    files_contain: dict[str, str] = Field(default_factory=dict)
    files_not_exist: list[str] = Field(default_factory=list)
    capture_git_diff: bool = False


class TurnJudgeOverride(BaseModel):
    """Per-turn override for an LLM judge configured at scenario / config level.

    Authored inside :attr:`Turn.llm_judges` as ``{<judge_name>: <override>}``.
    Every field is optional; an empty override (``{}``) is meaningful as a
    "this judge runs for this turn with no override" marker (consumed by
    the per-turn preflight to distinguish a non-mention from an explicit
    no-op).

    All fields are sized and shaped to match the scenario-level
    equivalents so per-turn rubrics inherit the same security envelope:

    - ``instruction``: capped at the same ``max_length`` as
      :attr:`Scenario.llm_scorer_instruction`; flows through the same
      ``</scenario_instruction>`` fence-neutralisation as the scenario
      path (:func:`belt.scorer.llm.scorer._build_dynamic_message`).
    - ``dimensions``: per-turn dimension list, parsed via the same
      :func:`belt.scorer.scenario_map.parse_dimension_defs` helper as
      the group / config path. ``list[Any]`` for parity with
      :attr:`belt.scorer.config_schema.JudgeDef.dimensions` and the
      runtime helper's accepted shape.
    - ``extend_default_dimensions``: when ``True``, declared dimensions
      extend (rather than replace) the judge's configured dimensions
      for this turn only.
    - ``evidence_files``: capped at the same length as
      :attr:`Scenario.llm_scorer_evidence_files`; resolved through the
      same path-traversal check via
      :func:`belt.scorer.llm.scorer._render_evidence_files_from`.
    - ``skip``: short-circuit this judge for this turn; the turn's
      :class:`belt.scorer.payloads.TurnVerdict` is recorded with empty
      ``dimensions``. The runtime taint rule in
      ``LLMScorer._score_per_turn`` AND the preflight check in
      ``validate_per_turn_judges_against_scenarios`` (both in
      :mod:`belt.scorer.pipeline`) refuse an all-skipped judge so a
      scenario can never vacuously pass.
    """

    model_config = ConfigDict(extra="forbid")

    instruction: Optional[str] = Field(default=None, max_length=10_000)
    dimensions: Optional[list[Any]] = Field(default=None, max_length=50)
    extend_default_dimensions: bool = False
    evidence_files: Optional[list[str]] = Field(default=None, max_length=20)
    skip: bool = False


class VerifySpec(BaseModel):
    """A deterministic verification command run in the scenario worktree.

    Declared on :attr:`Turn.verify` (runs after that turn) or
    :attr:`Scenario.verify` (runs once after the final turn). The runner
    executes ``cmd`` in the per-scenario worktree, through the active sandbox
    provider, and records the exit code and captured stdout into
    :class:`belt.entities.TurnOutput` for the rule-based scorer to assert on.

    Security envelope (see ``docs/glossary/SECURITY-MODEL.md``): ``cmd`` is an
    argv list, never a shell string; the command runs only with an isolated
    worktree and only when ``--allow-verify-exec`` (or
    ``BELT_ALLOW_VERIFY_EXEC=1``) is set - the runner refuses the group at
    setup otherwise (default-deny, because this executes an author-supplied
    command). ``output_contains`` entries are plain substrings, not regex.
    """

    model_config = ConfigDict(extra="forbid")

    cmd: list[str] = Field(
        min_length=1,
        description="Command argv (no shell). E.g. ['python', '-m', 'pytest', '-q'].",
    )
    exit_code: int = Field(default=0, description="Expected process exit code (the check passes when it matches).")
    output_contains: list[str] = Field(
        default_factory=list,
        max_length=50,
        description="Plain substrings (not regex) that must ALL appear in captured stdout.",
    )
    timeout: int = Field(
        default=300,
        gt=0,
        description="Maximum seconds before the command is killed; a timeout fails the check.",
    )

    @field_validator("cmd")
    @classmethod
    def _reject_empty_argv(cls, v: list[str]) -> list[str]:
        if not v or any((not isinstance(a, str) or a == "") for a in v):
            raise ValueError("verify.cmd must be a non-empty list of non-empty strings (argv form, no shell).")
        return v


class Turn(BaseModel):
    """A single conversation turn."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(max_length=TURN_MESSAGE_MAX_CHARS)
    flags: list[str] = Field(default_factory=list, max_length=50)
    expect: TurnExpectation = Field(default_factory=TurnExpectation)
    state_expect: StateExpectation = Field(default_factory=StateExpectation)
    verify: Optional[VerifySpec] = Field(
        default=None,
        description="Deterministic command run after this turn in the worktree; asserted as the `verify` dimension.",
    )
    # Per-turn LLM judge overrides. Keyed by judge name (matches the
    # name declared in ``--scorer-config`` YAML, or the implicit
    # ``"llm"`` judge name for single-judge runs). ``max_length=10``
    # caps cost amplification: combined with ``Scenario.turns:
    # max_length=100`` and user-supplied ``--trials N`` the worst-case
    # per-scenario judge call count is bounded at
    # ``100 × 10 × N``. Group-config ``llm_dimensions`` and
    # ``llm_dimensions_extend_defaults`` apply to scenario-level judges
    # only; per-turn judges read dimensions from the scorer config plus
    # per-turn overrides only.
    llm_judges: dict[str, TurnJudgeOverride] = Field(default_factory=dict, max_length=10)


class Scenario(BaseModel):
    """A reproducible multi-turn conversation scenario."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=_SAFE_NAME_PATTERN, max_length=256)
    description: str = Field(max_length=10_000)
    tags: list[SafeTag] = Field(default_factory=list, max_length=50)
    llm_scorer_instruction: str = Field(
        default="",
        max_length=10_000,
        description="Optional per-scenario instruction appended to the LLM judge system message",
    )
    llm_scorer_raw_transcript: bool = Field(
        default=False,
        description=(
            "When true, append the agent's full raw CLI transcript (TurnOutput.raw_cli) "
            "as a low-priority '## Raw CLI Output' section in the LLM judge prompt. "
            "Default false: the judge sees a structured summary built from TurnOutput "
            "fields (reply, tool sequence, metadata) instead of the noisy NDJSON. "
            "Opt in only when the scenario's evaluation genuinely depends on event-level "
            "transcript inspection."
        ),
    )
    llm_scorer_evidence_files: list[str] = Field(
        default_factory=list,
        max_length=20,
        description=(
            "Paths (relative to the scenario JSON's directory) to rubric or "
            "ground-truth files that are read into the LLM judge prompt but "
            "never copied into the agent's worktree. Lets a scenario keep its "
            "expected-findings document outside the workspace so the agent "
            "cannot peek at it."
        ),
    )
    turns: list[Turn] = Field(max_length=100)
    verify: Optional[VerifySpec] = Field(
        default=None,
        description=(
            "Deterministic command run once after the final turn (end-of-conversation), "
            "in the worktree; asserted as the `verify` dimension."
        ),
    )

    # Set by ``ScenarioLoader.load_scenario`` to the directory containing the
    # scenario JSON. Used by the LLM scorer to resolve
    # ``llm_scorer_evidence_files`` paths and reject anything that escapes
    # that directory. Private so it never round-trips through
    # ``model_dump_json`` (the scorer renders the scenario into the judge
    # prompt and we do not want filesystem paths leaking there).
    _source_dir: Optional[Path] = PrivateAttr(default=None)


class SandboxProfile(BaseModel):
    """OS-level isolation policy for a scenario group.

    Scenario authors declare what isolation provider runs the agent
    subprocess (``host`` for today's behaviour or ``docker`` for container
    isolation), which container image to use, and which hosts / paths /
    env-vars cross the sandbox boundary. Agents are sandbox-unaware: this
    entity is consumed entirely by the runner.

    The ``provider`` Literal is a fixed allow-list so a typo is rejected at
    parse time rather than silently downgrading to ``host``. Third-party
    providers register through the ``belt.sandbox_providers`` entry-point
    group; their names are accepted at runtime via the registry rebuild and
    fail validation only when the plugin is missing -- the right failure mode.
    """

    model_config = ConfigDict(extra="forbid")

    provider: Literal["host", "docker"] = Field(
        default="host",
        description=(
            "Sandbox provider name. ``host`` (default) runs the agent on the "
            "host with the invoking user's privileges and no isolation. "
            "``docker`` runs each agent subprocess inside a container with "
            "--cap-drop=ALL, --read-only rootfs, the worktree as the only "
            "writable mount, and env passthrough by exact name. Third-party "
            "providers register via the ``belt.sandbox_providers`` entry-point "
            "group."
        ),
    )
    image: Optional[str] = Field(
        default=None,
        description=(
            "Container image reference (e.g. ``agent-belt-sandbox-cursor:dev``). "
            "Required when ``provider != 'host'``. The framework ships a "
            "reference Dockerfile under ``examples/sandbox-images/`` and a "
            "worked example for cursor; users build their own images from "
            "those (see SANDBOXING.md)."
        ),
    )
    network_policy: Literal["open", "none"] = Field(
        default="open",
        description=(
            "Outbound network policy for the sandbox. ``open`` (default) "
            "leaves the provider's default network in place: on the docker "
            "provider that is the bridge network with full outbound, which "
            "real LLM-using agents need. ``none`` is kernel-enforced zero "
            "network: the docker provider passes ``--network=none`` so the "
            "container has no network interfaces other than loopback and "
            "any outbound socket call fails immediately. Use ``none`` for "
            "scenarios that operate purely on local files (e.g. offline "
            "code-edit tasks against a fixture project) so a misbehaving "
            "agent cannot exfiltrate worktree contents. The runtime gate "
            "lives in the provider layer: a profile that sets "
            "``network_policy='none'`` together with ``provider='host'`` "
            "is rejected at scenario start (HostSandboxProvider has no "
            "isolation layer and cannot honour the policy), so an "
            "unenforceable combination cannot silently downgrade to the "
            "host's open network. Hostname-level allowlisting (``open`` "
            "+ filtering) is tracked as future work."
        ),
    )
    allowed_hosts: list[str] = Field(
        default_factory=list,
        description=(
            "DNS hostnames the agent may reach from inside the sandbox. v1 "
            "best-effort: surfaced as ``--add-host`` entries on the docker "
            "provider; hard outbound enforcement is tracked separately. "
            "Wildcards rejected. Ignored when ``network_policy='none'``. "
            "See SANDBOXING.md -> 'Network policy'."
        ),
    )
    writable_paths: list[str] = Field(
        default_factory=list,
        description=(
            "Extra host paths bind-mounted writable into the container. The "
            "worktree is always mounted; this list adds anything else the "
            "agent legitimately writes (e.g. a shared cache). Each entry "
            "widens the trust boundary -- use sparingly."
        ),
    )
    env_passthrough: list[str] = Field(
        default_factory=list,
        description=(
            "Environment variable names (exact match, no wildcards) forwarded "
            "into the sandbox. Unioned with the agent's "
            "``required_env_vars()``. Values stay in the invoker's env -- "
            "docker is told 'pass NAME' rather than 'pass NAME=value' so "
            "belt itself never echoes a secret."
        ),
    )

    @field_validator("allowed_hosts")
    @classmethod
    def _reject_wildcard_hosts(cls, v: list[str]) -> list[str]:
        bad = [h for h in v if "*" in h or h == ""]
        if bad:
            raise ValueError(
                f"allowed_hosts entries must be concrete hostnames; " f"wildcards and empty entries are rejected: {bad}"
            )
        return v

    @field_validator("env_passthrough")
    @classmethod
    def _reject_wildcard_env(cls, v: list[str]) -> list[str]:
        bad = [n for n in v if "*" in n or "$" in n or n == "" or " " in n]
        if bad:
            raise ValueError(
                f"env_passthrough entries must be exact variable names; "
                f"wildcards, shell expansion, and whitespace are rejected: {bad}"
            )
        return v


class GroupConfig(BaseModel):
    """Configuration for a scenario group (_config.json).

    Uses Pydantic default ``extra='ignore'`` intentionally: plugins add their
    own keys to ``_config.json`` without coordinating with core.
    """

    agent: str = Field(
        description="Agent name (run 'belt agent list' to see available agents)",
    )
    default_tags: list[str] = Field(default_factory=list)
    llm_dimensions: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Custom LLM judge dimensions for this group. Each entry is a DimensionDef dict "
            "(name, description, high, medium, low, evidence_hints) or a string shorthand. "
            "Overrides the default dimensions when present."
        ),
    )
    llm_dimensions_extend_defaults: bool = Field(
        default=False,
        description="When true, llm_dimensions are appended to the default dimensions instead of replacing them.",
    )
    working_dir: Optional[str] = Field(
        default=None,
        description=(
            "Path to a git repository used as the workspace for code-editing scenarios. "
            "Relative paths resolve from the scenario group directory. "
            "When set, each scenario gets an isolated git worktree."
        ),
    )
    workspace_isolation: Literal["git-worktree", "none"] = Field(
        default="git-worktree",
        description=(
            "Isolation strategy. ``git-worktree`` (default) gives each scenario "
            "an isolated worktree of ``working_dir`` so agent edits never touch "
            "the real repo. ``none`` disables per-scenario worktrees and runs "
            "the agent in the harness CWD; the runner refuses ``none`` unless "
            "the user opts in via ``--allow-inplace`` (or "
            "``BELT_ALLOW_INPLACE=1``). Any other string is rejected at parse "
            "time so typos cannot silently disable isolation."
        ),
    )
    workspace_ref: str = Field(
        default="HEAD",
        description="Git ref to reset each worktree to (branch, tag, or commit SHA).",
    )
    fixture_repo: Optional[str] = Field(
        default=None,
        description=(
            "Git URL or local path to clone as the workspace base. When set, the runner clones "
            "this repository once per group into a cached fixture directory and uses that as "
            "the worktree base for every scenario in the group. Bare local paths are resolved "
            "against the process CWD; URL forms (``https://``, ``file://``, ``ssh://``, ``git@host:...``) "
            "pass through unchanged. Mutually exclusive with ``working_dir`` (the orchestrator "
            "rejects groups that set both)."
        ),
    )
    fixture_ref: str = Field(
        default="HEAD",
        description=(
            "Branch, tag, or commit SHA to check out after cloning ``fixture_repo``. "
            "Used as the ref for per-scenario worktrees."
        ),
    )
    resources: list[Resource] = Field(
        default_factory=list,
        description=(
            "Files or archives installed into each scenario's worktree before the agent runs. "
            "Captured per-scenario in ``resource_lock.json`` (``name``, ``kind``, ``source``, "
            "``dest``, ``version``, ``source_sha256``) so reviewers can pin which exact version "
            "of a payload a result corresponds to."
        ),
    )
    sandbox: SandboxProfile = Field(
        default_factory=SandboxProfile,
        description=(
            "OS-level isolation policy for this group. Default is ``host`` (no isolation, "
            "today's behaviour). Set ``sandbox.provider: 'docker'`` plus ``sandbox.image`` "
            "to run each agent subprocess inside a container. The runner-level ``--sandbox`` "
            "flag and ``BELT_SANDBOX_PROVIDER`` env var override this per invocation. "
            "See SANDBOXING.md."
        ),
    )
