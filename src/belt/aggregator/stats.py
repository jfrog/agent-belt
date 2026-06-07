# (c) JFrog Ltd. (2026)

"""Aggregate statistics, cost/timing, reliability, partial scoring.

Pure read-only computation over a list of ``ScenarioScore`` (and the
``TurnOutput`` files they reference for cost/timing).  No I/O beyond reads.
The renderers (``render_terminal``, ``render_markdown``) consume the dicts
this module returns; the CLI driver writes ``results.json``.

Reliability metrics activate when ``__trial_N`` suffixes are present
(written by the runner under ``--trials N``).
"""

from __future__ import annotations

from pathlib import Path

from belt.agent.error_types import ENVIRONMENTAL_ERROR_TYPES, UNKNOWN, remediation_for
from belt.constants import MAX_TURNS_PER_SCENARIO, TRIAL_SUFFIX_RE, TURN_OUTPUT_TEMPLATE
from belt.entities import ScenarioScore, TurnOutput
from belt.scorer.entities import ALL_VERDICT_TOKENS
from belt.scorer.payloads import RulesPayload, iter_llm_payloads, iter_llm_verdicts

from .thresholds import discover_llm_dimensions

# Histogram is initialised to zeros for every token in
# :data:`belt.scorer.entities.ALL_VERDICT_TOKENS` so consumers see a
# stable shape regardless of which dimension kinds are in play this run.
_VERDICT_BUCKETS: tuple[str, ...] = ALL_VERDICT_TOKENS

INCONCLUSIVE_CEILING_PCT: float = 20.0
"""Per-dimension inconclusive ratio (out of total verdicts for that dim)
above which the aggregator surfaces a warning. Pure heuristic - the
inconclusive verdict is only useful when it stays rare; a dimension
whose judge regularly cannot grade scenarios is more likely to be a
broken rubric than a string of unevidenced runs.
"""

# Re-export so existing aggregator-internal imports keep working.
__all__ = [
    "aggregate_cost_timing",
    "build_bottom_line",
    "build_stats",
    "build_task_quality_split",
    "collect_agent_errors",
    "collect_judge_errors",
    "compute_partial_score",
    "compute_reliability",
    "discover_llm_dimensions",
    "extract_scorer_usage",
    "load_turn_outputs_for_scenario",
]


def build_stats(scores: list[ScenarioScore]) -> dict:
    """Per-mode breakdown indexed by scorer key.

    The single source of truth for per-dimension histograms is
    ``stats["scorers"][<scorer_key>][<dim>]``. ``"rules"`` carries
    the rule-based scorer's per-dimension pass-rates; every LLM
    scorer (``"llm"`` for the consensus/single-judge path; the
    user-declared judge names for multi-judge non-consensus; any
    third-party scorer key) carries the verdict histogram with
    buckets ``high`` / ``medium`` / ``low`` / ``pass`` / ``fail`` /
    ``inconclusive`` / ``total``.

    A scorer key indexes the typed payload contract defined in
    :mod:`belt.scorer.payloads`. Aggregator/exporter consumers walk
    payloads through :func:`iter_llm_payloads` /
    :func:`iter_dimension_feedback`, never by hard-coded scorer
    name lookup.
    """
    stats: dict = {"pass_rate": 0.0, "scorers": {}}
    total = len(scores)
    if total == 0:
        return stats

    passed = sum(1 for s in scores if s.overall_pass)
    stats["pass_rate"] = round(passed / total, 2)

    checks_passed = 0
    checks_total = 0
    rule_dims: dict[str, dict[str, int]] = {}
    for s in scores:
        rules = s.scores.get("rules")
        if not isinstance(rules, RulesPayload):
            continue
        for check in rules.checks:
            dim = check.dimension or "unknown"
            rule_dims.setdefault(dim, {"passed": 0, "failed": 0, "total": 0})
            rule_dims[dim]["total"] += 1
            checks_total += 1
            if check.passed is True:
                rule_dims[dim]["passed"] += 1
                checks_passed += 1
            else:
                rule_dims[dim]["failed"] += 1
    for _dim, counts in rule_dims.items():
        counts["pass_rate"] = round(counts["passed"] / counts["total"], 2) if counts["total"] else 0.0
    if rule_dims:
        stats["scorers"]["rules"] = rule_dims

    if checks_total > 0:
        stats["partial_score"] = round(checks_passed / checks_total, 4)
        stats["checks_passed"] = checks_passed
        stats["checks_total"] = checks_total

    # One histogram per (scorer_key, dim) cell. ``iter_llm_verdicts``
    # applies the per-turn worst-of-turns rollup so each dim
    # contributes one row regardless of resolution.
    per_scorer_dims: dict[str, dict[str, dict[str, int]]] = {}
    for s in scores:
        for scorer_name, payload in iter_llm_payloads(s):
            scorer_block = per_scorer_dims.setdefault(scorer_name, {})
            for dim, score_token, _reasoning in iter_llm_verdicts(payload):
                bucket = scorer_block.setdefault(dim, {b: 0 for b in _VERDICT_BUCKETS} | {"total": 0})
                bucket["total"] += 1
                if score_token in _VERDICT_BUCKETS:
                    bucket[score_token] += 1
    inconclusive_warnings: list[str] = []
    for scorer_name, dim_map in per_scorer_dims.items():
        stats["scorers"][scorer_name] = dim_map
        for dim, counts in dim_map.items():
            dim_total = counts.get("total", 0)
            if dim_total <= 0:
                continue
            inconclusive = counts.get("inconclusive", 0)
            ratio = 100.0 * inconclusive / dim_total
            if ratio > INCONCLUSIVE_CEILING_PCT:
                inconclusive_warnings.append(
                    f"{scorer_name}/{dim}: {inconclusive}/{dim_total} verdicts ({ratio:.0f}%) "
                    f"inconclusive - rubric may be unevidenced or transcript may be too short."
                )
    if inconclusive_warnings:
        stats["llm_inconclusive_warnings"] = inconclusive_warnings
    return stats


def load_turn_outputs_for_scenario(outcome_dir: Path) -> list[TurnOutput]:
    """Load every persisted ``turn_*_output.json`` for one scenario.

    Mirrors the on-disk walk in :func:`aggregate_cost_timing`. Returns an
    ordered list (turn 0 first) and silently skips files that fail to
    parse - the function is best-effort because partial runs may have
    incomplete sidecars.
    """
    outputs: list[TurnOutput] = []
    for i in range(MAX_TURNS_PER_SCENARIO):
        path = outcome_dir / TURN_OUTPUT_TEMPLATE.format(i)
        if not path.exists():
            break
        try:
            outputs.append(TurnOutput.model_validate_json(path.read_text()))
        except Exception:
            continue
    return outputs


def collect_judge_errors(scores: list[ScenarioScore]) -> dict | None:
    """Walk ``scores`` and summarise LLM-judge infrastructure failures.

    A judge infrastructure failure is any scenario whose ``llm`` scorer
    payload has ``judge_errored=True`` - the judge did not produce a
    verdict (rate-limit, timeout, network, parse failure). Mirrors
    :func:`collect_agent_errors` in shape so :func:`build_task_quality_split`
    can partition both axes uniformly and renderers can iterate the two
    blocks with the same template.

    Returns ``None`` when no scenario had a judge failure. The non-None
    shape:

    - ``scenarios_with_errors``: count of judge-errored scenarios.
    - ``scenarios_total``: ``len(scores)``.
    - ``by_error_type``: dict mapping each :data:`JUDGE_ERROR_TYPES` token
      to the number of scenarios marked with it. Sorted by count desc.
    - ``per_scenario``: list of ``{"scenario": "group/name",
      "error_type": "rate_limited", "details": "<message>"}`` rows.
      ``build_task_quality_split`` uses these scenario identifiers to
      partition the run.
    """
    per_scenario: list[dict] = []
    by_error_type: dict[str, int] = {}
    for s in scores:
        # Walk every LLM-shaped payload so a per-judge multi-judge
        # config (judge_a + judge_b under --scorer-config without
        # consensus) still surfaces infra failures, and a per-turn
        # judge writing PerTurnLLMPayload is not silently dropped from
        # this section. A scenario contributes one row even when
        # multiple judges errored; the error_type is the alphabetically
        # first one for stable ordering across runs.
        first_etype: str | None = None
        for _name, payload in iter_llm_payloads(s):
            if not getattr(payload, "judge_errored", False):
                continue
            etype = getattr(payload, "judge_error_type", None) or "other"
            if first_etype is None or etype < first_etype:
                first_etype = etype
        if first_etype is not None:
            by_error_type[first_etype] = by_error_type.get(first_etype, 0) + 1
            per_scenario.append(
                {
                    "scenario": f"{s.group}/{s.scenario_name}",
                    "error_type": first_etype,
                }
            )
    if not per_scenario:
        return None
    return {
        "scenarios_with_errors": len(per_scenario),
        "scenarios_total": len(scores),
        "by_error_type": dict(sorted(by_error_type.items(), key=lambda kv: (-kv[1], kv[0]))),
        "per_scenario": per_scenario,
    }


def collect_agent_errors(
    outcomes_root: Path,
    scores: list[ScenarioScore],
    *,
    agent_name: str | None = None,
    judge_errors: dict | None = None,
) -> dict | None:
    """Walk persisted turn outputs and summarise agent runtime failures.

    A "runtime failure" is any turn with ``TurnOutput.has_error=true``.
    The ``error_type`` field carries the classification token (see
    :mod:`belt.agent.error_types`); turns with ``has_error=true``
    but no token fall back to :data:`UNKNOWN`.

    A "vacuous pass" is a scenario whose ``overall_pass=true`` rules
    score coexists with at least one erroring turn - the rules passed
    while the agent did not really run, so the pass is untrustworthy.

    Returns ``None`` when no scenario had any agent error. The non-None
    shape is a stable dict consumed by :func:`build_bottom_line`,
    :func:`AggregatedResults.agent_errors`, and the benchmark card.

    The returned dict additionally carries a ``task_quality`` block when
    at least one scenario is environment-blocked. The block partitions
    the run into "agent ran cleanly" vs "agent was blocked by the
    environment", giving CI dashboards a defensible headline number
    (``passed/completed``) that is not contaminated by transient
    auth / rate-limit / timeout failures. See
    :func:`build_task_quality_split` for the partition contract.
    """
    per_scenario: list[dict] = []
    by_error_type: dict[str, int] = {}
    scenarios_with_errors = 0
    vacuous_passes = 0

    for s in scores:
        outcome_dir = outcomes_root / s.group / s.scenario_name
        if not outcome_dir.is_dir():
            continue
        outputs = load_turn_outputs_for_scenario(outcome_dir)
        scenario_errors = [o for o in outputs if o.has_error]
        if not scenario_errors:
            continue
        scenarios_with_errors += 1

        scenario_types: list[str] = []
        first_reply: str | None = None
        for o in scenario_errors:
            etype = o.error_type or UNKNOWN
            scenario_types.append(etype)
            by_error_type[etype] = by_error_type.get(etype, 0) + 1
            if first_reply is None and o.reply_text and o.reply_text.strip():
                first_reply = o.reply_text.strip().splitlines()[0]

        if s.overall_pass:
            vacuous_passes += 1

        per_scenario.append(
            {
                "scenario": f"{s.group}/{s.scenario_name}",
                "passed": s.overall_pass,
                "vacuous_pass": s.overall_pass,
                "error_types": scenario_types,
                "first_reply_text": first_reply,
            }
        )

    judge_per_scenario = (judge_errors or {}).get("per_scenario") or []
    if not scenarios_with_errors and not judge_per_scenario:
        return None

    if not scenarios_with_errors:
        # No agent errors but the caller passed judge_errors so the
        # task-quality partition still needs computing. Return a minimal
        # dict that only carries the partition; downstream consumers
        # (bottom-line, renderer) check ``scenarios_with_errors`` before
        # rendering the agent-side block.
        split = build_task_quality_split(scores, [], judge_per_scenario=judge_per_scenario)
        return {"scenarios_with_errors": 0, "task_quality": split} if split else None

    summary: dict = {
        "scenarios_with_errors": scenarios_with_errors,
        "scenarios_total": len(scores),
        "vacuous_passes": vacuous_passes,
        "by_error_type": dict(sorted(by_error_type.items(), key=lambda kv: (-kv[1], kv[0]))),
        "per_scenario": per_scenario,
    }
    split = build_task_quality_split(scores, per_scenario, judge_per_scenario=judge_per_scenario)
    if split is not None:
        summary["task_quality"] = split
    if agent_name:
        # Hint resolution is per-(error_type, agent_name); the headline
        # picks the first non-empty hint so the user sees a remediation
        # tied to the actual agent in use.
        for etype in summary["by_error_type"]:
            hint = remediation_for(etype, agent_name)
            if hint:
                summary["remediation"] = hint
                break
    return summary


def build_task_quality_split(
    scores: list[ScenarioScore],
    per_scenario: list[dict],
    *,
    judge_per_scenario: list[dict] | None = None,
) -> dict | None:
    """Partition scenarios into "agent ran cleanly" vs "environment blocked".

    Returns ``None`` when no scenario had an environmental error on
    either axis, in which case the existing single-axis headline
    ("M/N scenarios failed") already tells the right story.

    Otherwise returns a stable dict:

    - ``attempted``: total scenarios in the run.
    - ``env_failed_agent``: scenarios where the agent itself was
      blocked by infra (one of :data:`ENVIRONMENTAL_ERROR_TYPES`:
      ``authentication_failed`` / ``rate_limited`` / ``timeout``).
    - ``env_failed_judge``: scenarios where the LLM judge backend
      itself failed for infra reasons (``LLMPayload.judge_errored``
      with one of :data:`belt.scorer.entities.JUDGE_ERROR_TYPES`).
      A scenario in both buckets counts once - the agent-axis wins
      because the judge could not have voted if the agent had not run.
    - ``env_failed``: ``env_failed_agent + env_failed_judge`` minus the
      overlap, preserved for back-compat with consumers that pre-date
      the judge axis.
    - ``completed``: ``attempted - env_failed`` -- scenarios where both
      the agent and the judge ran cleanly enough to produce a verdict
      the user can defensibly publish.
    - ``passed``: of ``completed``, how many passed rules.
    - ``task_failed``: ``completed - passed`` -- scenarios where the
      agent ran but did the wrong thing.
    - ``pct``: ``passed / completed`` rounded to one decimal, or
      ``None`` when ``completed == 0``.

    Vacuous passes (rules pass + agent error, or rules pass + judge
    error) are intentionally NOT counted in ``passed``: a scenario whose
    judge or agent was blocked by infra can never be a real pass, even
    if its (typically degenerate) rules happened to evaluate true.
    """
    env_blocked_agent: set[str] = set()
    for entry in per_scenario:
        types: list[str] = entry.get("error_types") or []
        if any(t in ENVIRONMENTAL_ERROR_TYPES for t in types):
            env_blocked_agent.add(entry["scenario"])

    env_blocked_judge: set[str] = set()
    for entry in judge_per_scenario or []:
        env_blocked_judge.add(entry["scenario"])

    # Agent-axis wins on overlap so a scenario whose agent errored AND
    # whose judge would have errored is attributed once, to the upstream
    # cause - the judge could not have voted on an agent that never ran.
    env_blocked = env_blocked_agent | env_blocked_judge
    if not env_blocked:
        return None

    attempted = len(scores)
    env_failed_agent = len(env_blocked_agent)
    env_failed_judge = len(env_blocked_judge - env_blocked_agent)
    env_failed = env_failed_agent + env_failed_judge
    completed = attempted - env_failed
    passed = sum(1 for s in scores if s.overall_pass and f"{s.group}/{s.scenario_name}" not in env_blocked)
    task_failed = max(completed - passed, 0)
    pct = round(100 * passed / completed, 1) if completed > 0 else None
    return {
        "attempted": attempted,
        "env_failed": env_failed,
        "env_failed_agent": env_failed_agent,
        "env_failed_judge": env_failed_judge,
        "completed": completed,
        "passed": passed,
        "task_failed": task_failed,
        "pct": pct,
    }


def build_bottom_line(
    scores: list[ScenarioScore],
    *,
    agent_errors: dict | None = None,
    judge_errors: dict | None = None,
) -> list[str]:
    """One- or several-line headline summary (used in ``results.json`` and console).

    When ``agent_errors`` is provided (typically from
    :func:`collect_agent_errors`), the agent-error headline is emitted
    *before* any rule-failure summary - "the agent didn't really run"
    must dominate "the agent ran and answered wrong". Vacuous passes
    (rules pass + at least one turn errored) get an explicit warning.

    When the run had at least one environment-blocked scenario (the
    ``task_quality`` block populated by :func:`build_task_quality_split`
    is present), the lead line is replaced by an explicit task-quality
    vs environmental-health split so CI dashboards do not conflate
    "agent did the wrong thing" with "OpenAI judge timed out".
    """
    lines: list[str] = []
    total = len(scores)
    failed = [s for s in scores if not s.overall_pass]

    if not failed and not agent_errors and not judge_errors:
        lines.append(f"All {total} scenarios passed.")
        return lines

    split = (agent_errors or {}).get("task_quality")
    if isinstance(split, dict) and split.get("env_failed", 0) > 0:
        # Environmental-blocked scenarios are present: lead with the
        # three-axis split headline. The single-axis "M/N scenarios
        # failed" line is dropped because the split tells that story
        # already, with the right denominator. The agent and judge
        # buckets are listed separately so a reader can attribute the
        # infra failure to the right side of the wall.
        passed = int(split.get("passed", 0))
        completed = int(split.get("completed", 0))
        env_failed_agent = int(split.get("env_failed_agent", split.get("env_failed", 0)))
        env_failed_judge = int(split.get("env_failed_judge", 0))
        task_failed = int(split.get("task_failed", 0))
        pct = split.get("pct")
        pct_str = f"{pct}%" if pct is not None else "N/A"
        task_word = "failure" if task_failed == 1 else "failures"
        env_parts: list[str] = []
        if env_failed_agent:
            word = "failure" if env_failed_agent == 1 else "failures"
            env_parts.append(f"{env_failed_agent} agent env {word}")
        if env_failed_judge:
            word = "failure" if env_failed_judge == 1 else "failures"
            env_parts.append(f"{env_failed_judge} judge env {word}")
        env_str = " - ".join(env_parts) if env_parts else ""
        head = f"{passed}/{completed} task quality ({pct_str})"
        if env_str:
            head += f" - {env_str}"
        head += f" - {task_failed} agent task {task_word}"
        lines.append(head)
    elif failed:
        lines.append(f"{len(failed)}/{total} scenarios failed.")
    else:
        # All scenarios technically passed but at least one had an
        # agent error - flag it instead of the misleading
        # "All N scenarios passed." headline.
        lines.append(f"All {total} scenarios passed (rules), but agent errors were detected.")

    if judge_errors and judge_errors.get("scenarios_with_errors", 0) > 0:
        n_err = judge_errors["scenarios_with_errors"]
        types_str = ", ".join(
            etype if count == 1 else f"{etype} ({count})" for etype, count in judge_errors["by_error_type"].items()
        )
        lines.append(
            f"LLM judge infrastructure failure in {n_err}/{total} scenario(s): {types_str}. "
            f"Re-run after the judge backend recovers; do not treat passing rules as a green."
        )

    if agent_errors and agent_errors.get("scenarios_with_errors", 0) > 0:
        n_err = agent_errors["scenarios_with_errors"]
        types_str = ", ".join(
            f"{etype}" if count == 1 else f"{etype} ({count})" for etype, count in agent_errors["by_error_type"].items()
        )
        line = f"Agent error in {n_err}/{total} scenario(s): {types_str}."
        hint = agent_errors.get("remediation")
        if hint:
            line += f" {hint}"
        lines.append(line)

        n_vacuous = agent_errors["vacuous_passes"]
        if n_vacuous:
            noun = "scenario" if n_vacuous == 1 else "scenarios"
            lines.append(
                f"WARNING: {n_vacuous} passing {noun} contained agent errors - "
                f"rules may have passed vacuously. Re-run after resolving the agent error."
            )

    rule_failure_counts: dict[str, int] = {}
    for s in failed:
        rules = s.scores.get("rules")
        if not isinstance(rules, RulesPayload):
            continue
        for c in rules.checks:
            if c.passed is False:
                key = f"{c.dimension}/{c.check}"
                rule_failure_counts[key] = rule_failure_counts.get(key, 0) + 1

    if rule_failure_counts:
        top_rule = max(rule_failure_counts, key=rule_failure_counts.get)
        lines.append(f"Most common rule failure: {top_rule} ({rule_failure_counts[top_rule]}x)")

    # Walk every LLM-shaped payload so per-judge multi-judge and
    # per-turn (PerTurnLLMPayload, rolled up via iter_llm_verdicts) low
    # / fail / inconclusive verdicts all contribute. ``reasoning`` for
    # per-turn entries already carries the ``[turn N]`` prefix from the
    # rollup so the operator sees which turn dragged the dim down.
    llm_low_dims: dict[str, list[str]] = {}
    llm_fail_dims: dict[str, list[str]] = {}
    llm_inconclusive_dims: dict[str, list[str]] = {}
    for s in failed:
        for _name, payload in iter_llm_payloads(s):
            for dim, score_token, reasoning in iter_llm_verdicts(payload):
                if score_token == "low":
                    llm_low_dims.setdefault(dim, []).append(reasoning)
                elif score_token == "fail":
                    llm_fail_dims.setdefault(dim, []).append(reasoning)
                elif score_token == "inconclusive":
                    llm_inconclusive_dims.setdefault(dim, []).append(reasoning)

    for dim, reasons in llm_low_dims.items():
        lines.append(f"LLM scored {dim} as low in {len(reasons)} scenario(s): {reasons[0][:120]}")
    for dim, reasons in llm_fail_dims.items():
        lines.append(f"LLM scored {dim} as fail in {len(reasons)} scenario(s): {reasons[0][:120]}")
    for dim, reasons in llm_inconclusive_dims.items():
        lines.append(f"LLM marked {dim} inconclusive in {len(reasons)} scenario(s): {reasons[0][:120]}")

    return lines


def _safe_int(val: object) -> int:
    """Coerce a value to int, returning 0 on any failure."""
    if val is None:
        return 0
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def _extract_usage_from_dict(usage: dict) -> tuple[int, int]:
    """Extract (prompt, completion) tokens from a usage dict.

    Provider-agnostic: tries both OpenAI (``prompt_tokens``) and
    Anthropic (``input_tokens``) naming conventions.
    """
    prompt = _safe_int(usage.get("prompt_tokens") or usage.get("input_tokens"))
    completion = _safe_int(usage.get("completion_tokens") or usage.get("output_tokens"))
    return prompt, completion


def extract_scorer_usage(score: ScenarioScore) -> dict[str, int] | None:
    """Extract scorer token usage from a scenario's score data.

    Walks all scorer payloads with a ``usage`` field and sums them.
    Consensus payloads already aggregate sub-judge usage into their
    top-level ``usage``, so returning that is correct (no double-count
    via ``individual_verdicts``).
    """
    prompt = 0
    completion = 0
    found = False

    for _key, payload in score.scores.items():
        if isinstance(payload, RulesPayload):
            continue
        usage = getattr(payload, "usage", None)
        if usage is None:
            continue
        usage_dict = usage.model_dump(mode="json", exclude_none=True) if hasattr(usage, "model_dump") else dict(usage)
        p, c = _extract_usage_from_dict(usage_dict)
        if p > 0 or c > 0:
            prompt += p
            completion += c
            found = True

    if not found:
        return None
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


def aggregate_cost_timing(root: Path, scores: list[ScenarioScore]) -> dict:
    """Aggregate cost and timing from turn output files alongside scored scenarios."""
    per_scenario: list[dict] = []
    total_cost = 0.0
    total_seconds = 0.0
    cost_count = 0
    timing_count = 0
    total_scorer_prompt = 0
    total_scorer_completion = 0
    scorer_count = 0
    total_judge_cost = 0.0
    judge_cost_count = 0

    for s in scores:
        outcome_dir = root / s.group / s.scenario_name
        sc_cost = 0.0
        sc_seconds = 0.0
        has_cost = False
        has_timing = False

        for i in range(MAX_TURNS_PER_SCENARIO):
            output_path = outcome_dir / TURN_OUTPUT_TEMPLATE.format(i)
            if not output_path.exists():
                break
            try:
                to = TurnOutput.model_validate_json(output_path.read_text())
            except Exception:
                continue
            if to.cost_usd is not None:
                sc_cost += to.cost_usd
                has_cost = True
            if to.timing and to.timing.total is not None:
                sc_seconds += to.timing.total
                has_timing = True

        entry: dict = {"scenario": f"{s.group}/{s.scenario_name}", "passed": s.overall_pass}
        if has_cost:
            entry["agent_cost_usd"] = round(sc_cost, 6)
            total_cost += sc_cost
            cost_count += 1
        if has_timing:
            entry["total_seconds"] = round(sc_seconds, 2)
            total_seconds += sc_seconds
            timing_count += 1

        scorer_usage = extract_scorer_usage(s)
        if scorer_usage:
            entry["scorer_tokens"] = scorer_usage
            total_scorer_prompt += scorer_usage["prompt_tokens"]
            total_scorer_completion += scorer_usage["completion_tokens"]
            scorer_count += 1

        if s.judge_cost_usd is not None:
            entry["judge_cost_usd"] = round(s.judge_cost_usd, 6)
            total_judge_cost += s.judge_cost_usd
            judge_cost_count += 1

        per_scenario.append(entry)

    result: dict = {"scenarios": per_scenario}
    if cost_count:
        result["agent_cost_usd"] = round(total_cost, 6)
        result["total_cost_usd"] = round(total_cost, 6)
        result["mean_cost_usd"] = round(total_cost / cost_count, 6)
    if judge_cost_count:
        result["judge_cost_usd"] = round(total_judge_cost, 6)
        combined = total_cost + total_judge_cost
        result["total_cost_usd"] = round(combined, 6)
    if timing_count:
        result["total_seconds"] = round(total_seconds, 2)
        result["mean_seconds"] = round(total_seconds / timing_count, 2)
    if scorer_count:
        total_scorer = total_scorer_prompt + total_scorer_completion
        result["scorer_tokens"] = {
            "prompt_tokens": total_scorer_prompt,
            "completion_tokens": total_scorer_completion,
            "total_tokens": total_scorer,
            "scenarios_counted": scorer_count,
        }
    return result


def compute_partial_score(scores: list[ScenarioScore]) -> dict | None:
    """Fraction of individual rule checks that passed across all scenarios."""
    checks_passed = 0
    checks_total = 0
    for s in scores:
        rules = s.scores.get("rules")
        if not isinstance(rules, RulesPayload):
            continue
        for c in rules.checks:
            checks_total += 1
            if c.passed is True:
                checks_passed += 1
    if checks_total == 0:
        return None
    return {
        "checks_passed": checks_passed,
        "checks_total": checks_total,
        "partial_score": checks_passed / checks_total,
    }


def compute_reliability(scores: list[ScenarioScore]) -> dict | None:
    """Reliability metrics from ``__trial_N`` outcome dirs.

    Groups scores by base scenario name (stripping the ``__trial_N`` suffix
    via ``constants.TRIAL_SUFFIX_RE``), computes empirical pass rate p, and
    reports for k=1, 3, 8:

    - ``pass_at_k = 1 - (1-p)^k`` -- at least one of k trials passes.
    - ``pass_pow_k = p^k`` -- all k trials pass.

    Returns ``None`` when no trials were detected (single-trial runs).
    """
    pat = TRIAL_SUFFIX_RE
    trial_groups: dict[str, list[bool]] = {}
    has_trials = False

    for s in scores:
        m = pat.search(s.scenario_name)
        if m:
            has_trials = True
            base = pat.sub("", s.scenario_name)
            key = f"{s.group}/{base}"
        else:
            key = f"{s.group}/{s.scenario_name}"
        trial_groups.setdefault(key, []).append(s.overall_pass)

    if not has_trials:
        return None

    scenarios: list[dict] = []
    total_p = 0.0
    for key, results in sorted(trial_groups.items()):
        n = len(results)
        passed = sum(results)
        p = passed / n if n else 0.0
        scenarios.append(
            {
                "scenario": key,
                "trials": n,
                "passed": passed,
                "pass_rate": round(p, 4),
                "pass_at_1": round(p, 4),
                "pass_at_3": round(1.0 - (1.0 - p) ** 3, 4),
                "pass_at_8": round(1.0 - (1.0 - p) ** 8, 4),
                "pass_pow_1": round(p, 4),
                "pass_pow_3": round(p**3, 4),
                "pass_pow_8": round(p**8, 4),
            }
        )
        total_p += p

    n_scenarios = len(scenarios)
    mean_p = total_p / n_scenarios if n_scenarios else 0.0

    return {
        "mean_pass_rate": round(mean_p, 4),
        "mean_pass_at_1": round(mean_p, 4),
        "mean_pass_at_3": round(1.0 - (1.0 - mean_p) ** 3, 4),
        "mean_pass_at_8": round(1.0 - (1.0 - mean_p) ** 8, 4),
        "mean_pass_pow_1": round(mean_p, 4),
        "mean_pass_pow_3": round(mean_p**3, 4),
        "mean_pass_pow_8": round(mean_p**8, 4),
        "scenarios": scenarios,
    }
