# (c) JFrog Ltd. (2026)

"""JUnit XML exporter - canonical CI test-report contract.

Layout:

* One ``<testsuites>`` root.
* One ``<testsuite>`` per scenario group, with ``tests`` / ``failures`` /
  ``errors`` counters that match the GitHub / GitLab / Jenkins / Buildkite
  test-report integrations all keying off the same XML schema.
* One ``<testcase>`` per base scenario (``--trials N`` collapses to one
  testcase; per-trial pass/fail surfaces as ``<property>`` entries on the
  testcase plus an aggregate ``pass^k`` summary, mirroring the convention
  used by JUnit-aware retry plugins).
* Failures - rule check that did not pass, or LLM judgement at low/medium
  level - render inside ``<failure>`` with a ``message`` attribute that is
  the one-line headline; the body is the multi-line breakdown.
* Agent execution errors (``has_error=True``) render as ``<error>`` instead
  of ``<failure>`` so reporters can distinguish bug-from-the-agent from
  scenario-correctness regressions.

Output bodies pass through :func:`belt._safe.xml_safe` (the centralised
XML text escaper); ``stdout`` / ``stderr`` capture is opt-in
(``include_stdout: true``) so the default report stays small (CI uploaders
cap individual file sizes).

Options:
    suite_name: top-level ``name`` attribute for matrix builds
        (default: the run directory's basename).
    include_stdout: when truthy, also embed each scenario's
        ``turn_*_cli.txt`` head as ``<system-err>`` (default: ``False``).
    max_body_bytes: cap on ``<failure>`` / ``<error>`` body bytes
        (default: ``16384``). The capped tail gets a visible truncation
        marker so the operator knows the run dir holds the full picture.

Stdlib-only: no optional dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.sax.saxutils import quoteattr

from belt._safe import xml_safe
from belt.constants import TURN_CLI_TEMPLATE
from belt.entities import ScenarioScore
from belt.exporter.base import BaseExporter
from belt.exporter.entities import ExportContext
from belt.exporter.helpers import collapse_trials, get_bool_option, get_int_option, truncate_with_marker
from belt.scorer.entities import DEFAULT_FAIL_LEVELS
from belt.scorer.payloads import PerTurnLLMPayload, RulesPayload, iter_llm_payloads, iter_llm_verdicts

_DEFAULT_MAX_BODY_BYTES = 16 * 1024
_HEAD_CLI_BYTES = 4 * 1024


def _scenario_seconds(ctx: ExportContext, key: str) -> float:
    for entry in ctx.results.cost_timing.get("scenarios", []) or []:
        if entry.get("scenario") == key:
            seconds = entry.get("total_seconds")
            try:
                return float(seconds) if seconds is not None else 0.0
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _agent_cli_head(ctx: ExportContext, group: str, scenario: str, max_bytes: int) -> str:
    """Read the head of ``turn_0_cli.txt`` for a scenario, truncated.

    Best-effort: if the run dir has been pruned, the file is missing, or
    decoding fails, return ``""`` so the exporter still emits a valid
    ``<testcase>`` rather than aborting.
    """
    path = ctx.run_dir / group / scenario / TURN_CLI_TEMPLATE.format(0)
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    head = raw[:max_bytes]
    try:
        return head.decode("utf-8", errors="replace")
    except Exception:
        return ""


class JUnitExporter(BaseExporter):
    """JUnit XML report - one ``<testcase>`` per base scenario."""

    @property
    def name(self) -> str:
        return "junit"

    def export(self, ctx: ExportContext, output: Path, options: dict[str, Any]) -> None:
        suite_name = options.get("suite_name") or ctx.run_dir.name
        include_stdout = get_bool_option(options, "include_stdout", False)
        max_body_bytes = get_int_option(options, "max_body_bytes", _DEFAULT_MAX_BODY_BYTES)

        groups = collapse_trials(ctx.scores)
        suites: dict[str, list[tuple[str, list[ScenarioScore]]]] = {}
        for key, trial_scores in groups.items():
            group, scenario = key.split("/", 1)
            suites.setdefault(group, []).append((scenario, trial_scores))

        lines: list[str] = ['<?xml version="1.0" encoding="UTF-8"?>']
        total_tests = sum(len(v) for v in suites.values())
        total_failures = 0
        total_errors = 0
        suite_blocks: list[str] = []

        for group, cases in sorted(suites.items()):
            suite_failures = 0
            suite_errors = 0
            suite_lines: list[str] = []
            suite_seconds = 0.0
            for scenario, trial_scores in cases:
                seconds = _scenario_seconds(ctx, f"{group}/{trial_scores[0].scenario_name}")
                # If trials expanded, take the longest trial as the testcase
                # duration so reporters get a representative number.
                if len(trial_scores) > 1:
                    seconds = max(
                        (_scenario_seconds(ctx, f"{group}/{s.scenario_name}") for s in trial_scores),
                        default=seconds,
                    )
                suite_seconds += seconds
                case_lines, case_status = self._render_case(
                    ctx,
                    group,
                    scenario,
                    trial_scores,
                    seconds=seconds,
                    include_stdout=include_stdout,
                    max_body_bytes=max_body_bytes,
                )
                if case_status == "failure":
                    suite_failures += 1
                elif case_status == "error":
                    suite_errors += 1
                suite_lines.extend(case_lines)
            total_failures += suite_failures
            total_errors += suite_errors
            suite_blocks.append(
                f"  <testsuite name={quoteattr(group)} "
                f'tests="{len(cases)}" '
                f'failures="{suite_failures}" '
                f'errors="{suite_errors}" '
                f'time="{suite_seconds:.3f}">'
            )
            suite_blocks.extend(suite_lines)
            suite_blocks.append("  </testsuite>")

        lines.append(
            f"<testsuites "
            f"name={quoteattr(suite_name)} "
            f'tests="{total_tests}" '
            f'failures="{total_failures}" '
            f'errors="{total_errors}">'
        )
        lines.extend(suite_blocks)
        lines.append("</testsuites>")

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _render_case(
        self,
        ctx: ExportContext,
        group: str,
        scenario: str,
        trial_scores: list[ScenarioScore],
        *,
        seconds: float,
        include_stdout: bool,
        max_body_bytes: int,
    ) -> tuple[list[str], str]:
        all_passed = all(s.overall_pass for s in trial_scores)
        any_agent_error = any(
            isinstance(s.scores.get("rules"), RulesPayload) and bool(s.scores["rules"].has_error) for s in trial_scores
        )
        status = "pass"
        if not all_passed:
            status = "error" if any_agent_error else "failure"

        case_open = (
            f"    <testcase classname={quoteattr(group)} " f"name={quoteattr(scenario)} " f'time="{seconds:.3f}">'
        )
        body: list[str] = [case_open]

        # Property block: trial telemetry (always emitted when trials > 1).
        if len(trial_scores) > 1:
            body.append("      <properties>")
            body.append(f'        <property name="trials" value="{len(trial_scores)}"/>')
            passed_count = sum(1 for s in trial_scores if s.overall_pass)
            body.append(f'        <property name="trials_passed" value="{passed_count}"/>')
            p = passed_count / len(trial_scores) if trial_scores else 0.0
            body.append(f'        <property name="pass_at_1" value="{p:.4f}"/>')
            for k in (3, 8):
                body.append(f'        <property name="pass_at_{k}" value="{1.0 - (1.0 - p) ** k:.4f}"/>')
                body.append(f'        <property name="pass_pow_{k}" value="{p**k:.4f}"/>')
            body.append("      </properties>")

        if status != "pass":
            element = "error" if status == "error" else "failure"
            headline, detail = self._failure_payload(trial_scores)
            capped = truncate_with_marker(detail, max_body_bytes)
            body.append(
                f"      <{element} "
                f"message={quoteattr(headline)} "
                f'type="belt.{element}">'
                f"{xml_safe(capped)}"
                f"</{element}>"
            )

        if include_stdout:
            head = _agent_cli_head(ctx, group, trial_scores[0].scenario_name, _HEAD_CLI_BYTES)
            if head:
                body.append(
                    "      <system-err>" f"{xml_safe(truncate_with_marker(head, max_body_bytes))}" "</system-err>"
                )

        body.append("    </testcase>")
        return body, status

    def _failure_payload(self, trial_scores: list[ScenarioScore]) -> tuple[str, str]:
        """Return ``(headline, body)`` describing the failure across trials."""
        rule_lines: list[str] = []
        llm_lines: list[str] = []
        for s in trial_scores:
            rules = s.scores.get("rules")
            if isinstance(rules, RulesPayload):
                for c in rules.checks:
                    if c.passed is False:
                        suffix = f" - {c.details}" if c.details else ""
                        rule_lines.append(f"  rules/{c.dimension}/{c.check}{suffix}")
            # Walk every LLM-shaped payload so multi-judge and per-turn
            # failing verdicts both surface in the JUnit failure body.
            # Per-judge prefix keeps the lines attributable when more
            # than one judge ran.
            for name, payload in iter_llm_payloads(s):
                prefix = "llm" if name == "llm" else f"llm[{name}]"
                for dim, score_token, reasoning in iter_llm_verdicts(payload):
                    if score_token not in DEFAULT_FAIL_LEVELS:
                        continue
                    snippet = reasoning if len(reasoning) <= 240 else reasoning[:240] + "..."
                    llm_lines.append(f"  {prefix}/{dim} ({score_token}): {snippet}")
                # Nested per-turn detail so a reviewer reading the JUnit
                # body in CI can see which specific turn caused the
                # rolled-up dimension to fail, not just the worst-of-
                # turns headline. Already-bounded by ``max_body_bytes``
                # caller truncation.
                if isinstance(payload, PerTurnLLMPayload):
                    for tv in payload.turns:
                        if tv.judge_errored:
                            etype = tv.judge_error_type or "other"
                            llm_lines.append(f"    [turn {tv.turn_idx}] judge errored ({etype})")
                            continue
                        for dim, vd in tv.dimensions.items():
                            if vd.score not in DEFAULT_FAIL_LEVELS:
                                continue
                            snippet = vd.reasoning if len(vd.reasoning) <= 200 else vd.reasoning[:200] + "..."
                            llm_lines.append(f"    [turn {tv.turn_idx}] {dim}={vd.score}: {snippet}")

        headline_parts: list[str] = []
        if rule_lines:
            headline_parts.append(f"{len(rule_lines)} rule check(s) failed")
        if llm_lines:
            headline_parts.append(f"{len(llm_lines)} LLM judgement(s) below pass")
        if not headline_parts:
            headline_parts.append("scenario failed (no per-check breakdown)")
        headline = "; ".join(headline_parts)

        body_lines: list[str] = []
        if rule_lines:
            body_lines.append("Failed rule checks:")
            body_lines.extend(rule_lines)
        if llm_lines:
            if body_lines:
                body_lines.append("")
            body_lines.append("Failing LLM judgements:")
            body_lines.extend(llm_lines)
        body = "\n".join(body_lines) if body_lines else headline
        return headline, body
