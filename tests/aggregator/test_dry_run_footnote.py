# (c) JFrog Ltd. (2026)

"""Tests for the ``dry-run-only`` failure footnote in terminal and markdown
renderers. The bare showcase invocation produces failures by design (schema
fields no generic CLI agent surfaces); the renderers add a single footnote
when at least one failed scenario carries the ``dry-run-only`` tag, pointing
the user at ``--tags real-runnable``. This file pins the wording invariant.
"""

from __future__ import annotations

import io

from rich.console import Console

from belt.aggregator import dry_run_only_failure_count
from belt.aggregator.render_markdown import build_markdown
from belt.aggregator.render_terminal import print_terminal
from belt.entities import ScenarioScore


def _score(group: str, name: str, *, passed: bool, tags: list[str] | None = None) -> ScenarioScore:
    return ScenarioScore(
        scenario_name=name,
        group=group,
        tags=tags or [],
        scores={},
        overall_pass=passed,
    )


class TestDryRunOnlyFailureCount:
    def test_counts_only_failed_dry_run_only_scenarios(self) -> None:
        scores = [
            _score("g", "passed_dry", passed=True, tags=["dry-run-only"]),
            _score("g", "failed_dry", passed=False, tags=["dry-run-only", "showcase"]),
            _score("g", "failed_real", passed=False, tags=["real-runnable"]),
            _score("g", "passed_real", passed=True, tags=["real-runnable"]),
        ]
        assert dry_run_only_failure_count(scores) == 1

    def test_zero_when_no_dry_run_failures(self) -> None:
        scores = [
            _score("g", "a", passed=True, tags=["dry-run-only"]),
            _score("g", "b", passed=False, tags=["real-runnable"]),
        ]
        assert dry_run_only_failure_count(scores) == 0

    def test_zero_when_tags_field_empty(self) -> None:
        # Older scorer outputs without tags must not crash the renderer.
        scores = [_score("g", "a", passed=False, tags=[])]
        assert dry_run_only_failure_count(scores) == 0


class TestTerminalFootnote:
    def _render(self, scores: list[ScenarioScore]) -> str:
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, color_system=None, width=200)
        print_terminal(scores, run_label="run-id", console=console)
        return buf.getvalue()

    def test_footnote_appears_when_dry_run_only_failure_present(self) -> None:
        scores = [
            _score("agent-capabilities", "skills_invoked_dry_run", passed=False, tags=["dry-run-only"]),
            _score("correctness", "correctness_basic", passed=True, tags=["real-runnable"]),
        ]
        out = self._render(scores)
        assert "dry-run-only" in out
        assert "--tags real-runnable" in out

    def test_footnote_pluralisation(self) -> None:
        scores = [
            _score("a", "x_dry_run", passed=False, tags=["dry-run-only"]),
            _score("a", "y_dry_run", passed=False, tags=["dry-run-only"]),
        ]
        out = self._render(scores)
        # 2 failures → "2 ... scenarios" (plural). The phrasing was
        # compacted in the eval-failure-UX work to "N dry-run-only
        # scenarios. Re-run with --tags real-runnable to skip." -- the
        # singular/plural pivot is on the noun "scenario(s)".
        assert "2 dry-run-only" in out
        assert "scenarios" in out

    def test_singular_pluralisation(self) -> None:
        scores = [_score("a", "x_dry_run", passed=False, tags=["dry-run-only"])]
        out = self._render(scores)
        assert "1 dry-run-only" in out
        # Singular form: "1 ... scenario." (no trailing "s")
        assert " scenario." in out
        assert " scenarios." not in out

    def test_footnote_absent_when_no_dry_run_failures(self) -> None:
        scores = [
            _score("g", "a", passed=False, tags=["real-runnable"]),
            _score("g", "b", passed=True, tags=["dry-run-only"]),
        ]
        out = self._render(scores)
        # Free-standing "dry-run-only" sentence must not appear; only the
        # footnote uses it. Real failures keep the existing format.
        assert "Note:" not in out or "dry-run-only" not in out


class TestMarkdownFootnote:
    def test_footnote_appears_when_dry_run_only_failure_present(self) -> None:
        scores = [
            _score("agent-capabilities", "skills_invoked_dry_run", passed=False, tags=["dry-run-only"]),
            _score("correctness", "correctness_basic", passed=True, tags=["real-runnable"]),
        ]
        md = build_markdown(scores)
        assert "`dry-run-only`" in md
        assert "`--tags real-runnable`" in md

    def test_footnote_absent_when_no_failures(self) -> None:
        scores = [_score("g", "a", passed=True, tags=["dry-run-only"])]
        md = build_markdown(scores)
        # No "Failures" section → no footnote either.
        assert "real-runnable" not in md
