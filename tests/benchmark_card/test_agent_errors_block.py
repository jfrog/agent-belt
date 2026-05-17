# (c) JFrog Ltd. (2026)

"""Tests for the benchmark card's ``agent_errors`` block.

Two surfaces are exercised:

1. ``build_card`` faithfully projects the aggregator's ``agent_errors``
   dict into the card's typed :class:`AgentErrorsSummary`.
2. ``render_markdown`` emits the dedicated section iff the card carries
   a non-None ``agent_errors`` block.

The "missing block" case is the historic baseline (no card section,
no markdown) so existing card consumers don't see new noise on clean
runs.
"""

from __future__ import annotations

from pathlib import Path

from belt.benchmark_card import build_card, render_markdown


def _seed_run_meta(run_dir: Path) -> None:
    """Write the minimal ``run_meta.json`` ``build_card`` requires.

    ``build_card`` is best-effort: missing fields fall back to safe
    defaults. The card test only needs the ``run_meta.json`` file to
    exist so the validation path runs.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_meta.json").write_text("{}")


class TestBuildCardAgentErrors:
    def test_block_omitted_when_results_lack_field(self, tmp_path: Path) -> None:
        _seed_run_meta(tmp_path)
        card = build_card(tmp_path, {"total": 1, "passed": 1, "failed": 0})
        assert card.agent_errors is None

    def test_block_populated_from_results_dict(self, tmp_path: Path) -> None:
        _seed_run_meta(tmp_path)
        results = {
            "total": 2,
            "passed": 1,
            "failed": 1,
            "agent_errors": {
                "scenarios_with_errors": 2,
                "scenarios_total": 2,
                "vacuous_passes": 1,
                "by_error_type": {"authentication_failed": 2},
                "remediation": "Re-authenticate the Claude Code CLI: run `claude login`.",
                "per_scenario": [{"scenario": "g/a", "vacuous_pass": True}],
            },
        }
        card = build_card(tmp_path, results)
        ae = card.agent_errors
        assert ae is not None
        assert ae.scenarios_with_errors == 2
        assert ae.vacuous_passes == 1
        assert ae.by_error_type == {"authentication_failed": 2}
        assert "claude login" in ae.remediation

    def test_task_quality_preserves_both_environmental_axes(self, tmp_path: Path) -> None:
        """F6: the agent-axis and judge-axis env counters survive the card projection.

        The aggregator computes ``env_failed_agent`` and
        ``env_failed_judge`` (in addition to the rolled-up
        ``env_failed`` sum) so downstream tooling can attribute env
        failures to the right backend. Earlier card projections only
        copied the sum; this test pins the fix.
        """
        _seed_run_meta(tmp_path)
        results = {
            "total": 5,
            "passed": 2,
            "failed": 3,
            "agent_errors": {
                "scenarios_with_errors": 3,
                "scenarios_total": 5,
                "vacuous_passes": 0,
                "by_error_type": {"authentication_failed": 2, "rate_limited": 1},
                "task_quality": {
                    "attempted": 5,
                    "env_failed": 3,
                    "env_failed_agent": 2,
                    "env_failed_judge": 1,
                    "completed": 2,
                    "passed": 2,
                    "task_failed": 0,
                    "pct": 100.0,
                },
            },
        }
        card = build_card(tmp_path, results)
        tq = card.agent_errors.task_quality
        assert tq is not None
        assert tq.env_failed == 3
        assert tq.env_failed_agent == 2
        assert tq.env_failed_judge == 1
        assert tq.task_failed == 0
        assert tq.completed == 2


class TestBuildCardJudgeErrors:
    """F6: ``judge_errors`` is a sibling block on the card.

    Earlier card builds had no ``judge_errors`` field at all, so a
    judge-rate-limited run looked indistinguishable from a clean run
    on the manifest.
    """

    def test_block_omitted_when_results_lack_field(self, tmp_path: Path) -> None:
        _seed_run_meta(tmp_path)
        card = build_card(tmp_path, {"total": 1, "passed": 1, "failed": 0})
        assert card.judge_errors is None

    def test_block_populated_from_results_dict(self, tmp_path: Path) -> None:
        _seed_run_meta(tmp_path)
        results = {
            "total": 4,
            "passed": 2,
            "failed": 2,
            "judge_errors": {
                "scenarios_with_errors": 2,
                "scenarios_total": 4,
                "by_error_type": {"rate_limited": 2},
                "per_scenario": [
                    {"scenario": "g/a", "error_type": "rate_limited"},
                    {"scenario": "g/b", "error_type": "rate_limited"},
                ],
            },
        }
        card = build_card(tmp_path, results)
        je = card.judge_errors
        assert je is not None
        assert je.scenarios_with_errors == 2
        assert je.scenarios_total == 4
        assert je.by_error_type == {"rate_limited": 2}
        # Per-scenario detail is intentionally dropped from the card.
        assert not hasattr(je, "per_scenario")


class TestRenderMarkdown:
    def _base_card(self, tmp_path: Path):
        _seed_run_meta(tmp_path)
        return build_card(tmp_path, {"total": 0, "passed": 0, "failed": 0})

    def test_section_omitted_when_no_agent_errors(self, tmp_path: Path) -> None:
        md = render_markdown(self._base_card(tmp_path))
        assert "## Agent errors" not in md

    def test_section_included_when_agent_errors_present(self, tmp_path: Path) -> None:
        _seed_run_meta(tmp_path)
        results = {
            "total": 1,
            "passed": 0,
            "failed": 1,
            "agent_errors": {
                "scenarios_with_errors": 1,
                "scenarios_total": 1,
                "vacuous_passes": 0,
                "by_error_type": {"authentication_failed": 1},
                "remediation": "Re-authenticate the Claude Code CLI: run `claude login`.",
            },
        }
        card = build_card(tmp_path, results)
        md = render_markdown(card)
        assert "## Agent errors" in md
        # Markdown sinks escape underscores in cells via ``md_safe``.
        # ``authentication_failed`` therefore appears as
        # ``authentication\_failed`` in the rendered table.
        assert r"authentication\_failed" in md
        # Backticks are escaped inside the remediation block, so match
        # the unescaped substring rather than the literal command.
        assert "claude login" in md or r"claude login" in md.replace("\\", "")

    def test_judge_errors_section_included_when_present(self, tmp_path: Path) -> None:
        _seed_run_meta(tmp_path)
        results = {
            "total": 4,
            "passed": 2,
            "failed": 2,
            "judge_errors": {
                "scenarios_with_errors": 2,
                "scenarios_total": 4,
                "by_error_type": {"rate_limited": 2},
            },
        }
        md = render_markdown(build_card(tmp_path, results))
        assert "## Judge errors" in md
        # 2/4 appears in the kv-table row for "scenarios with errors".
        assert "2/4" in md
        # ``rate_limited`` escaped by md_safe in the by-error-type table.
        assert r"rate\_limited" in md

    def test_judge_errors_section_omitted_when_absent(self, tmp_path: Path) -> None:
        md = render_markdown(self._base_card(tmp_path))
        assert "## Judge errors" not in md

    def test_task_quality_renders_both_environmental_axes(self, tmp_path: Path) -> None:
        _seed_run_meta(tmp_path)
        results = {
            "total": 4,
            "passed": 1,
            "failed": 3,
            "agent_errors": {
                "scenarios_with_errors": 3,
                "scenarios_total": 4,
                "vacuous_passes": 0,
                "by_error_type": {"authentication_failed": 2, "rate_limited": 1},
                "task_quality": {
                    "attempted": 4,
                    "env_failed": 3,
                    "env_failed_agent": 2,
                    "env_failed_judge": 1,
                    "completed": 1,
                    "passed": 1,
                    "task_failed": 0,
                    "pct": 100.0,
                },
            },
        }
        md = render_markdown(build_card(tmp_path, results))
        # md_safe escapes hyphens in label cells (``agent\-axis``); strip the
        # escapes when looking for the human-readable phrasing.
        plain = md.replace("\\-", "-").replace("\\_", "_")
        assert "agent-axis env failures" in plain
        assert "judge-axis env failures" in plain
        # The actual axis counts must also appear in the corresponding rows.
        assert "| `2` |" in md  # env_failed_agent
        assert "| `1` |" in md  # env_failed_judge
