# (c) JFrog Ltd. (2026)

"""Per-turn LLM judging integration into exporters.

Pinned guarantees:

1. **CSV sidecar** - when any scenario carries a
   :class:`PerTurnLLMPayload`, the CSV exporter writes
   ``<output>.per_turn<ext>`` alongside the main file with one row
   per ``(scenario, scorer_key, turn_idx, dimension)``.
2. **CSV sidecar gate** - sidecar is NOT emitted when no per-turn
   payload exists (don't clutter the run dir for pure scenario-level
   evals).
3. **CSV escaping** - every per-turn cell flows through
   :func:`belt._safe.csv_safe`, so a formula-prefix reasoning
   (``=cmd|'/c calc'!A1``) gets the OWASP single-quote prefix.
4. **JUnit failure body** - per-turn judgements appear as
   ``[turn N] <dim>=<score>`` lines inside the ``<failure>`` body
   for failing scenarios, so a reviewer reading the CI report can
   attribute the worst-of-turns headline to the specific turn that
   caused it.
5. **Markdown per-turn block** - the markdown exporter emits a
   ``<details>`` block with one bullet per turn under each per-turn
   judge section, fenced by ``md_safe`` to avoid markdown injection.
"""

from __future__ import annotations

from pathlib import Path

from belt.entities import AggregatedResults, ScenarioScore
from belt.exporter.csv import CsvExporter
from belt.exporter.entities import ExportContext
from belt.exporter.junit import JUnitExporter
from belt.exporter.markdown import MarkdownExporter
from belt.scorer.payloads import LLMDimensionVerdict, LLMPayload, PerTurnLLMPayload, TurnVerdict


def _per_turn_payload(turn_scores: list[str]) -> PerTurnLLMPayload:
    """Build a ``PerTurnLLMPayload`` whose turns have ``correctness=<score>``."""
    turns = [
        TurnVerdict(
            turn_idx=i,
            dimensions={"correctness": LLMDimensionVerdict(score=s, reasoning=f"reason turn {i}")},
        )
        for i, s in enumerate(turn_scores)
    ]
    overall = all(s == "high" for s in turn_scores)
    return PerTurnLLMPayload(overall_pass=overall, turns=turns)


def _scenario(name: str, turn_scores: list[str], *, scorer_key: str = "per_turn_judge") -> ScenarioScore:
    p = _per_turn_payload(turn_scores)
    return ScenarioScore(
        scenario_name=name,
        group="g",
        overall_pass=p.overall_pass,
        scores={scorer_key: p},
    )


def _aggregated(scores: list[ScenarioScore]) -> AggregatedResults:
    return AggregatedResults(
        total=len(scores),
        passed=sum(1 for s in scores if s.overall_pass),
        failed=sum(1 for s in scores if not s.overall_pass),
        bottom_line=[],
        stats={},
        cost_timing={"scenarios": []},
        reliability={},
        thresholds_passed=None,
        thresholds=[],
        thresholds_failed=[],
    )


def _ctx(tmp_path: Path, scores: list[ScenarioScore]) -> ExportContext:
    return ExportContext(
        scores=scores,
        results=_aggregated(scores),
        run_dir=tmp_path,
        scorer_config=None,
    )


# ── CSV sidecar ──


class TestCsvSidecar:
    def test_sidecar_emitted_when_per_turn_payload_present(self, tmp_path: Path) -> None:
        scores = [_scenario("a", ["high", "low"])]
        out = tmp_path / "results.csv"
        CsvExporter().export(_ctx(tmp_path, scores), out, options={})

        sidecar = tmp_path / "results.per_turn.csv"
        assert sidecar.exists(), "Per-turn sidecar missing"
        body = sidecar.read_text(encoding="utf-8")
        # Header row.
        assert body.startswith("group,scenario,scorer_key,turn_idx,dimension,score,reasoning,judge_errored\n")
        # One row per (turn, dim) - 2 turns × 1 dim = 2 data rows.
        assert body.count("\n") == 3  # header + 2 data rows
        assert "per_turn_judge,0,correctness,high" in body
        assert "per_turn_judge,1,correctness,low" in body

    def test_sidecar_not_emitted_when_only_scenario_level_llm(self, tmp_path: Path) -> None:
        """Pure scenario-level LLM run must NOT produce a sidecar."""
        score = ScenarioScore(
            scenario_name="a",
            group="g",
            overall_pass=True,
            scores={
                "llm": LLMPayload(
                    overall_pass=True,
                    dimensions={"correctness": LLMDimensionVerdict(score="high", reasoning="ok")},
                )
            },
        )
        out = tmp_path / "results.csv"
        CsvExporter().export(_ctx(tmp_path, [score]), out, options={})
        sidecar = tmp_path / "results.per_turn.csv"
        assert not sidecar.exists(), "Sidecar emitted for non-per-turn run"

    def test_sidecar_csv_safe_neutralises_formula_prefix(self, tmp_path: Path) -> None:
        # Reasoning that begins with ``=`` would otherwise execute in
        # spreadsheet apps. csv_safe prefixes with a single quote.
        p = PerTurnLLMPayload(
            overall_pass=False,
            turns=[
                TurnVerdict(
                    turn_idx=0,
                    dimensions={"correctness": LLMDimensionVerdict(score="low", reasoning="=cmd|'/c calc'!A1")},
                )
            ],
        )
        score = ScenarioScore(
            scenario_name="a",
            group="g",
            overall_pass=False,
            scores={"per_turn_judge": p},
        )
        out = tmp_path / "results.csv"
        CsvExporter().export(_ctx(tmp_path, [score]), out, options={})
        body = (tmp_path / "results.per_turn.csv").read_text(encoding="utf-8")
        # csv_safe prepends a single-quote to formula-prefix cells.
        assert "'=cmd" in body or "\"'=cmd" in body

    def test_sidecar_extension_preserved_for_tsv(self, tmp_path: Path) -> None:
        scores = [_scenario("a", ["high"])]
        out = tmp_path / "results.tsv"
        CsvExporter().export(_ctx(tmp_path, scores), out, options={"delimiter": "\t"})
        assert (tmp_path / "results.per_turn.tsv").exists()


# ── JUnit per-turn body ──


class TestJunitPerTurnBody:
    def test_failure_body_carries_turn_idx_lines(self, tmp_path: Path) -> None:
        scores = [_scenario("a", ["high", "low"])]
        out = tmp_path / "report.xml"
        JUnitExporter().export(_ctx(tmp_path, scores), out, options={})
        xml = out.read_text(encoding="utf-8")
        # Per-turn lines must surface BOTH turns and label which one
        # caused the downgrade (turn 1 with score "low").
        assert "[turn 1]" in xml
        assert "correctness=low" in xml


# ── Markdown per-turn block ──


class TestMarkdownPerTurnBlock:
    def test_per_turn_judge_renders_details_block(self, tmp_path: Path) -> None:
        scores = [_scenario("a", ["high", "low"])]
        out = tmp_path / "summary.md"
        MarkdownExporter().export(_ctx(tmp_path, scores), out, options={})
        body = out.read_text(encoding="utf-8")
        # Per-turn detail uses a ``<details>`` block so the headline
        # stays compact while the per-turn drill-down stays inline.
        assert "<details>" in body
        assert "Per-turn detail" in body
        # Both turns' verdicts are listed so a reviewer can attribute
        # the dimension-level downgrade to the specific turn.
        assert "Turn 0" in body
        assert "Turn 1" in body
