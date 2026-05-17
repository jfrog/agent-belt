# (c) JFrog Ltd. (2026)

"""Unit tests for scorer.events - ScoreEvent and format_score_event."""

from __future__ import annotations

from belt.scorer.llm.events import ScoreEvent, format_score_event


class TestScoreEvent:
    def test_to_dict_minimal(self):
        e = ScoreEvent(kind="start", scenario="test")
        d = e.to_dict()
        assert d == {"kind": "start", "scenario": "test"}

    def test_to_dict_full(self):
        e = ScoreEvent(
            kind="verdict",
            scenario="sec",
            dimension="execution",
            score="high",
            reasoning="Clean run",
            judge="judge-a",
            passed=True,
        )
        d = e.to_dict()
        assert d["kind"] == "verdict"
        assert d["dimension"] == "execution"
        assert d["score"] == "high"
        assert d["reasoning"] == "Clean run"
        assert d["judge"] == "judge-a"
        assert d["passed"] is True

    def test_to_dict_extra(self):
        e = ScoreEvent(kind="start", scenario="x", extra={"tokens": 42})
        d = e.to_dict()
        assert d["tokens"] == 42


class TestFormatScoreEvent:
    def test_start(self):
        e = ScoreEvent(kind="start", scenario="x")
        line = format_score_event(e)
        assert "🎯" in line
        assert "scoring" in line

    def test_cache_hit(self):
        e = ScoreEvent(kind="cache_hit", scenario="x")
        line = format_score_event(e)
        assert "⚡" in line
        assert "cache hit" in line

    def test_verdict_high(self):
        e = ScoreEvent(kind="verdict", scenario="x", dimension="exec", score="high", reasoning="Good")
        line = format_score_event(e)
        assert "📊" in line
        assert "exec" in line
        assert "high" in line
        assert "Good" in line

    def test_verdict_low(self):
        e = ScoreEvent(kind="verdict", scenario="x", dimension="safety", score="low", reasoning="Bad")
        line = format_score_event(e)
        assert "red" in line
        assert "safety" in line

    def test_done_pass(self):
        e = ScoreEvent(kind="done", scenario="x", passed=True)
        line = format_score_event(e)
        assert "✅" in line

    def test_done_fail(self):
        e = ScoreEvent(kind="done", scenario="x", passed=False)
        line = format_score_event(e)
        assert "❌" in line

    def test_judge_prefix(self):
        e = ScoreEvent(kind="start", scenario="x", judge="correctness")
        line = format_score_event(e)
        assert "correctness:" in line

    def test_truncates_long_reasoning(self):
        long = "A" * 200
        e = ScoreEvent(kind="verdict", scenario="x", dimension="d", score="high", reasoning=long)
        line = format_score_event(e)
        assert "…" in line
        assert len(long) > 60

    def test_error(self):
        e = ScoreEvent(kind="error", scenario="x")
        line = format_score_event(e)
        assert "⚠" in line
        assert "API error" in line

    def test_unknown_kind(self):
        e = ScoreEvent(kind="mystery", scenario="x")
        line = format_score_event(e)
        assert "❓" in line
        assert "mystery" in line
