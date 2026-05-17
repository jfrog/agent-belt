# (c) JFrog Ltd. (2026)

"""Integration tests for scorer streaming - callback wiring, NDJSON output, progress display."""

from __future__ import annotations

import io
import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

from rich.console import Console

from belt.entities import JudgeConfig, Scenario, Turn, TurnOutput
from belt.progress import ScorerProgress
from belt.scorer.llm.consensus import ConsensusScorer
from belt.scorer.llm.events import ScoreEvent
from belt.scorer.llm.scorer import LLMScorer

_OPENAI_ENV = {"BELT_OPENAI_API_KEY": "sk-test"}


def _make_scorer(**kwargs) -> LLMScorer:
    config = JudgeConfig(model="openai/gpt-4.1", temperature=0.0, seed=42)
    from belt.scorer.llm.backend import OpenAIBackend

    return LLMScorer(config, backend=OpenAIBackend(), skip_availability=True, **kwargs)


class TestLLMScorerEmitsEvents:
    def test_on_event_receives_start_and_done(self):
        events: list[ScoreEvent] = []
        scorer = _make_scorer(on_event=events.append)

        verdict_json = json.dumps(
            {
                "overall_pass": True,
                "execution": {"score": "high", "reasoning": "Good"},
                "trajectory": {"score": "high", "reasoning": "Fine"},
                "response_quality": {"score": "high", "reasoning": "Great"},
                "efficiency": {"score": "high", "reasoning": "Fast"},
            }
        )
        resp_json = {
            "choices": [{"message": {"content": verdict_json}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = resp_json
        mock_resp.status_code = 200

        scenario = Scenario(name="test-sc", description="A test", turns=[Turn(message="hi")])
        turn_outputs = [TurnOutput(raw_cli="output")]

        with patch("belt.scorer.llm.scorer.httpx.post", return_value=mock_resp):
            result = scorer.score(scenario, turn_outputs)

        assert result is not None
        assert result.passed is True

        kinds = [e.kind for e in events]
        assert "start" in kinds
        assert "done" in kinds
        assert kinds[0] == "start"
        assert kinds[-1] == "done"

        verdicts = [e for e in events if e.kind == "verdict"]
        assert len(verdicts) >= 1
        assert all(v.scenario == "test-sc" for v in verdicts)

    def test_on_event_none_is_safe(self):
        scorer = _make_scorer(on_event=None)
        assert scorer.on_event is None

    def test_cache_hit_emits_event(self):
        events: list[ScoreEvent] = []
        scorer = _make_scorer(on_event=events.append)

        mock_cache = MagicMock()
        mock_cache.get.return_value = {
            "verdict": {
                "overall_pass": True,
                "execution": {"score": "high", "reasoning": "cached"},
                "trajectory": {"score": "high", "reasoning": "cached"},
                "response_quality": {"score": "high", "reasoning": "cached"},
                "efficiency": {"score": "high", "reasoning": "cached"},
            },
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
        scorer.cache = mock_cache

        scenario = Scenario(name="cached-sc", description="test", turns=[Turn(message="hi")])
        turn_outputs = [TurnOutput(raw_cli="output")]

        result = scorer.score(scenario, turn_outputs)
        assert result is not None

        kinds = [e.kind for e in events]
        assert "cache_hit" in kinds


class TestConsensusCallbackPropagation:
    def test_set_on_event_injects_judge_name(self):
        events: list[ScoreEvent] = []

        with patch.dict("os.environ", _OPENAI_ENV):
            j1 = _make_scorer()
            j1.judge_name = "judge-a"
            j2 = _make_scorer()
            j2.judge_name = "judge-b"

        consensus = ConsensusScorer([j1, j2])
        consensus.set_on_event(events.append)

        j1.on_event(ScoreEvent(kind="start", scenario="test"))
        j2.on_event(ScoreEvent(kind="start", scenario="test"))

        assert events[0].judge == "judge-a"
        assert events[1].judge == "judge-b"

    def test_set_on_event_none_clears(self):
        with patch.dict("os.environ", _OPENAI_ENV):
            j1 = _make_scorer()
            j1.judge_name = "a"
            j2 = _make_scorer()
            j2.judge_name = "b"

        consensus = ConsensusScorer([j1, j2])
        consensus.set_on_event(lambda e: None)
        consensus.set_on_event(None)

        assert j1.on_event is None
        assert j2.on_event is None


class TestScorerProgressAddEvent:
    def test_add_event_stores_and_orders(self):
        progress = ScorerProgress(live=True)
        progress.add_event("sc-a", "  🎯 scoring…")
        progress.add_event("sc-b", "  ⚡ cache hit")
        progress.add_event("sc-a", "  📊 exec: high")

        assert progress._scenario_order == ["sc-a", "sc-b"]
        assert len(progress._scenario_events["sc-a"]) == 2
        assert len(progress._scenario_events["sc-b"]) == 1

    def test_add_event_thread_safe(self):
        progress = ScorerProgress(live=True)
        barrier = threading.Barrier(4)

        def _add(name: str):
            barrier.wait()
            for i in range(50):
                progress.add_event(name, f"event-{i}")

        threads = [threading.Thread(target=_add, args=(f"sc-{i}",)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total = sum(len(v) for v in progress._scenario_events.values())
        assert total == 200

    def test_build_stream_panel_renders(self):
        buf = io.StringIO()
        console = Console(file=buf, no_color=True, width=120)
        progress = ScorerProgress(console=console, live=True)
        progress.add_event("sc-a", "  🎯 scoring…")
        progress.add_event("sc-a", "  📊 exec: [green]high[/green]")

        panel = progress._build_stream_panel()
        console.print(panel)
        output = buf.getvalue()
        assert "sc-a" in output
        assert "Scorer Output" in output


class TestNdjsonWriter:
    def test_writes_events(self, tmp_path: Path):
        from belt.commands.score import _NdjsonWriter

        path = tmp_path / "score_stream.ndjson"
        writer = _NdjsonWriter.from_path(path)

        writer.write(ScoreEvent(kind="start", scenario="test"))
        writer.write(ScoreEvent(kind="verdict", scenario="test", dimension="exec", score="high"))
        writer.write(ScoreEvent(kind="done", scenario="test", passed=True))
        writer.close()

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3

        first = json.loads(lines[0])
        assert first["kind"] == "start"
        assert first["scenario"] == "test"

        last = json.loads(lines[2])
        assert last["kind"] == "done"
        assert last["passed"] is True


class TestEventSinkWiring:
    def test_build_event_sink_fans_out(self, tmp_path: Path):
        from belt.commands.score import _build_event_sink, _NdjsonWriter

        buf = io.StringIO()
        console = Console(file=buf, no_color=True, width=120)
        progress = ScorerProgress(console=console, live=True)
        writer = _NdjsonWriter.from_path(tmp_path / "stream.ndjson")

        sink = _build_event_sink(progress, writer)
        sink(ScoreEvent(kind="start", scenario="my-sc"))
        writer.close()

        assert "my-sc" in progress._scenario_order
        assert len(progress._scenario_events["my-sc"]) == 1

        ndjson = (tmp_path / "stream.ndjson").read_text().strip()
        assert json.loads(ndjson)["kind"] == "start"

    def test_wire_event_callbacks_to_llm_scorer(self):
        from belt.commands.score import _wire_event_callbacks

        events: list[ScoreEvent] = []
        scorer = _make_scorer()
        _wire_event_callbacks([scorer], events.append)
        assert scorer.on_event is not None

        scorer.on_event(ScoreEvent(kind="start", scenario="x"))
        assert len(events) == 1

    def test_wire_event_callbacks_to_consensus(self):
        from belt.commands.score import _wire_event_callbacks

        events: list[ScoreEvent] = []
        with patch.dict("os.environ", _OPENAI_ENV):
            j1 = _make_scorer()
            j1.judge_name = "a"
            j2 = _make_scorer()
            j2.judge_name = "b"

        consensus = ConsensusScorer([j1, j2])
        _wire_event_callbacks([consensus], events.append)

        j1.on_event(ScoreEvent(kind="start", scenario="x"))
        assert events[0].judge == "a"

    def test_wire_none_is_noop(self):
        scorer = _make_scorer()
        scorer.on_event = lambda e: None
        from belt.commands.score import _wire_event_callbacks

        _wire_event_callbacks([scorer], None)
        assert scorer.on_event is not None  # unchanged - None callback means don't wire
