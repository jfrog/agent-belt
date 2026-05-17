# (c) JFrog Ltd. (2026)

"""Tests for :func:`belt.scorer.llm.preflight.preflight_judges`.

Verifies the fan-out / aggregation contract:

- A single judge: a failing preflight raises the backend's ``ScorerError``
  verbatim (no wrapping noise).
- Multi-judge: every judge is probed, EVERY failure is collected, and the
  composite ``ScorerError`` includes every failed judge's label and hint
  (so the user fixes their config in one round-trip, not N).
- Consensus: flattened into constituent judges; same probe set as
  declared, no double-probing.
- No LLM scorers (rules-only ``--modes``): no-op, no exceptions.

These are unit-level tests that patch each ``LLMScorer.backend``'s
``preflight_model`` so no network is touched and the test is fast.
"""

from __future__ import annotations

from typing import Any

import pytest

from belt.errors import ScorerError
from belt.scorer.llm.preflight import collect_llm_scorers, preflight_judges


class _StubBackend:
    """Minimal backend stub: just provider_name + preflight_model behaviour."""

    def __init__(self, provider: str = "stub", outcome: Any = None) -> None:
        self._provider = provider
        self._outcome = outcome
        self.calls: int = 0

    def provider_name(self) -> str:
        return self._provider

    def preflight_model(self, config: Any, *, timeout: float | None = None) -> None:
        self.calls += 1
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        # None ⇒ success


class _StubScorer:
    """Stand-in for :class:`LLMScorer` minus the real construction.

    Quacks as an ``LLMScorer`` for ``collect_llm_scorers``: it carries
    a ``backend``, a ``judge_name``, and a ``config`` with a ``model``.
    Avoids the env-var dance of building a real one.
    """

    def __init__(self, name: str, model: str, backend: _StubBackend) -> None:
        self.judge_name = name
        self.backend = backend

        class _Cfg:
            pass

        cfg = _Cfg()
        cfg.model = model  # type: ignore[attr-defined]
        self.config = cfg


@pytest.fixture
def patch_llmscorer_isinstance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make :func:`collect_llm_scorers` accept :class:`_StubScorer` as an LLMScorer.

    We swap ``LLMScorer`` in the preflight module with a tuple that
    matches both the real class AND our stub, so the production
    ``isinstance`` check picks up stubs. This is far cleaner than
    building real ``LLMScorer`` instances (which would require BELT_*
    env vars and a resolvable backend).
    """
    from belt.scorer.llm import preflight as preflight_module
    from belt.scorer.llm.scorer import LLMScorer

    monkeypatch.setattr(preflight_module, "LLMScorer", (LLMScorer, _StubScorer))


class TestCollectLLMScorers:
    """The flatten step: walk scorers, return one entry per unique judge."""

    def test_collects_a_single_llm_scorer(self, patch_llmscorer_isinstance: None) -> None:
        s = _StubScorer("judge_a", "openai/gpt-x", _StubBackend("OpenAI"))
        assert collect_llm_scorers([s]) == [s]

    def test_skips_non_llm_scorers(self, patch_llmscorer_isinstance: None) -> None:
        # A rules scorer is just a plain object - shouldn't be returned.
        rules = object()
        s = _StubScorer("j", "openai/x", _StubBackend("OpenAI"))
        assert collect_llm_scorers([rules, s]) == [s]

    def test_deduplicates_identical_objects(self, patch_llmscorer_isinstance: None) -> None:
        s = _StubScorer("j", "openai/x", _StubBackend("OpenAI"))
        # Same object passed twice ⇒ probe once.
        assert collect_llm_scorers([s, s]) == [s]


class TestPreflightOrchestration:
    """End-to-end ``preflight_judges`` contract."""

    def test_no_scorers_is_noop(self, patch_llmscorer_isinstance: None) -> None:
        # Rules-only run: no LLM scorers, no exceptions.
        preflight_judges([])

    def test_all_pass_returns_silently(self, patch_llmscorer_isinstance: None) -> None:
        scorers = [
            _StubScorer("j1", "openai/gpt-x", _StubBackend("OpenAI", outcome=None)),
            _StubScorer("j2", "anthropic/claude-x", _StubBackend("Anthropic", outcome=None)),
        ]
        preflight_judges(scorers)
        # Every backend was probed exactly once.
        for s in scorers:
            assert s.backend.calls == 1

    def test_single_failure_reraises_verbatim(self, patch_llmscorer_isinstance: None) -> None:
        original = ScorerError(
            "OpenAI judge preflight failed for model 'gpt-X': HTTP 403\n"
            "  Hint: 403 + model_not_found - your project / API key ..."
        )
        s = _StubScorer("only_judge", "openai/gpt-x", _StubBackend("OpenAI", outcome=original))
        with pytest.raises(ScorerError) as exc:
            preflight_judges([s])
        # Same instance: no wrapping for a single failure (no information to add).
        assert exc.value is original

    def test_multi_failure_aggregates_with_all_labels(self, patch_llmscorer_isinstance: None) -> None:
        err_a = ScorerError("OpenAI preflight failed: 401 bad key")
        err_b = ScorerError("Anthropic preflight failed: 403 model_not_found")
        scorers = [
            _StubScorer("alpha", "openai/x", _StubBackend("OpenAI", outcome=err_a)),
            _StubScorer("beta", "anthropic/y", _StubBackend("Anthropic", outcome=err_b)),
        ]
        with pytest.raises(ScorerError) as exc:
            preflight_judges(scorers)
        msg = str(exc.value)
        # Composite message: every judge labelled, every body included.
        assert "alpha" in msg
        assert "beta" in msg
        assert "OpenAI" in msg
        assert "Anthropic" in msg
        assert "401" in msg
        assert "403" in msg
        # And the headline mentions the count so the user knows it's
        # 2 distinct config bugs, not one wrapped twice.
        assert "2 judge(s)" in msg

    def test_partial_failure_still_aggregates(self, patch_llmscorer_isinstance: None) -> None:
        # Three judges, two fail. Composite must include both failures.
        scorers = [
            _StubScorer(
                "alpha",
                "openai/x",
                _StubBackend("OpenAI", outcome=ScorerError("OpenAI preflight failed: 401")),
            ),
            _StubScorer("beta", "anthropic/y", _StubBackend("Anthropic", outcome=None)),
            _StubScorer(
                "gamma",
                "openai/z",
                _StubBackend("OpenAI", outcome=ScorerError("OpenAI preflight failed: 404")),
            ),
        ]
        with pytest.raises(ScorerError) as exc:
            preflight_judges(scorers)
        msg = str(exc.value)
        assert "alpha" in msg and "gamma" in msg
        # Successful judge label MAY appear (we include the headline only),
        # but its error body must not.
        assert "2 judge(s)" in msg

    def test_message_is_stable_across_runs(self, patch_llmscorer_isinstance: None) -> None:
        # Determinism: same input ⇒ same composite message. The fan-out
        # uses ThreadPoolExecutor (non-deterministic order), so the
        # composite must sort the failures before rendering.
        def build_scorers() -> list[_StubScorer]:
            return [
                _StubScorer(
                    "zulu",
                    "openai/x",
                    _StubBackend("OpenAI", outcome=ScorerError("OpenAI: 401 bad key")),
                ),
                _StubScorer(
                    "alpha",
                    "anthropic/y",
                    _StubBackend("Anthropic", outcome=ScorerError("Anthropic: 404 typo")),
                ),
            ]

        first = None
        for _ in range(10):
            try:
                preflight_judges(build_scorers())
            except ScorerError as exc:
                msg = str(exc)
                if first is None:
                    first = msg
                else:
                    assert msg == first, "preflight composite message is not deterministic"
