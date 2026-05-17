# (c) JFrog Ltd. (2026)

"""Integration: ``validate_scorers(probe_api=...)`` ↔ ``preflight_judges``.

The CLI surface for the judge preflight is one flag on ``validate_scorers``:

- ``probe_api=True`` (live ``belt eval``): build scorers, then
  run :func:`preflight_judges`. A 4xx from any judge raises
  :class:`ScorerError` and ``commands/eval.py`` aborts before the
  agent phase starts.
- ``probe_api=False`` (``--dry-run``, or callers that re-validate
  an already-probed config): build scorers only - no network.

These tests pin both branches: the live-eval default must probe (or
agent runs waste time on configs that will fail at score time), and
``--dry-run`` must NOT probe (or air-gapped CI breaks).

Pure unit-level: we don't instantiate real ``LLMScorer``s (which need
BELT_* env vars). Instead we patch ``build_scorers`` so the test
controls what ``validate_scorers`` sees.
"""

from __future__ import annotations

from typing import Any

import pytest

from belt.errors import ScorerError
from belt.scorer.pipeline import validate_scorers


class _FakeJudge:
    """Quacks as an :class:`LLMScorer` for the preflight collector."""

    def __init__(self, name: str, raises: BaseException | None = None) -> None:
        self.judge_name = name
        self._raises = raises
        self.preflight_calls = 0

        backend = _FakeBackend(self)
        self.backend = backend

        class _Cfg:
            pass

        cfg = _Cfg()
        cfg.model = "openai/fake"  # type: ignore[attr-defined]
        cfg.temperature = 0.0  # type: ignore[attr-defined]
        cfg.seed = 0  # type: ignore[attr-defined]
        cfg.max_tokens = 1  # type: ignore[attr-defined]
        self.config = cfg


class _FakeBackend:
    def __init__(self, parent: _FakeJudge) -> None:
        self._parent = parent

    def provider_name(self) -> str:
        return "Fake"

    def preflight_model(self, config: Any, *, timeout: float | None = None) -> None:
        self._parent.preflight_calls += 1
        if self._parent._raises is not None:
            raise self._parent._raises


def _install_fake_build(monkeypatch: pytest.MonkeyPatch, judges: list[_FakeJudge]) -> None:
    """Patch ``build_scorers`` to return our fakes (no real construction)."""

    def fake_build(modes: str, scorer_args: dict[str, str], scorer_config_path: str | None = None, **_: Any):
        return list(judges), list(judges)

    monkeypatch.setattr("belt.scorer.pipeline.build_scorers", fake_build)
    # The preflight module AND validate_scorers each ``isinstance``-check
    # ``LLMScorer``; treat our fake as one in both modules for the
    # duration of the test so the description-building branch and the
    # preflight-collector branch both see it.
    from belt.scorer import pipeline as pipeline_module
    from belt.scorer.llm import preflight as preflight_module
    from belt.scorer.llm.scorer import LLMScorer

    monkeypatch.setattr(preflight_module, "LLMScorer", (LLMScorer, _FakeJudge))
    monkeypatch.setattr(pipeline_module, "LLMScorer", (LLMScorer, _FakeJudge))


def test_probe_api_true_invokes_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    judges = [_FakeJudge("j1"), _FakeJudge("j2")]
    _install_fake_build(monkeypatch, judges)

    descriptions = validate_scorers("llm", scorer_args=[], probe_api=True)

    for j in judges:
        assert j.preflight_calls == 1, f"{j.judge_name} was not preflighted"
    # And descriptions are still built (the banner string).
    assert isinstance(descriptions, list)


def test_probe_api_false_skips_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    judges = [_FakeJudge("j1"), _FakeJudge("j2")]
    _install_fake_build(monkeypatch, judges)

    validate_scorers("llm", scorer_args=[], probe_api=False)

    for j in judges:
        assert j.preflight_calls == 0, (
            f"{j.judge_name} was preflighted in probe_api=False mode " "(would break --dry-run on air-gapped CI)"
        )


def test_probe_api_true_aborts_on_judge_4xx(monkeypatch: pytest.MonkeyPatch) -> None:
    # A misconfigured judge model must raise BEFORE the agent phase,
    # not after a full run; with probe_api=True the failure surfaces
    # at preflight rather than as N silent score-time errors.
    bad_judge = _FakeJudge(
        "bad",
        raises=ScorerError(
            "OpenAI judge preflight failed for model 'fake': HTTP 403\n"
            "  Hint: 403 + model_not_found - project not authorised"
        ),
    )
    good_judge = _FakeJudge("good")
    _install_fake_build(monkeypatch, [good_judge, bad_judge])

    with pytest.raises(ScorerError) as exc:
        validate_scorers("llm", scorer_args=[], probe_api=True)
    assert "403" in str(exc.value)
    assert "model_not_found" in str(exc.value)


def test_default_is_probe_api_true(monkeypatch: pytest.MonkeyPatch) -> None:
    # Guard against a future refactor flipping the default off; the
    # whole point of the preflight is that it runs by default - opt-in
    # would mean most callers never get the abort-before-agent-phase
    # protection.
    judges = [_FakeJudge("j1")]
    _install_fake_build(monkeypatch, judges)
    validate_scorers("llm", scorer_args=[])
    assert judges[0].preflight_calls == 1
