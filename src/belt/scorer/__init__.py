# (c) JFrog Ltd. (2026)

"""Scorer phase package - re-exports the public scorer surface lazily.

The package is intentionally lazy because :mod:`belt.entities`
imports :mod:`belt.scorer.payloads` to type
``ScenarioScore.scores``; an eager re-export of every scorer class
here would force ``scorer.base`` to load while ``entities`` is still
mid-init and trigger a circular ImportError. Lazy attribute
resolution lets ``from belt.scorer import RuleBasedScorer``
keep working from the test suite and the pipeline without forcing
that eager load.
"""

from __future__ import annotations

from typing import Any

# Map of public names to their canonical home, mirroring the pattern in
# ``belt.entities``. Adding a new public scorer symbol means a single
# entry here; nothing else changes.
_LAZY_MAP: dict[str, str] = {
    "BaseScorer": "belt.scorer.base",
    "AnthropicBackend": "belt.scorer.llm",
    "AzureBackend": "belt.scorer.llm",
    "BaseJudgeBackend": "belt.scorer.llm",
    "ConsensusScorer": "belt.scorer.llm",
    "LLMScorer": "belt.scorer.llm",
    "OllamaBackend": "belt.scorer.llm",
    "OpenAIBackend": "belt.scorer.llm",
    "ScoreCache": "belt.scorer.llm",
    "resolve_backend": "belt.scorer.llm",
    "available_scorers": "belt.scorer.registry",
    "get_scorer_class": "belt.scorer.registry",
    "RuleBasedScorer": "belt.scorer.rules",
}


def __getattr__(name: str) -> Any:
    target = _LAZY_MAP.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(target), name)


__all__ = list(_LAZY_MAP.keys())
