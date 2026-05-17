# (c) JFrog Ltd. (2026)

"""LLM judge scoring - all modules specific to the LLM scoring mode.

The framework-shared parts of scoring (BaseScorer, ScorerResult, ScoreLevel,
the score CLI driver, the scorer registry) live one level up at
``belt.scorer``. Everything in this subpackage is LLM-only.

Mirrors the ``rules/`` subpackage shape: the public scorer class lives in
``scorer.py`` and is re-exported from this ``__init__`` so that
``belt.scorer.llm:LLMScorer`` keeps resolving as an entry-point target.
"""

from belt.scorer.llm.backend import (
    AnthropicBackend,
    AzureBackend,
    BaseJudgeBackend,
    OllamaBackend,
    OpenAIBackend,
    parse_model_spec,
    resolve_backend,
)
from belt.scorer.llm.cache import ScoreCache
from belt.scorer.llm.consensus import CONSENSUS_STRATEGIES, ConsensusScorer
from belt.scorer.llm.events import ScoreEvent, format_score_event
from belt.scorer.llm.pricing import ModelPricing, compute_cost, lookup_pricing
from belt.scorer.llm.scorer import LLMScorer

__all__ = [
    "AnthropicBackend",
    "AzureBackend",
    "BaseJudgeBackend",
    "CONSENSUS_STRATEGIES",
    "ConsensusScorer",
    "LLMScorer",
    "ModelPricing",
    "OllamaBackend",
    "OpenAIBackend",
    "ScoreCache",
    "ScoreEvent",
    "compute_cost",
    "format_score_event",
    "lookup_pricing",
    "parse_model_spec",
    "resolve_backend",
]
