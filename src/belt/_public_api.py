# (c) JFrog Ltd. (2026)

"""Single source of truth for the belt public Python API.

``PUBLIC_API`` maps each name plugins may import from the top-level package
to the internal module that defines it. Both
:mod:`belt.__init__` (runtime lazy re-export) and
``scripts/check_design.py`` (plugin-import lint) read this dict, so the
public surface, the lazy loader, and the design check can never drift
(Design Principle 9).

Adding a symbol here makes it part of the published contract: removals or
renames are breaking changes. Anything not in this dict is internal and
plugins must not import it.

The dict is intentionally flat (one entry per symbol, no grouping by
category) because the keys are the user-facing surface; categorisation
belongs in ``docs/glossary/PLUGGABILITY.md``.
"""

from __future__ import annotations

PUBLIC_API: dict[str, str] = {
    # Agent plugin contract
    "BaseAgentAdapter": "belt.agent.base",
    "AgentOption": "belt.agent.base",
    "AgentNotAvailableError": "belt.agent.base",
    "AgentConfig": "belt.runner.entities",
    "GroupConfig": "belt.scenario",
    "TurnJudgeOverride": "belt.scenario",
    "DimensionDef": "belt.agent.scoring",
    "ScoringStrategy": "belt.agent.scoring",
    # Strict scenario-config validation (``--strict-config``).
    # Plugins call ``register_plugin_scenario_key`` at import time to
    # declare extension keys their scorer / agent reads from
    # scenario JSON or group config; without registration, those
    # keys are rejected as typos when the user opts into strict mode.
    "register_plugin_scenario_key": "belt.parser.strict",
    "registered_plugin_scenario_keys": "belt.parser.strict",
    # Cross-phase data entities (read by exporters, written by runner/scorer)
    "TurnOutput": "belt.entities",
    "ToolCall": "belt.entities",
    "TurnTiming": "belt.entities",
    "ScenarioScore": "belt.entities",
    "AggregatedResults": "belt.entities",
    # Exporter plugin contract
    "BaseExporter": "belt.exporter.base",
    "ExportContext": "belt.exporter.entities",
    # Sandbox provider plugin contract
    "BaseSandboxProvider": "belt.runner.sandbox.base",
    "SandboxContext": "belt.runner.sandbox.base",
    "SandboxHandle": "belt.runner.sandbox.base",
    "SandboxProfile": "belt.scenario",
    # Scorer plugin contract
    "BaseScorer": "belt.scorer.base",
    "BaseJudgeBackend": "belt.scorer.llm.backend",
    "ScorerResult": "belt.scorer.entities",
    # Resolution literals for per-turn vs scenario-level LLM judging.
    # Plugins reading ``--scorer-config`` YAML or constructing an
    # ``LLMScorer`` directly should use these aliases instead of
    # hard-coding the literal pair.
    "Resolution": "belt.scorer.entities",
    "EvidenceScope": "belt.scorer.entities",
    # Scorer payload contract (the canonical on-disk shape that
    # ``ScenarioScore.scores[name]`` carries plus the public iterator
    # any consumer should use to walk per-dimension feedback)
    "CheckEntry": "belt.scorer.payloads",
    "RulesPayload": "belt.scorer.payloads",
    "LLMDimensionVerdict": "belt.scorer.payloads",
    "LLMPayload": "belt.scorer.payloads",
    "PerTurnLLMPayload": "belt.scorer.payloads",
    "TurnVerdict": "belt.scorer.payloads",
    "ConsensusMeta": "belt.scorer.payloads",
    "UsageStats": "belt.scorer.payloads",
    "ScorerPayload": "belt.scorer.payloads",
    "DimensionFeedback": "belt.scorer.payloads",
    "iter_dimension_feedback": "belt.scorer.payloads",
    # Walk every LLM-shaped payload (LLMPayload + PerTurnLLMPayload)
    # uniformly. Plugins that read ``ScenarioScore.scores`` should
    # use these helpers instead of ``s.scores.get("llm")`` so a
    # per-turn or renamed multi-judge config doesn't silently fall
    # off their iteration path.
    "iter_llm_payloads": "belt.scorer.payloads",
    "iter_llm_verdicts": "belt.scorer.payloads",
    # Aggregator helpers consumed by exporter / dashboard plugins that
    # render the agent-vs-task vs judge-vs-task partition derived in
    # ``belt.aggregator.stats``. Promoted from internal to public so a
    # downstream UI can recompute the breakdown without re-implementing
    # the partition logic.
    "build_task_quality_split": "belt.aggregator.stats",
    "collect_agent_errors": "belt.aggregator.stats",
    "collect_judge_errors": "belt.aggregator.stats",
    "level_to_score": "belt.scorer.payloads",
    "register_payload_type": "belt.scorer.payloads",
    "registered_payload_types": "belt.scorer.payloads",
    # Typed ``--scorer-config`` YAML shape. Plugins that consume the
    # parsed scorer-config (e.g. a custom orchestrator) should depend
    # on these Pydantic models rather than untyped dicts so unknown
    # keys, reserved-name collisions, and ``Resolution`` enum
    # validation all stay in one place.
    "JudgeDef": "belt.scorer.config_schema",
    "ScorerConfigFile": "belt.scorer.config_schema",
}

__all__ = ["PUBLIC_API"]
