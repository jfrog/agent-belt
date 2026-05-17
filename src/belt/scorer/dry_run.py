# (c) JFrog Ltd. (2026)

"""``--dry-run`` handler for ``belt score``.

Two preview modes:

- **LLM** - print the exact payload (system message, dynamic message, schema)
  that *would* be sent to the judge.  Verifies judge config + prompt size.
- **Rules** - list the scenario's rule-check categories (no model needed).
  Lets users sanity-check what rule-only scoring will exercise before a run.

Both modes accept any outcome dir (selector handled by the caller).
Neither path spends tokens or executes scorers.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from belt._ui import eprint
from belt.constants import TURN_CLI_TEMPLATE, TURN_OUTPUT_TEMPLATE
from belt.entities import TurnOutput
from belt.parser.scenario import ScenarioLoader
from belt.scorer import ConsensusScorer, LLMScorer
from belt.scorer.base import BaseScorer
from belt.scorer.scenario_map import map_to_scenario, resolve_scoring_strategy


def _collect_llm_judges(scorers: Sequence[BaseScorer] | None, fallback: LLMScorer | None) -> list[LLMScorer]:
    """Flatten LLM judges out of the active scorer list.

    ``ConsensusScorer`` wraps multiple judges; each one needs its own
    preview block because per-judge model, dimensions, and system
    preamble can differ. ``fallback`` is the explicit ``llm_scorer``
    argument and is appended only when the scorer list yields no LLM
    judges, so callers that pass an LLM scorer without including it in
    the scorer list still get a preview.
    """
    judges: list[LLMScorer] = []
    for s in scorers or []:
        if isinstance(s, ConsensusScorer):
            judges.extend(s.judges)
        elif isinstance(s, LLMScorer):
            judges.append(s)
    if not judges and fallback is not None:
        judges.append(fallback)
    return judges


def handle_dry_run(
    llm_scorer: LLMScorer | None,
    outcome_dir: Path,
    outcomes_root: Path,
    scorers: Sequence[BaseScorer] | None = None,
) -> int:
    """Print a no-cost preview for the selected scenario.

    Prints rule expectations when rule scorers are present and a per-judge
    LLM payload preview for every ``LLMScorer`` (including each judge
    inside a ``ConsensusScorer``). Returns 1 only on actual error (load
    failure, no scorers configured at all).
    """
    if llm_scorer is None and not scorers:
        eprint("  \u274c --dry-run requires at least one scorer (--modes must include 'rules' and/or 'llm')")
        return 1

    scenario_path = map_to_scenario(outcome_dir, outcomes_root)
    try:
        scenario = ScenarioLoader.load_scenario(scenario_path)
    except Exception as e:
        eprint(f"  \u274c Failed to load scenario {scenario_path}: {e}")
        return 1

    turn_outputs: list[TurnOutput] = []
    for i in range(len(scenario.turns)):
        output_path = outcome_dir / TURN_OUTPUT_TEMPLATE.format(i)
        if output_path.exists():
            try:
                turn_outputs.append(TurnOutput.model_validate_json(output_path.read_text()))
                continue
            except Exception:
                pass
        cli_path = outcome_dir / TURN_CLI_TEMPLATE.format(i)
        if cli_path.exists():
            turn_outputs.append(TurnOutput(raw_cli=cli_path.read_text()))

    if not turn_outputs:
        eprint(f"  \u274c No turn outputs found in {outcome_dir}")
        return 1

    sep = "\u2500" * 72
    eq_sep = "=" * 72
    eprint(f"\n{eq_sep}")
    eprint(f"  DRY RUN - {scenario.name}")
    eprint(eq_sep)

    if scorers:
        _print_rules_preview(scenario, scorers, sep)

    judges = _collect_llm_judges(scorers, llm_scorer)
    if judges:
        strategy = resolve_scoring_strategy(outcome_dir, outcomes_root)
        for judge in judges:
            effective = judge
            if strategy is not None:
                effective = LLMScorer(judge.config, judge.max_retries, strategy=strategy)
                effective.judge_name = judge.judge_name
            _print_llm_preview(effective, scenario, turn_outputs, sep)

    eprint()
    return 0


def _print_rules_preview(scenario: object, scorers: Sequence[BaseScorer], sep: str) -> None:
    """Print which expectations the scenario's turns will exercise.

    Dumping ``expect.model_dump(exclude_defaults=True)`` keeps this honest
    without maintaining a duplicate field-to-category map: the user sees
    exactly the fields they wrote, and the rule scorer dispatches on the
    same fields at run time.
    """
    rules_scorers = [s for s in scorers if s.name != "llm"]
    eprint(f"\n  Mode: rules ({', '.join(s.name for s in rules_scorers) or 'none'})")
    turns = getattr(scenario, "turns", []) or []
    eprint(f"  Turns: {len(turns)}")
    eprint(f"\n{sep}")
    eprint("  RULE EXPECTATIONS (per turn, defaults omitted):")
    eprint(sep)
    for i, turn in enumerate(turns):
        expect = getattr(turn, "expect", None)
        state = getattr(turn, "state_expect", None)
        active: dict[str, object] = {}
        if expect is not None and hasattr(expect, "model_dump"):
            active.update(expect.model_dump(exclude_defaults=True))
        if state is not None and hasattr(state, "model_dump"):
            state_dump = state.model_dump(exclude_defaults=True)
            if state_dump:
                active["state_expect"] = state_dump
        if active:
            eprint(f"  Turn {i}:")
            for key, val in active.items():
                eprint(f"    {key}: {val}")
        else:
            eprint(f"  Turn {i}: (defaults only - structural checks)")


def _print_llm_preview(
    llm_scorer: LLMScorer,
    scenario: object,
    turn_outputs: list[TurnOutput],
    sep: str,
) -> None:
    """Print the exact LLM payload that would be sent."""
    payload = llm_scorer.dry_run(scenario, turn_outputs)
    suffix = f"  (judge: {llm_scorer.judge_name})" if llm_scorer.judge_name != "llm" else ""
    eprint(f"\n  Mode: llm{suffix}")
    eprint(f"  Backend:     {payload['backend']}")
    eprint(f"  Model:       {payload['model']}")
    eprint(f"  Temperature: {payload['temperature']}")
    eprint(f"  Seed:        {payload['seed']}")
    eprint(f"  Max tokens:  {payload['max_tokens']}")
    eprint(f"  Dimensions:  {', '.join(payload['dimensions'])}")
    eprint(f"\n{sep}")
    eprint("  SYSTEM MESSAGE:")
    eprint(sep)
    eprint(payload["system_message"])
    eprint(f"\n{sep}")
    eprint("  DYNAMIC MESSAGE (truncated to 2000 chars):")
    eprint(sep)
    dynamic = payload["dynamic_message"]
    eprint(dynamic[:2000] + ("..." if len(dynamic) > 2000 else ""))
    eprint(f"\n{sep}")
    eprint("  SCHEMA:")
    eprint(sep)
    eprint(json.dumps(payload["schema"], indent=2))
