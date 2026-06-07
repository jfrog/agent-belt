# (c) JFrog Ltd. (2026)

"""Scoring pipeline - scenario → ``ScenarioScore``.

Three responsibilities:

- ``score_scenario`` - load the source scenario, replay turn outputs, run
  every configured scorer, and assemble a ``ScenarioScore``.
- ``build_scorers`` - turn ``--modes`` + ``-S`` flags + optional
  ``--scorer-config`` YAML into concrete scorer instances.
- ``validate_scorers`` - preflight wrapper used by ``commands/eval`` to
  fail fast (and to produce the human-readable banner descriptions) before
  the run phase starts.

The CLI entry point (``commands/score.py``) is intentionally thin: it
parses argv and delegates here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from belt.cli_utils import parse_kv_args
from belt.constants import SCHEMA_VERSION, TURN_CLI_TEMPLATE, TURN_OUTPUT_TEMPLATE, TURN_STATE_TEMPLATE
from belt.entities import ScenarioScore, TurnOutput
from belt.parser.scenario import ScenarioLoader
from belt.scenario import Scenario
from belt.schema import check_schema_version
from belt.scorer import BaseScorer, ConsensusScorer, LLMScorer, RuleBasedScorer
from belt.scorer.config_schema import JudgeDef, ScorerConfigFile
from belt.scorer.entities import JudgeConfig
from belt.scorer.payloads import CheckEntry, LLMPayload, PerTurnLLMPayload, RulesPayload, ScorerPayload
from belt.scorer.scenario_map import map_to_scenario, parse_dimension_defs

if TYPE_CHECKING:
    pass

BUILTIN_MODES = frozenset({"rules", "llm"})


def discover_outcome_dirs(root: Path) -> list[Path]:
    """Find every directory containing at least one ``turn_*_cli_output.txt``."""
    return sorted({p.parent for p in root.rglob(TURN_CLI_TEMPLATE.format("*"))})


def build_judge_config_from_scorer_args(scorer_args: dict[str, str]) -> JudgeConfig:
    """Build a JudgeConfig from -S scorer args, with type coercion."""
    overrides: dict[str, object] = {}
    if "model" in scorer_args:
        overrides["model"] = scorer_args["model"]
    if "temperature" in scorer_args:
        overrides["temperature"] = float(scorer_args["temperature"])
    if "seed" in scorer_args:
        overrides["seed"] = int(scorer_args["seed"])
    if "max_tokens" in scorer_args:
        overrides["max_tokens"] = int(scorer_args["max_tokens"])
    if "max_prompt_chars" in scorer_args:
        overrides["max_prompt_chars"] = int(scorer_args["max_prompt_chars"])

    from belt.config import load_judge_config

    return load_judge_config(cli_overrides=overrides)


def load_scorer_config(config_path: str) -> tuple[list[JudgeDef], str | None]:
    """Load multi-judge config from YAML.

    Returns ``(judge_defs, consensus_strategy)``. ``consensus_strategy`` is
    ``None`` when the ``consensus`` key is absent (independent judges).

    Validation lives in :class:`belt.scorer.config_schema.ScorerConfigFile`:
    unknown top-level / per-judge keys, name clashes with reserved
    scorer keys (``rules`` / ``llm``), out-of-range numerics, and bad
    ``resolution`` / ``evidence_scope`` literals are all caught here
    with a structured Pydantic error rather than at the call site that
    would silently misread the dict.
    """
    import yaml
    from pydantic import ValidationError

    from belt.errors import ConfigError

    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"Scorer config not found: {config_path}")
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Scorer config {config_path}: top-level must be a mapping, got {type(data).__name__}")

    try:
        parsed = ScorerConfigFile.model_validate(data)
        judge_defs = parsed.to_judge_defs()
    except (ValidationError, ValueError) as exc:
        raise ConfigError(f"Invalid scorer config {config_path}: {exc}") from exc

    return judge_defs, parsed.consensus


def build_scorers(
    modes: str,
    scorer_args: dict[str, str],
    scorer_config_path: str | None = None,
    skip_availability: bool = False,
) -> tuple[list[BaseScorer], list[LLMScorer]]:
    """Build scorer instances from modes, -S args, and optional scorer-config.

    Built-in modes (``rules``, ``llm``) are handled with their specific
    configuration logic.  Any other mode name is resolved via the scorer
    registry (entry points under ``belt.scorers`` or dotted import
    paths).
    """
    mode_set = {m.strip() for m in modes.split(",") if m.strip()}
    plugin_modes = mode_set - BUILTIN_MODES
    if plugin_modes:
        from belt.scorer.registry import get_scorer_class

        for pm in sorted(plugin_modes):
            get_scorer_class(pm)
    scorers: list[BaseScorer] = []
    llm_scorers: list[LLMScorer] = []

    if "rules" in mode_set:
        scorers.append(RuleBasedScorer())

    if "llm" in mode_set:
        if scorer_config_path:
            judge_defs, consensus_strategy = load_scorer_config(scorer_config_path)
            config_llm_scorers: list[LLMScorer] = []
            for jdef in judge_defs:
                from belt.agent.scoring import ScoringStrategy

                overrides: dict[str, object] = {}
                if jdef.model is not None:
                    overrides["model"] = jdef.model
                if jdef.temperature is not None:
                    overrides["temperature"] = jdef.temperature
                if jdef.seed is not None:
                    overrides["seed"] = jdef.seed
                if jdef.max_tokens is not None:
                    overrides["max_tokens"] = jdef.max_tokens
                if jdef.max_prompt_chars is not None:
                    overrides["max_prompt_chars"] = jdef.max_prompt_chars
                if jdef.cost_per_prompt_token is not None:
                    overrides["cost_per_prompt_token"] = jdef.cost_per_prompt_token
                if jdef.cost_per_completion_token is not None:
                    overrides["cost_per_completion_token"] = jdef.cost_per_completion_token

                from belt.config import load_judge_config

                cfg = load_judge_config(cli_overrides=overrides)

                strategy = None
                if jdef.dimensions or jdef.system_preamble:
                    parsed_dims = parse_dimension_defs(jdef.dimensions or [])
                    if jdef.extend_defaults:
                        from belt.agent.scoring import GENERIC_DIMENSIONS

                        existing_names = {d.name for d in parsed_dims}
                        base = [d for d in GENERIC_DIMENSIONS if d.name not in existing_names]
                        parsed_dims = base + parsed_dims
                    strategy = ScoringStrategy(
                        dimensions=parsed_dims,
                        agent_context=jdef.system_preamble or "",
                    )

                s = LLMScorer(
                    cfg,
                    max_retries=jdef.max_retries,
                    strategy=strategy,
                    skip_availability=skip_availability,
                    resolution=jdef.resolution,
                    evidence_scope=jdef.evidence_scope,
                )
                s.judge_name = jdef.name
                config_llm_scorers.append(s)

            if consensus_strategy and len(config_llm_scorers) >= 2:
                consensus = ConsensusScorer(config_llm_scorers, strategy=consensus_strategy)
                consensus.emit_dimension_warnings()
                llm_scorers.extend(config_llm_scorers)
                scorers.append(consensus)
            else:
                for s in config_llm_scorers:
                    llm_scorers.append(s)
                    scorers.append(s)
        else:
            judge_config = build_judge_config_from_scorer_args(scorer_args)
            max_retries = int(scorer_args.get("max_retries", "5"))
            llm_scorer = LLMScorer(judge_config, max_retries=max_retries, skip_availability=skip_availability)
            llm_scorers.append(llm_scorer)
            scorers.append(llm_scorer)

    for pm in sorted(plugin_modes):
        from belt.scorer.registry import get_scorer_class

        plugin_cls = get_scorer_class(pm)
        plugin_instance = plugin_cls()
        if not skip_availability and not plugin_instance.is_available():
            logger.warning("Scorer '{}' is not available - skipping", pm)
            continue
        scorers.append(plugin_instance)

    return scorers, llm_scorers


def validate_scorers(
    modes: str,
    scorer_args: list[str] | None = None,
    scorer_config_path: str | None = None,
    *,
    probe_api: bool = True,
) -> list[str]:
    """Preflight: build all scorers, raise ``ConfigError`` if any backend unavailable.

    When ``probe_api`` is ``True`` (the default for live ``belt eval``
    runs), also call each LLM judge's
    :meth:`belt.scorer.llm.backend.BaseJudgeBackend.preflight_model` in
    parallel via :func:`belt.scorer.llm.preflight.preflight_judges`.
    This catches the three judge config bugs (wrong key / wrong model /
    project-scoped key without model access) before the agent phase
    runs - without it, the agent phase spends real money on calls that
    were always going to fail at score time. Transient failures (5xx,
    timeout, rate-limit) do not block the run; the per-scenario
    :class:`belt.errors.JudgeInfraError` path catches them at runtime.

    Callers pass ``probe_api=False`` for ``--dry-run`` (no network is
    appropriate then) or when re-validating already-probed scorers in a
    subcommand that the live run forwarded to.

    Returns a list of human-readable scorer descriptions for the startup banner.
    """
    parsed = {}
    for item in scorer_args or []:
        if "=" in item:
            k, _, v = item.partition("=")
            parsed[k] = v
    scorers, _llm_scorers = build_scorers(modes, parsed, scorer_config_path)
    if probe_api:
        from belt.scorer.llm.preflight import preflight_judges

        preflight_judges(scorers)
    descriptions: list[str] = []
    for s in scorers:
        if isinstance(s, ConsensusScorer):
            judge_descs = []
            for j in s.judges:
                c = j.config
                judge_descs.append(f"{j.judge_name}: {j.backend.provider_name()} / {c.model}")
            descriptions.append(f"consensus ({s.consensus_strategy}): [{', '.join(judge_descs)}]")
        elif isinstance(s, LLMScorer):
            c = s.config
            descriptions.append(
                f"{s.judge_name}: {s.backend.provider_name()} / {c.model} "
                f"(temp={c.temperature}, seed={c.seed}, max_tokens={c.max_tokens})"
            )
        else:
            descriptions.append(s.name)
    return descriptions


def _iter_per_turn_judges(scorers: list[BaseScorer]) -> list[LLMScorer]:
    """Yield all LLM scorers running at ``resolution="turn"``.

    A consensus block is per-turn iff every sub-judge is per-turn
    (enforced at :class:`ConsensusScorer.__init__`), so we flatten
    consensus into its sub-judges so the per-turn preflight can check
    each judge by its declared name.
    """
    out: list[LLMScorer] = []
    for s in scorers:
        if isinstance(s, ConsensusScorer):
            for sub in s.judges:
                if sub.resolution == "turn":
                    out.append(sub)
        elif isinstance(s, LLMScorer) and s.resolution == "turn":
            out.append(s)
    return out


def validate_per_turn_judges_against_scenarios(
    scorers: list[BaseScorer],
    scenarios: list[Scenario],
) -> None:
    """Preflight: every per-turn judge must end up voting at least once.

    For each per-turn judge AND each scenario, scan
    ``Turn.llm_judges[judge.judge_name]`` and refuse the run when:

    - Every turn in the scenario carries ``skip=True`` for this judge
      (``"all_turns_skipped"`` static case). Such a scenario would
      enter :meth:`LLMScorer._score_per_turn` and exit with
      ``judge_errored=True, judge_error_type="all_turns_skipped"`` -
      the runtime taint rule handles it correctly, but catching it
      pre-run gives the author a clear authoring error rather than a
      midway abort.

    - A turn references a judge that is not declared at the
      scorer-config level (typo). Without this check the per-turn
      override is silently ignored.

    Raises :class:`belt.errors.ConfigError` with the offending
    judge / scenario / turn so the author can fix the YAML or
    scenario before any judge call happens.
    """
    from belt.errors import ConfigError

    per_turn_judges = _iter_per_turn_judges(scorers)
    if not per_turn_judges:
        return

    declared_judges = {j.judge_name for j in per_turn_judges}

    errors: list[str] = []
    for scenario in scenarios:
        # Catch typos: a turn references a judge name that isn't declared.
        for i, turn in enumerate(scenario.turns):
            for judge_name in turn.llm_judges:
                if judge_name not in declared_judges:
                    errors.append(
                        f"scenario {scenario.name!r} turn {i}: ``llm_judges[{judge_name!r}]`` "
                        f"references an unknown judge. Declared per-turn judges: "
                        f"{sorted(declared_judges) or 'none'}"
                    )

        # Catch all-skipped: every turn skipped this per-turn judge.
        for judge in per_turn_judges:
            if not scenario.turns:
                continue
            skips = 0
            mentions = 0
            for turn in scenario.turns:
                override = turn.llm_judges.get(judge.judge_name)
                if override is None:
                    continue
                mentions += 1
                if override.skip:
                    skips += 1
            if mentions > 0 and skips == mentions and mentions == len(scenario.turns):
                errors.append(
                    f"scenario {scenario.name!r}: every turn carries ``llm_judges[{judge.judge_name!r}].skip=true`` - "
                    f"per-turn judge {judge.judge_name!r} would never vote. Remove the judge from the scorer "
                    "config, drop the skip overrides, or omit this scenario from the run."
                )

    if errors:
        raise ConfigError("Per-turn judge preflight failed:\n  - " + "\n  - ".join(errors))


def resolve_scorer_args(args: argparse.Namespace) -> dict[str, str]:
    """Parse -S key=value args."""
    return parse_kv_args(args.scorer_args)


def score_scenario(
    outcome_dir: Path,
    outcomes_root: Path,
    scorers: list[BaseScorer],
) -> ScenarioScore:
    """Run every scorer against one outcome dir; return a ``ScenarioScore``."""
    relative = outcome_dir.relative_to(outcomes_root)
    group = str(Path(*relative.parts[:-1]))
    scenario_name = relative.parts[-1]

    scenario_path = map_to_scenario(outcome_dir, outcomes_root)
    # Effective tags ride along on ScenarioScore so the aggregator can render
    # tag-aware annotations without re-walking the scenario tree. Best-effort:
    # an unparseable group config or scenario just leaves tags empty.
    effective_tags: list[str] = []
    try:
        group_config = ScenarioLoader.load_group_config(scenario_path.parent)
        effective_tags = list(group_config.default_tags)
    except Exception:
        group_config = None

    try:
        scenario = ScenarioLoader.load_scenario(scenario_path)
    except FileNotFoundError:
        logger.error("Scenario file not found: {}", scenario_path)
        return ScenarioScore(
            schema_version=SCHEMA_VERSION,
            scenario_name=scenario_name,
            group=group,
            tags=effective_tags,
            scores={
                "rules": RulesPayload(
                    checks=[
                        CheckEntry(
                            dimension="execution",
                            check="scenario_exists",
                            passed=False,
                            details=str(scenario_path),
                        )
                    ],
                    passed=False,
                )
            },
            overall_pass=False,
        )
    except Exception as e:
        logger.error("Failed to load scenario {}: {}", scenario_path, e)
        return ScenarioScore(
            schema_version=SCHEMA_VERSION,
            scenario_name=scenario_name,
            group=group,
            tags=effective_tags,
            scores={
                "rules": RulesPayload(
                    checks=[
                        CheckEntry(
                            dimension="execution",
                            check="scenario_valid",
                            passed=False,
                            details=str(e),
                        )
                    ],
                    passed=False,
                )
            },
            overall_pass=False,
        )

    effective_tags = sorted(set(effective_tags) | set(scenario.tags))

    turn_outputs: list[TurnOutput] = []
    missing_checks: list[CheckEntry] = []

    for i in range(len(scenario.turns)):
        output_path = outcome_dir / TURN_OUTPUT_TEMPLATE.format(i)
        if output_path.exists():
            try:
                to = TurnOutput.model_validate_json(output_path.read_text())
                check_schema_version(to.schema_version, str(output_path))
                turn_outputs.append(to)
                continue
            except Exception as e:
                logger.warning("Failed to parse {}, falling back to raw files: {}", output_path, e)

        cli_path = outcome_dir / TURN_CLI_TEMPLATE.format(i)
        if not cli_path.exists():
            logger.warning("Missing CLI output: {}", cli_path)
            missing_checks.append(
                CheckEntry(dimension="execution", check=f"turn_{i}_exists", passed=False, details="file missing")
            )
            continue

        try:
            cli_text = cli_path.read_text()
        except Exception as e:
            logger.error("Failed to read {}: {}", cli_path, e)
            missing_checks.append(
                CheckEntry(dimension="execution", check=f"turn_{i}_readable", passed=False, details=str(e))
            )
            continue

        raw_state = None
        state_path = outcome_dir / TURN_STATE_TEMPLATE.format(i)
        if state_path.exists():
            try:
                raw_state = state_path.read_text()
            except Exception as e:
                logger.warning("Failed to read thread state {}: {}", state_path, e)

        turn_outputs.append(TurnOutput(raw_cli=cli_text, raw_state=raw_state))

    all_scores: dict[str, ScorerPayload] = {}
    overall_pass = True

    for scorer in scorers:
        try:
            result = scorer.score(scenario, turn_outputs)
        except Exception as e:
            logger.error("{} scorer failed for {}/{}: {}", scorer.name, group, scenario_name, e)
            result = None
            # A scorer that was requested but produced no payload is not a
            # silent pass: surface the failure as a synthetic ``execution``
            # check on the rules scorer (so it shows up in score.json,
            # results.json, exporters, and the failure renderer) and force
            # the scenario to fail. Without this, an HTTP 401 / 5xx from an
            # LLM judge would let CI gates green-light a broken pipeline.
            missing_checks.append(
                CheckEntry(
                    dimension="execution",
                    check=f"{scorer.name}_scorer_ran",
                    passed=False,
                    details=f"{scorer.name} scorer failed: {e}",
                )
            )
            overall_pass = False

        if result is not None:
            payload = result.data
            # An LLM payload with judge_errored=True is a non-verdict (infra
            # failure: rate-limit / timeout / network / parse failure). Treat
            # it the same way the exception path above treats a raised
            # scorer: append a synthetic execution check so the failure
            # reaches every exporter and threshold gate uniformly, and force
            # overall_pass=False so a passing rules scorer cannot silently
            # green-light a scenario whose judge never voted.
            if isinstance(payload, (LLMPayload, PerTurnLLMPayload)) and payload.judge_errored:
                etype = payload.judge_error_type or "other"
                missing_checks.append(
                    CheckEntry(
                        dimension="execution",
                        check=f"{scorer.name}_scorer_ran",
                        passed=False,
                        details=f"{scorer.name} scorer judge-infra failure: {etype}",
                    )
                )
            if scorer.name == "rules" and missing_checks:
                # Prepend turn-load failures to the rules scorer's own checks so
                # downstream consumers see a single, ordered checks list. We
                # rebuild the model rather than mutating it to keep the
                # discriminated-union path validated.
                assert isinstance(
                    payload, RulesPayload
                ), f"'rules' scorer must produce RulesPayload, got {type(payload).__name__}"
                payload = RulesPayload(
                    checks=missing_checks + list(payload.checks),
                    passed=False,
                    has_error=payload.has_error,
                )
                result.passed = False
            all_scores[scorer.name] = payload
            if not result.passed:
                overall_pass = False

    if missing_checks:
        # Two paths into ``missing_checks`` exist: turn-load failures
        # (collected before the scorer loop) and scorer-raised failures
        # (collected inside the loop, possibly after the ``rules`` scorer
        # has already produced its payload). Merge any leftover synthetic
        # checks into the rules payload so they always reach
        # ``score.json`` / exporters / threshold gating, regardless of
        # which scorer raised and in what order.
        existing = all_scores.get("rules")
        if isinstance(existing, RulesPayload):
            already = {(c.check, c.dimension, c.details) for c in existing.checks}
            extras = [c for c in missing_checks if (c.check, c.dimension, c.details) not in already]
            if extras:
                all_scores["rules"] = RulesPayload(
                    checks=list(existing.checks) + extras,
                    passed=False,
                    has_error=existing.has_error,
                )
        else:
            all_scores["rules"] = RulesPayload(checks=missing_checks, passed=False)
        overall_pass = False

    return ScenarioScore(
        schema_version=SCHEMA_VERSION,
        scenario_name=scenario_name,
        group=group,
        tags=effective_tags,
        scores=all_scores,
        overall_pass=overall_pass,
    )
