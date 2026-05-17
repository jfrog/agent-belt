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
from belt.schema import check_schema_version
from belt.scorer import BaseScorer, ConsensusScorer, LLMScorer, RuleBasedScorer
from belt.scorer.entities import JudgeConfig
from belt.scorer.payloads import CheckEntry, LLMPayload, RulesPayload, ScorerPayload
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


def load_scorer_config(config_path: str) -> tuple[list[dict], str | None]:
    """Load multi-judge config from YAML.

    Returns ``(judge_defs, consensus_strategy)``. ``consensus_strategy`` is
    ``None`` when the ``consensus`` key is absent (independent judges).

    Raises ``ConfigError`` on bad path or missing ``judges`` section.
    """
    import yaml

    from belt.errors import ConfigError
    from belt.scorer.llm.consensus import CONSENSUS_STRATEGIES

    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"Scorer config not found: {config_path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    judges = data.get("judges", {})
    if not judges:
        raise ConfigError(f"No 'judges' section in {config_path}")

    consensus = data.get("consensus")
    if consensus is not None:
        consensus = str(consensus).strip().lower()
        if consensus not in CONSENSUS_STRATEGIES:
            raise ConfigError(
                f"Unknown consensus strategy '{consensus}'. Valid: {', '.join(sorted(CONSENSUS_STRATEGIES))}"
            )

    return [{"name": name, **cfg} for name, cfg in judges.items()], consensus


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
                if "model" in jdef:
                    overrides["model"] = jdef["model"]
                try:
                    if "temperature" in jdef:
                        overrides["temperature"] = float(jdef["temperature"])
                    if "seed" in jdef:
                        overrides["seed"] = int(jdef["seed"])
                    if "max_tokens" in jdef:
                        overrides["max_tokens"] = int(jdef["max_tokens"])
                    if "cost_per_prompt_token" in jdef:
                        overrides["cost_per_prompt_token"] = float(jdef["cost_per_prompt_token"])
                    if "cost_per_completion_token" in jdef:
                        overrides["cost_per_completion_token"] = float(jdef["cost_per_completion_token"])
                except (TypeError, ValueError) as e:
                    from belt.errors import ConfigError

                    raise ConfigError(
                        f"Invalid numeric value in scorer config judge '{jdef.get('name', '?')}': {e}"
                    ) from e

                from belt.config import load_judge_config

                cfg = load_judge_config(cli_overrides=overrides)
                try:
                    max_r = int(jdef.get("max_retries", "5"))
                except (TypeError, ValueError):
                    max_r = 5

                strategy = None
                dims = jdef.get("dimensions")
                preamble = jdef.get("system_preamble")
                extend = jdef.get("extend_defaults", False)
                if dims or preamble:
                    parsed_dims = parse_dimension_defs(dims or [])
                    if extend:
                        from belt.agent.scoring import GENERIC_DIMENSIONS

                        existing_names = {d.name for d in parsed_dims}
                        base = [d for d in GENERIC_DIMENSIONS if d.name not in existing_names]
                        parsed_dims = base + parsed_dims
                    strategy = ScoringStrategy(
                        dimensions=parsed_dims,
                        agent_context=preamble or "",
                    )

                s = LLMScorer(cfg, max_retries=max_r, strategy=strategy, skip_availability=skip_availability)
                s.judge_name = jdef["name"]
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
            if isinstance(payload, LLMPayload) and payload.judge_errored:
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
