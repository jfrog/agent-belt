# (c) JFrog Ltd. (2026)

"""``belt export`` - emit a completed run to one or more destinations.

Reads ``results.json`` (typed as :class:`AggregatedResults`) and per-scenario
``score.json`` from a run directory, then dispatches to every requested
exporter. The library entry point :func:`run_exporters` is what
``commands/aggregate.py`` calls when chained via ``--export`` / ``--export-config``;
the CLI entry point :func:`main` powers ``belt export <run-dir>``.

Failure isolation (Design Principle 7): each exporter runs in its own
try/except. A single exporter raising surfaces as a typed
:class:`belt.errors.BeltError` and the next exporter still runs.
The exit code is non-zero only when every requested exporter failed, mirroring
the convention ``belt eval`` uses for the run/score/aggregate chain.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from belt import _io, envvars
from belt._ui import eprint
from belt.constants import RESULTS_FILE, SCORE_FILE
from belt.entities import AggregatedResults, ScenarioScore
from belt.errors import BeltError, ConfigError
from belt.exporter import (
    BaseExporter,
    ExportContext,
    ExporterEntry,
    ExporterFile,
    available_exporters,
    get_exporter_class,
)
from belt.schema import check_schema_version

# ── Spec parsing ──


def parse_to_spec(raw: str) -> tuple[str, str]:
    """Parse a ``NAME:PATH`` ``--export`` / ``--to`` argument.

    Path is required: the CLI surface stays explicit so a user reading a CI
    log sees exactly where each exporter wrote. ``parse_threshold`` uses the
    same colon-split convention; reusing it keeps the CLI vocabulary uniform.
    """
    if ":" not in raw:
        raise ConfigError(f"--export expects NAME:PATH (got {raw!r}). " f"Example: --export csv:results.csv")
    name, path = raw.split(":", 1)
    name = name.strip()
    path = path.strip()
    if not name:
        raise ConfigError(f"--export missing exporter name in {raw!r}")
    if not path:
        raise ConfigError(f"--export missing output path in {raw!r}")
    return name, path


def _resolve_path(run_dir: Path, raw: str) -> Path:
    """Resolve a per-exporter path against the run directory.

    Absolute paths are honoured as-is so CI workflows that need a fixed
    artefact location (``/tmp/belt-junit.xml``) work without surprise.
    Relative paths land under ``run_dir`` so the default behaviour keeps
    every artefact next to the run that produced it.
    """
    p = Path(raw)
    if p.is_absolute():
        return p
    return run_dir / p


def load_export_config(path: str | Path) -> ExporterFile:
    """Load and validate an ``--export-config`` YAML file."""
    import yaml

    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"--export-config: file not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"--export-config: invalid YAML in {p}: {e}") from e
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"--export-config: expected a mapping at the top level of {p}, got {type(raw).__name__}")
    try:
        return ExporterFile.model_validate(raw)
    except Exception as e:
        raise ConfigError(f"--export-config: schema error in {p}: {e}") from e


# ── Run discovery ──


def _resolve_latest_run() -> Path:
    """Return the latest run dir from the manifest, or raise."""
    from belt.manifest import Manifest

    latest = Manifest().latest_run
    if not latest:
        raise ConfigError("No previous run found. Run `belt eval ...` first or pass --run-dir.")
    return Path(latest)


def _load_results(run_dir: Path) -> AggregatedResults:
    """Read and parse ``results.json`` into the typed model."""
    path = run_dir / RESULTS_FILE
    raw = _io.read_json(path)
    if raw is None:
        raise ConfigError(
            f"{RESULTS_FILE} not found under {run_dir}. " f"Run `belt aggregate --run-dir {run_dir}` first."
        )
    try:
        results = AggregatedResults.model_validate(raw)
    except Exception as e:
        raise ConfigError(f"Failed to parse {path}: {e}") from e
    check_schema_version(results.schema_version, str(path))
    return results


def _load_scores(run_dir: Path) -> list[ScenarioScore]:
    """Walk ``run_dir`` for ``score.json`` files and return parsed entries."""
    scores: list[ScenarioScore] = []
    for p in sorted(run_dir.rglob(SCORE_FILE)):
        try:
            score = ScenarioScore.model_validate_json(p.read_text())
            check_schema_version(score.schema_version, str(p))
            scores.append(score)
        except Exception as e:
            logger.warning("Failed to parse {}: {}", p, e)
    return scores


def _load_benchmark_card(run_dir: Path) -> dict | None:
    """Best-effort load of ``benchmark-card.json`` for exporter context."""
    from belt.constants import BENCHMARK_CARD_JSON_FILE

    return _io.read_json(run_dir / BENCHMARK_CARD_JSON_FILE)


# ── Slot assembly + dispatch ──


def _build_slots(
    to_specs: list[str],
    config_path: str | Path | None,
) -> list[ExporterEntry]:
    """Merge ``--export NAME:PATH`` and ``--export-config`` into a flat list.

    CLI flags run first (more visible to humans reading the invocation),
    YAML entries follow (programmatic). When two slots resolve to the same
    on-disk path, both still run; later writes win, with a warning.
    """
    slots: list[ExporterEntry] = []
    for raw in to_specs:
        name, path = parse_to_spec(raw)
        slots.append(ExporterEntry(name=name, path=path))
    if config_path is not None:
        cfg = load_export_config(config_path)
        slots.extend(cfg.exporters)
    return slots


def _instantiate(name: str) -> BaseExporter:
    """Resolve an exporter name to an instance via the registry."""
    cls = get_exporter_class(name)
    return cls()


def run_exporters(
    *,
    run_dir: Path,
    results: AggregatedResults,
    scores: list[ScenarioScore],
    to_specs: list[str],
    config_path: str | Path | None,
) -> int:
    """Drive every requested exporter once. Returns 0 unless ALL exporters failed.

    The same library entry point that ``belt export`` and the
    ``--export``/``--export-config`` chain on ``aggregate`` and ``eval``
    use, so the dispatch behaviour is identical regardless of which CLI
    surface invoked it.
    """
    try:
        slots = _build_slots(to_specs, config_path)
    except BeltError as e:
        eprint(f"\n  ❌ {e}")
        return 1

    if not slots:
        eprint(
            "\n  ⚠ No exporters requested; pass --export NAME:PATH or --export-config "
            f"PATH (available: {', '.join(available_exporters())})."
        )
        return 1

    ctx = ExportContext(
        run_dir=run_dir,
        results=results,
        scores=scores,
        benchmark_card=_load_benchmark_card(run_dir),
    )

    seen_paths: dict[str, str] = {}
    succeeded = 0
    failed = 0
    for slot in slots:
        try:
            exporter = _instantiate(slot.name)
        except BeltError as e:
            eprint(f"  ❌ {slot.name}: {e}")
            failed += 1
            continue

        output = _resolve_path(run_dir, slot.path)
        canonical = str(output.resolve())
        if canonical in seen_paths:
            logger.warning(
                "exporter '{}' overwriting path also targeted by '{}': {}",
                slot.name,
                seen_paths[canonical],
                output,
            )
        seen_paths[canonical] = slot.name

        try:
            exporter.export(ctx, output, slot.options)
        except Exception as e:
            # Per Design Principle 7, exporter failures surface as a typed
            # one-line error rather than aborting the whole CLI. Other
            # exporters still run.
            eprint(f"  ❌ {slot.name}: {type(e).__name__}: {e}")
            logger.exception("exporter {} failed", slot.name)
            failed += 1
            continue

        eprint(f"  → {slot.name}: {output}")
        succeeded += 1

    if succeeded == 0 and failed > 0:
        return 1
    return 0


# ── CLI entry point ──


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="belt export",
        description=(
            "Emit a completed run to one or more configured destinations. "
            "Run `belt doctor` for the live list of registered exporters."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        default=None,
        help="Run directory (default: latest run from .manifest.json).",
    )
    # Within this group, ``add_argument`` calls are alphabetised by long flag
    # name. Enforced by ``tests/test_cli_order.py``.
    exe = parser.add_argument_group("Execution")
    exe.add_argument(
        "--allow-arbitrary-exporter",
        action="store_true",
        default=False,
        help=(
            "Allow --export / --to to resolve to a dotted import path. By default "
            "only built-in exporters and exporters registered as ``belt.exporters`` "
            "entry points are loadable."
        ),
    )
    exe.add_argument(
        "--to",
        action="append",
        default=[],
        metavar="NAME:PATH",
        help=(
            "Run an exporter, writing to PATH (repeatable). NAME is any name "
            "shown by `belt doctor` under Exporters. "
            "Example: --to csv:results.csv --to junit:report.xml."
        ),
    )
    exe.add_argument(
        "--to-config",
        metavar="PATH",
        help="YAML file describing exporter entries (name + path + options).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    envvars.forward_security_toggles(args)

    try:
        run_dir = Path(args.run_dir) if args.run_dir else _resolve_latest_run()
    except BeltError as e:
        eprint(f"\n  ❌ {e}")
        return 1

    if not run_dir.is_dir():
        eprint(f"\n  ❌ Run directory does not exist: {run_dir}")
        return 1

    try:
        results = _load_results(run_dir)
    except BeltError as e:
        eprint(f"\n  ❌ {e}")
        return 1

    scores = _load_scores(run_dir)

    return run_exporters(
        run_dir=run_dir,
        results=results,
        scores=scores,
        to_specs=list(args.to or []),
        config_path=args.to_config,
    )


if __name__ == "__main__":
    sys.exit(main())
