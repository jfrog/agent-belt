# (c) JFrog Ltd. (2026)

"""End-to-end smoke tests for every CLI subcommand.

Every advertised CLI subcommand must be exercised end-to-end. ``--help`` smoke-tests (in ``make verify-wheel``) prove the
command is reachable; this file proves each command does something useful
when invoked with a minimal real input.

For commands that need prior outcomes (``score``, ``aggregate``, ``view``,
``compare``, ``gc``), we synthesise the outcome dir in a tmp path. For
commands that need an agent (``run``, ``eval``, ``quickstart``), we use
``--dry-run`` where the command supports it. Subcommands gated on real
agent CLIs are skipped here and covered by future credentialed CI jobs.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SHOWCASE = REPO_ROOT / "examples" / "scenarios" / "showcase"
_BELT_BIN = shutil.which("belt")


def _need_bin() -> None:
    if not _BELT_BIN:
        pytest.skip("belt console script not on PATH")


def _run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    _need_bin()
    return subprocess.run(
        [_BELT_BIN, *args],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(cwd) if cwd else None,
    )


def _make_outcomes(root: Path, *, scenarios: list[tuple[str, bool]]) -> Path:
    """Build a synthetic ``outcomes/<run-id>/`` tree with ``score.json`` per scenario."""
    run = root / "outcomes" / "20260503-160000"
    for group, scenario, passed in [(g, s, p) for g, (s, p) in zip(["demo"] * len(scenarios), scenarios)]:
        sd = run / group / scenario
        sd.mkdir(parents=True, exist_ok=True)
        score = {
            "schema_version": "1",
            "scenario_name": scenario,
            "group": group,
            "scores": {
                "rules": {
                    "schema_version": "rules.v1",
                    "checks": [
                        {
                            "dimension": "execution",
                            "check": "no_errors",
                            "passed": passed,
                            "details": "" if passed else "synthetic failure",
                            "turn_idx": 0,
                        }
                    ],
                    "passed": passed,
                }
            },
            "overall_pass": passed,
        }
        (sd / "score.json").write_text(json.dumps(score))
    return run


@pytest.fixture
def synth_run(tmp_path: Path) -> Path:
    return _make_outcomes(
        tmp_path,
        scenarios=[("scenario_pass", True), ("scenario_fail", False)],
    )


# ── individual subcommand smokes ──


def test_smoke_version() -> None:
    p = _run("--version")
    assert p.returncode == 0, p.stderr
    assert "belt" in p.stdout.lower(), p.stdout


def test_smoke_doctor_json_is_parseable() -> None:
    p = _run("doctor", "--json")
    # rc may be 1 (no agents configured); we only need valid JSON.
    data = json.loads(p.stdout)
    assert "agents" in data
    assert "llm_providers" in data


def test_smoke_agent_list_runs() -> None:
    p = _run("agent", "list")
    assert p.returncode == 0, p.stderr
    # Agent table is UI; lives on stderr (see ``belt._ui.eprint``). Stdout
    # is reserved for future ``--output`` flags / pipeable views.
    assert "claude-code" in p.stderr
    assert "cursor" in p.stderr


def test_smoke_agent_info_for_each_bundled_agent() -> None:
    """``agent info <name>`` must complete for every entry-point agent."""
    from importlib.metadata import entry_points

    names = sorted(ep.name for ep in entry_points(group="belt.agents"))
    for name in names:
        p = _run("agent", "info", name)
        assert p.returncode == 0, f"agent info {name} failed: {p.stderr}"
        assert name in p.stderr


def test_smoke_eval_dry_run_against_showcase() -> None:
    """Dry-run eval against the bundled showcase must succeed without an agent."""
    p = _run("eval", str(SHOWCASE), "--modes", "rules", "--dry-run")
    assert p.returncode == 0, p.stderr


def test_smoke_run_dry_run_against_showcase() -> None:
    p = _run("run", str(SHOWCASE), "--dry-run")
    assert p.returncode == 0, p.stderr


def test_smoke_aggregate_against_synthetic_outcomes(synth_run: Path) -> None:
    p = _run("aggregate", "--run-dir", str(synth_run))
    assert p.returncode == 0, p.stderr
    assert (synth_run / "results.json").is_file()


def test_smoke_export_against_synthetic_outcomes(synth_run: Path) -> None:
    """``export`` must re-emit a completed run through every built-in exporter."""
    agg = _run("aggregate", "--run-dir", str(synth_run))
    assert agg.returncode == 0, agg.stderr

    exp = _run(
        "export",
        str(synth_run),
        "--to",
        "csv:results.csv",
        "--to",
        "jsonl:results.jsonl",
        "--to",
        "junit:report.xml",
        "--to",
        "markdown:summary.md",
    )
    assert exp.returncode == 0, exp.stderr
    for fname in ("results.csv", "results.jsonl", "report.xml", "summary.md"):
        assert (synth_run / fname).is_file(), f"export produced no {fname}"
    assert (synth_run / "report.xml").read_text().startswith('<?xml version="1.0"')


def test_smoke_view_against_synthetic_outcomes(synth_run: Path) -> None:
    """``view`` browses results - must complete cleanly given a real run dir."""
    # ``view`` needs results.json - produce it via aggregate first.
    aggregate = _run("aggregate", "--run-dir", str(synth_run))
    assert aggregate.returncode == 0, aggregate.stderr
    p = _run("view", str(synth_run))
    assert p.returncode == 0, p.stderr


def test_smoke_compare_two_synthetic_runs(tmp_path: Path) -> None:
    """``compare`` accepts two ``results.json`` paths produced by ``aggregate``."""
    a = _make_outcomes(tmp_path / "a", scenarios=[("s1", True), ("s2", False)])
    b = _make_outcomes(tmp_path / "b", scenarios=[("s1", True), ("s2", True)])
    for r in (a, b):
        rc = _run("aggregate", "--run-dir", str(r)).returncode
        assert rc == 0
    p = _run("compare", str(a / "results.json"), str(b / "results.json"))
    assert p.returncode == 0, p.stderr


def test_smoke_score_against_synthetic_outcomes(synth_run: Path) -> None:
    """``score --modes rules`` against existing outcomes is a no-LLM path."""
    p = _run("score", "--run-dir", str(synth_run), "--modes", "rules")
    # ``score`` may legitimately exit non-zero if the synthetic outcomes don't
    # have the turn_*_output.json layout it expects. The smoke check here is
    # that the binary loaded and produced human-readable output, not a Python
    # traceback. Trace = bug; clean error message = expected.
    combined = (p.stdout + p.stderr).lower()
    assert "traceback" not in combined, f"score crashed instead of erroring cleanly:\n{p.stdout}\n{p.stderr}"


def test_smoke_gc_dry_run_against_outcomes_root(synth_run: Path) -> None:
    """``gc --dry-run`` must inspect outcomes safely without touching anything."""
    outcomes_root = synth_run.parent  # tmp/outcomes/
    p = _run("gc", "--dry-run", "--outcomes-dir", str(outcomes_root))
    assert p.returncode == 0, p.stderr
    # Synthetic data still on disk (dry-run must not delete).
    assert synth_run.is_dir()
    assert (synth_run / "demo" / "scenario_pass" / "score.json").is_file()


def test_smoke_watch_starts_and_exits_cleanly(synth_run: Path) -> None:
    """``watch`` (non-follow mode) must exit on its own when no live data is available."""
    p = _run("watch", str(synth_run))
    assert p.returncode == 0, p.stderr
    combined = (p.stdout + p.stderr).lower()
    assert "traceback" not in combined, f"watch crashed instead of exiting cleanly:\n{p.stdout}\n{p.stderr}"
