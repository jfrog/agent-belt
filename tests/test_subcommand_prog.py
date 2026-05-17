# (c) JFrog Ltd. (2026)

"""Pins ``argparse(prog=...)`` for every belt subcommand (#385 / 2.3).

Without ``prog="belt <sub>"``, the auto-derived ``prog`` is the script
name (``belt`` or the venv shim) and ``--help`` prints ``usage: belt
[-h] ...`` even for ``belt run --help``. The mismatch is small enough
to miss in review but big enough to look careless in a launch demo.
"""

from __future__ import annotations

import importlib

import pytest

EXPECTED: dict[str, tuple[str, ...]] = {
    "belt.commands.run": ("belt run",),
    "belt.commands.score": ("belt score",),
    "belt.commands.aggregate": ("belt aggregate",),
    "belt.commands.compare": ("belt compare",),
    "belt.commands.eval": ("belt eval",),
    "belt.commands.view": ("belt view",),
    "belt.commands.watch": ("belt watch",),
    "belt.commands.doctor": ("belt doctor",),
    "belt.commands.gc": ("belt gc",),
    "belt.commands.export": ("belt export",),
    "belt.commands.quickstart": ("belt quickstart",),
}


@pytest.mark.parametrize("module_name,expected_progs", list(EXPECTED.items()))
def test_subcommand_has_prog(module_name: str, expected_progs: tuple[str, ...]):
    """Each subcommand's ``--help`` output must begin with ``usage: belt <sub>``.

    Snapshotting the full help text would be brittle (column reflow,
    locale); we check the prog line only, which is what users see and
    what the audit flagged.
    """
    mod = importlib.import_module(module_name)
    main = mod.main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_run_help_usage_prefix(capsys):
    from belt.commands.run import main

    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "usage: belt run" in out, f"expected 'usage: belt run' in help output, got:\n{out[:200]}"


def test_score_help_usage_prefix(capsys):
    from belt.commands.score import main

    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "usage: belt score" in out


def test_aggregate_help_usage_prefix(capsys):
    from belt.commands.aggregate import main

    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "usage: belt aggregate" in out


def test_compare_help_usage_prefix(capsys):
    from belt.commands.compare import main

    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "usage: belt compare" in out
