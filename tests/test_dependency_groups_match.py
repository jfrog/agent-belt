# (c) JFrog Ltd. (2026)

"""Drift check: ``[project.optional-dependencies] dev`` (PEP 621, read by
``pip install -e ".[dev]"``) and ``[dependency-groups] dev`` (PEP 735, read
by ``uv sync``) must list the identical dev dependency set.

We can't drop either today: pip's ``--group`` flag for PEP 735 is not yet
stable, so pip-only contributors still rely on ``[project.optional-dependencies]``.
``uv sync`` reads ``[dependency-groups]`` natively. Both sources must therefore
be kept in sync by hand - and ``uv sync --locked`` does NOT enforce equality
between the two sections (it only catches drift between ``pyproject.toml`` and
``uv.lock``). This test is the missing drift gate.

A contributor who adds ``mypy`` to one section but not the other will fail this
test in CI rather than discover the inconsistency in production.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_pep621_and_pep735_dev_sets_are_identical() -> None:
    pyproject = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    pep621 = set(pyproject["project"]["optional-dependencies"]["dev"])
    pep735 = set(pyproject["dependency-groups"]["dev"])
    pip_only = pep621 - pep735
    uv_only = pep735 - pep621
    assert pep621 == pep735, (
        "[project.optional-dependencies] dev and [dependency-groups] dev have "
        f"drifted.\n  Present only via pip path (project.optional-dependencies): {sorted(pip_only)}\n"
        f"  Present only via uv path (dependency-groups): {sorted(uv_only)}\n"
        "Update both sections together (or neither) so pip and uv contributors "
        "install the same dev dependency set."
    )
