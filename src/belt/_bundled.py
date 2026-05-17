# (c) JFrog Ltd. (2026)

"""Resolve scenarios shipped inside the installed ``agent-belt`` wheel.

After ``pip install agent-belt`` the user has no on-disk copy of the
``examples/scenarios/`` tree, so commands that referenced it by relative
path (``belt eval examples/scenarios/showcase``) silently failed -- the
broken first-run reported in #385 (1.1, 1.5).

Wheels force-include ``examples/scenarios/showcase`` and
``examples/fixtures/sample-project`` under
``belt/_bundled_examples/`` (see ``pyproject.toml``). This module is the
single source of truth for locating that tree, so the CLI and the
quickstart both ask the same question instead of each rolling its own
``importlib.resources`` lookup.

Editable installs and source checkouts fall back to the in-tree
``examples/`` directory; that branch keeps behaviour stable for
contributors without forcing them to build a wheel locally.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

# The on-wheel prefix used by the hatchling ``force-include`` mapping. A
# constant rather than an inline literal so a future rename only touches
# one spot.
_BUNDLED_PREFIX = "_bundled_examples"


def bundled_scenarios_root() -> Path | None:
    """Return the on-disk path to the bundled ``scenarios/`` root, or ``None``.

    Resolution order (first hit wins):

    1. ``belt/_bundled_examples/scenarios/`` inside the installed wheel.
    2. ``<repo>/examples/scenarios/`` for editable installs / source checkouts.
    3. ``<cwd>/examples/scenarios/`` when invoked from a clone whose
       package was installed non-editably (``pip install .`` from
       repo root).

    Returns ``None`` when no copy of the showcase tree is reachable -
    callers should surface a clear error pointing at ``pip install
    agent-belt`` or ``--scenarios <path>`` rather than crashing.
    """
    try:
        root = Path(str(files("belt") / _BUNDLED_PREFIX / "scenarios"))
        if (root / "showcase").is_dir():
            return root
    except (ModuleNotFoundError, FileNotFoundError, TypeError):
        pass

    candidates = [
        Path(__file__).resolve().parent.parent.parent / "examples" / "scenarios",
        Path.cwd() / "examples" / "scenarios",
    ]
    for candidate in candidates:
        if (candidate / "showcase").is_dir():
            return candidate
    return None


def resolve_bundled_path(subpath: str | None) -> Path | None:
    """Map a ``--bundled <NAME>`` argument to the scenarios path on disk.

    ``subpath`` may be:

    - ``None`` or ``""`` - point at the scenarios root itself
      (``--bundled`` with no argument).
    - A group name like ``showcase`` or a deeper path like
      ``showcase/correctness`` - resolved relative to the scenarios root.

    Returns ``None`` if the bundled tree can't be located or the
    requested subpath does not exist. The caller produces the error
    message; this helper only reports "not found".
    """
    root = bundled_scenarios_root()
    if root is None:
        return None
    if not subpath:
        return root
    target = (root / subpath).resolve()
    if not target.exists():
        return None
    # Defence-in-depth: ``--bundled ../../etc/passwd`` would otherwise
    # escape the bundled root. ``resolve()`` then ``is_relative_to``
    # confines the resolved path to the shipped tree.
    if not target.is_relative_to(root.resolve()):
        return None
    return target


def bundled_groups() -> list[str]:
    """List the group directories visible under the bundled scenarios root.

    Empty list when the bundled tree is unreachable. Used by the CLI to
    enumerate valid ``--bundled <NAME>`` values in error messages.
    """
    root = bundled_scenarios_root()
    if root is None:
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("_"))
