# (c) JFrog Ltd. (2026)

"""belt - evaluation harness for real headless CLI agents.

The package re-exports a curated public Python API for plugin authors.
Plugins should write ``from belt import BaseExporter, ExportContext``
rather than reaching into internal modules like ``belt.exporter.base``;
the ``scripts/check_design.py`` plugin-import check enforces this for code
under ``plugins/`` and ``examples/custom-agent/``.

The mapping of public names to their defining modules lives in
:mod:`belt._public_api`. ``__getattr__`` resolves names lazily on first
access (PEP 562) so importing the top-level package stays cheap - none of
the heavy submodules (runner, scorer, exporter, agent adapters) load until
a plugin actually touches a public symbol.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Any

from belt._public_api import PUBLIC_API

# Read ``__version__`` from the installed wheel's metadata rather than a
# generated ``_version.py`` file. ``hatch-vcs`` writes the version into the
# wheel's ``.dist-info/METADATA`` at build time; ``importlib.metadata`` reads
# it back at runtime. This is the same pattern used by ruff, uv, hatch, attrs,
# and packaging - one canonical source, no generated file in the source tree.
try:
    __version__ = _pkg_version("agent-belt")
except PackageNotFoundError:
    # Source checkout that has never been installed. Rare; only shows up for
    # ``python -c "import belt"`` from a fresh clone before ``pip install``.
    __version__ = "0.0.0+unknown"


def __getattr__(name: str) -> Any:
    module_path = PUBLIC_API.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return ["__version__", *sorted(PUBLIC_API)]


__all__ = ["__version__", *sorted(PUBLIC_API)]
