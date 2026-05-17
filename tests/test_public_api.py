# (c) JFrog Ltd. (2026)

"""Tests for the top-level :mod:`belt` public Python API.

The public API is the single thing plugin authors are guaranteed to be able
to import. These tests pin three properties:

* every name in :data:`belt._public_api.PUBLIC_API` resolves and is
  the *same object* as what its declared internal module exposes - no
  silent drift if someone moves a class without updating the mapping;
* unrelated attributes raise :class:`AttributeError` so typos surface
  immediately at import time rather than as ``None`` references later;
* :func:`dir` reports the public API surface so editor autocomplete
  picks up the lazy re-exports.

The eager ``importlib.import_module`` calls below are deliberate - the
whole point of lazy ``__getattr__`` is that those modules do *not* load
until something touches the public name. The test cross-checks both
sides explicitly to detect any divergence.
"""

from __future__ import annotations

import importlib

import pytest

import belt
from belt._public_api import PUBLIC_API


@pytest.mark.parametrize("name,module_path", sorted(PUBLIC_API.items()))
def test_public_symbol_resolves_to_internal_definition(name: str, module_path: str) -> None:
    """Every PUBLIC_API entry must yield the same object as the internal module's attribute."""
    public = getattr(belt, name)
    internal_module = importlib.import_module(module_path)
    assert public is getattr(internal_module, name)


def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError):
        belt.NotARealSymbol  # noqa: B018


def test_dir_lists_public_api() -> None:
    listed = set(dir(belt))
    assert set(PUBLIC_API).issubset(listed)
    assert "__version__" in listed


def test_all_matches_public_api() -> None:
    assert set(belt.__all__) == {"__version__", *PUBLIC_API}


def test_version_is_string() -> None:
    assert isinstance(belt.__version__, str)
    assert belt.__version__  # non-empty


def test_version_falls_back_when_package_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the ``PackageNotFoundError`` branch in ``belt/__init__.py``.

    On a fresh source clone (``git clone`` then ``python -c "import belt"``
    without ``pip install``), ``importlib.metadata.version("agent-belt")``
    raises ``PackageNotFoundError`` because the wheel's ``.dist-info`` is
    not on ``sys.path``. The package must fall back to a placeholder rather
    than crash on import - otherwise the public API is unimportable from a
    fresh clone, which would silently break new-contributor onboarding.
    """
    import importlib
    import importlib.metadata as md

    def raise_pnf(name: str) -> str:
        raise md.PackageNotFoundError(name)

    monkeypatch.setattr(md, "version", raise_pnf)
    importlib.reload(belt)
    try:
        assert belt.__version__ == "0.0.0+unknown"
    finally:
        # Restore the real ``__version__`` so subsequent tests in the same
        # process see the installed version, not the placeholder. ``monkeypatch``
        # undoes the metadata patch automatically; we just need to re-trigger
        # the module-level ``try`` block with the real ``version`` function.
        importlib.reload(belt)
