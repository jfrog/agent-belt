# (c) JFrog Ltd. (2026)

"""Argparse ordering invariants for every public CLI module.

Within each ``argparse.ArgumentParser`` (or ``add_argument_group(...)`` block)
the ``add_argument(...)`` calls must be ordered alphabetically by long flag
name. Positional arguments precede flags and keep their declaration order.

Why AST-based instead of runtime introspection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Runtime introspection (``parser._actions``) cannot tell you what *source
order* the developer wrote, only what argparse has been told. AST
analysis reads the actual ``add_argument`` calls in source order. That's
what we want to enforce: when a contributor opens a CLI module, the
flags should appear alphabetically so drift is visible in PR diffs.

Out of scope
~~~~~~~~~~~~
* Sub-parsers built dynamically from registries (none exist today).
* Parsers composed via ``parents=``: not used in this repo, but if added
  the test will treat each parent's flags as a separate group.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "belt"


def _cli_files() -> list[Path]:
    """Return every Python file with at least one ``add_argument`` call.

    We don't hardcode a list because the surface grows; rg-style discovery
    keeps the test honest.
    """
    files: list[Path] = []
    for py in _SRC_ROOT.rglob("*.py"):
        if py.name.startswith("_"):
            continue
        if "add_argument" in py.read_text(encoding="utf-8"):
            files.append(py)
    return sorted(files)


def _long_flag(call: ast.Call) -> str | None:
    """Return the long flag name for an ``add_argument(...)`` call.

    A long flag is the first positional argument that starts with ``--``.
    Positional argparse arguments (no leading ``-``) return ``None`` so the
    caller can keep them in declaration order.
    """
    for arg in call.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            if arg.value.startswith("--"):
                return arg.value
            if arg.value.startswith("-"):
                # short flag like "-X"; if no long flag follows, this is
                # the canonical name. Continue scanning so a long flag
                # would still take precedence.
                continue
            # Bare positional - no long flag.
            return None
    # ``add_argument()`` with only kwargs (rare) - skip ordering for it.
    return None


def _walk_argparse_blocks(tree: ast.Module) -> list[tuple[str, list[tuple[ast.Call, str | None]]]]:
    """Yield (block_label, calls) per logical argparse block in source order.

    A block boundary is created by:

    * ``argparse.ArgumentParser(...)`` (a fresh parser starts a new block)
    * ``parser.add_argument_group(...)`` (a sub-group is its own block)

    ``calls`` is the list of ``add_argument`` ``ast.Call`` nodes inside
    that block, in source order, paired with their long-flag name (or
    ``None`` for positionals).
    """
    blocks: list[tuple[str, list[tuple[ast.Call, str | None]]]] = []
    block_by_var: dict[str, list[tuple[ast.Call, str | None]]] = {}
    block_label_by_var: dict[str, str] = {}

    def _ensure_block(var: str, label: str) -> None:
        if var not in block_by_var:
            block_by_var[var] = []
            block_label_by_var[var] = label
            blocks.append((label, block_by_var[var]))

    for node in ast.walk(tree):
        # Match: ``parser = argparse.ArgumentParser(...)`` etc. We don't
        # need to capture the constructor call itself; we only need to
        # know that the variable now refers to a fresh parser so its
        # add_argument calls form a block.
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
            func = call.func
            if isinstance(func, ast.Attribute):
                # ``argparse.ArgumentParser(...)``  → top-level parser
                if func.attr == "ArgumentParser":
                    for tgt in node.targets:
                        if isinstance(tgt, ast.Name):
                            _ensure_block(tgt.id, f"{tgt.id} (parser)")
                # ``parser.add_argument_group(...)`` → group
                elif func.attr == "add_argument_group":
                    title = "<unnamed>"
                    if call.args and isinstance(call.args[0], ast.Constant):
                        title = str(call.args[0].value)
                    for tgt in node.targets:
                        if isinstance(tgt, ast.Name):
                            _ensure_block(tgt.id, f"group: {title}")

    # Second pass: collect ``<var>.add_argument(...)`` calls per known var.
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and isinstance(node.func.value, ast.Name)
        ):
            var = node.func.value.id
            if var in block_by_var:
                block_by_var[var].append((node, _long_flag(node)))

    return blocks


def _assert_block_sorted(file: Path, label: str, calls: list[tuple[ast.Call, str | None]]) -> None:
    """Assert long flags within ``calls`` are sorted alphabetically.

    Positionals (long flag is ``None``) are ignored for the comparison;
    they keep their declaration order, which argparse also requires.
    """
    flags = [name for _, name in calls if name is not None]
    if flags == sorted(flags):
        return
    expected = sorted(flags)
    diffs = []
    for got, want in zip(flags, expected, strict=False):
        if got != want:
            diffs.append(f"  saw {got!r}, expected {want!r}")
    pytest.fail(
        f"{file.relative_to(_SRC_ROOT.parent)}: "
        f"add_argument calls in '{label}' are not alphabetised by long flag.\n"
        f"  source order:   {flags}\n"
        f"  expected order: {expected}\n" + "\n".join(diffs)
    )


class TestArgparseOrdering:
    """Each parser / argument-group block has alphabetic long flags."""

    @pytest.mark.parametrize("file", _cli_files(), ids=lambda p: str(p.relative_to(_SRC_ROOT.parent)))
    def test_long_flags_alphabetised_per_block(self, file: Path) -> None:
        tree = ast.parse(file.read_text(encoding="utf-8"))
        blocks = _walk_argparse_blocks(tree)
        if not blocks:
            pytest.skip(f"{file.name}: no argparse blocks discovered")
        for label, calls in blocks:
            _assert_block_sorted(file, label, calls)
