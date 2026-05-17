#!/usr/bin/env python3
# (c) JFrog Ltd. (2026)

"""Mechanical enforcement of belt design principles.

Each check below corresponds to a numbered principle in the
architecture doc (path in the ``DOC`` constant). When a check fails,
the message points at the principle's anchor in that document so
contributors can read the rationale and the fix guidance directly -
no internal context required.

Run standalone:  python scripts/check_design.py
Run via hook:    pre-commit run check-design --all-files

Adding a new check:

    1. Add a function ``check_principle_N_<short_name>(errors)``.
    2. Append failures via ``fail(errors, file, line, principle, msg)``.
    3. Wire the function into ``main()`` below.
    4. Update the Design Principles section of the architecture doc
       if the check enforces a previously-implicit rule.

Coverage today: principles 1, 2, 3, 5, 6, 7, 8, plus a repo-wide
copyright-header convention check. Principles 4 (agents-as-plumbing),
9 (one-source-of-truth) and 10 (docs-travel-with-code) are enforced
elsewhere - 9 by ``tests/test_envvars.py``, 10 by the
``test_*_doc_parity.py`` suite, and 4 by code review (no
mechanically-enforceable rule short of full taint analysis).
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "belt"
DOC = "docs/glossary/ARCHITECTURE.md"


# ── Helpers ─────────────────────────────────────────────────────────────


def fail(errors: list[str], file: Path, line: int, principle: int, msg: str) -> None:
    rel = file.relative_to(ROOT) if file.is_absolute() else file
    errors.append(f"  {rel}:{line}: Principle {principle}: {msg}\n" f"    → see {DOC}#principle-{principle}")


def grep(path: Path, pattern: str) -> list[tuple[int, str]]:
    """Return ``(line_number, line)`` pairs in ``path`` matching ``pattern``."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    rx = re.compile(pattern)
    return [(i + 1, line) for i, line in enumerate(text.splitlines()) if rx.search(line)]


def py_files(*subdirs: str) -> list[Path]:
    out: list[Path] = []
    for sub in subdirs:
        d = SRC / sub
        if d.is_dir():
            out.extend(d.rglob("*.py"))
    return out


# ── Principle 1: phase independence ─────────────────────────────────────


def check_principle_1_phase_independence(errors: list[str]) -> None:
    """Runner / scorer / aggregator / exporter must not import each other.

    ``benchmark_card`` is the aggregator's artifact assembler (built in
    ``commands/aggregate.py``) and is treated as part of the aggregator
    phase for the purposes of this check; the runner and scorer must not
    pull it in either, even though it lives outside ``aggregator/``.
    Shared redaction primitives belong in :mod:`belt._redact`.

    ``scorer.payloads`` and ``scorer.entities`` (excluding the
    ``ScorerResult`` writer side) are the *cross-phase data contract* -
    typed shapes the aggregator and exporter read from disk - so they
    are explicitly allowed to be imported from any phase, mirroring the
    role of :mod:`belt.entities`. ``scorer.display`` carries the same
    role for *rendering* attributes (verdict icons / colours) and is
    allowed for the same reason.
    """
    # ``scorer.payloads``, ``scorer.entities`` and ``scorer.display``
    # carry cross-phase data shapes (RulesPayload / LLMPayload / verdict
    # display attrs / etc.) that downstream phases must read; they are
    # the typed contract, not scorer phase logic.
    _SCORER_DATA_MODULES = "(payloads|entities|display)"
    rules = {
        "runner": rf"from belt\.scorer(?!\.{_SCORER_DATA_MODULES})|"
        rf"from belt\.(aggregator|benchmark_card|exporter)",
        "scorer": r"from belt\.(runner|aggregator|benchmark_card|exporter)",
        "aggregator": rf"from belt\.scorer(?!\.{_SCORER_DATA_MODULES})|" rf"from belt\.(runner|exporter)",
        "exporter": rf"from belt\.scorer(?!\.{_SCORER_DATA_MODULES})|"
        rf"from belt\.(runner|aggregator|benchmark_card)",
    }
    for phase, pattern in rules.items():
        for f in py_files(phase):
            for lineno, line in grep(f, pattern):
                fail(
                    errors,
                    f,
                    lineno,
                    1,
                    f"{phase} imports {line.strip()!r} - the four phases must be "
                    f"independent (filesystem-only handoff)",
                )


# ── Principle 2: entities are data, not behaviour ───────────────────────


_ALLOWED_DUNDERS = frozenset({"__init__", "__repr__", "__str__", "__eq__", "__hash__"})
_ALLOWED_ENTITY_DECORATORS = frozenset(
    {
        "validator",
        "model_validator",
        "field_validator",
        "property",
        "staticmethod",
        "classmethod",
        "computed_field",
        "cached_property",
    }
)


def _decorator_name(node: ast.expr) -> str:
    """Best-effort extraction of a decorator's identifier."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return ""


def check_principle_2_entities_no_logic(errors: list[str]) -> None:
    entities_py = SRC / "entities.py"
    if not entities_py.exists():
        return
    try:
        tree = ast.parse(entities_py.read_text(encoding="utf-8"))
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for member in node.body:
            if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            name = member.name
            if name.startswith("_") or name.startswith("model_") or name in _ALLOWED_DUNDERS:
                continue
            decorators = {_decorator_name(d) for d in member.decorator_list}
            if decorators & _ALLOWED_ENTITY_DECORATORS:
                continue
            fail(
                errors,
                entities_py,
                member.lineno,
                2,
                f"{node.name}.{name}() looks like business logic - entities are data structures, "
                f"not behaviour containers (allowed: validators, properties, model_*, dunders)",
            )


# ── Principle 3: base interface is closed to modification ───────────────


_BASE_SIGNATURES: dict[str, tuple[str, ...]] = {
    "execute": ("self", "message", "flags"),
    "fetch_results": ("self", "raw_output"),
}


def check_principle_3_base_signatures(errors: list[str]) -> None:
    base_py = SRC / "agent" / "base.py"
    if not base_py.exists():
        return
    try:
        tree = ast.parse(base_py.read_text(encoding="utf-8"))
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == "BaseAgentAdapter"):
            continue
        for member in node.body:
            if not isinstance(member, ast.FunctionDef):
                continue
            expected = _BASE_SIGNATURES.get(member.name)
            if expected is None:
                continue
            actual = tuple(a.arg for a in member.args.args)
            if actual != expected:
                fail(
                    errors,
                    base_py,
                    member.lineno,
                    3,
                    f"BaseAgentAdapter.{member.name}() signature is {actual!r}, "
                    f"must be {expected!r} - adding agent-specific params to the base "
                    f"breaks every downstream agent",
                )


# ── Principle 5: framework code does not branch on concrete agents ─────


_AGENT_ISINSTANCE = re.compile(r"isinstance\s*\([^)]*?\b\w*AgentAdapter\b")


def check_principle_5_no_isinstance_on_agents(errors: list[str]) -> None:
    """Framework code (runner/scorer/aggregator) must not ``isinstance``-check
    concrete ``*AgentAdapter`` classes. Use optional ``TurnOutput`` fields
    with safe defaults instead."""
    for f in py_files("runner", "scorer", "aggregator"):
        for lineno, line in grep(f, _AGENT_ISINSTANCE.pattern):
            fail(
                errors,
                f,
                lineno,
                5,
                "framework code uses isinstance(*, *AgentAdapter) - branching on concrete "
                "agents leaks plumbing into the framework; use optional TurnOutput fields "
                "with null-tolerant rule scorers instead",
            )


# ── Principle 7: callable main() returns an int, never sys.exit() ──────


def check_principle_7_no_sys_exit_in_callable_main(errors: list[str]) -> None:
    for f in py_files("commands", "runner", "scorer", "aggregator"):
        if f.name == "__main__.py":
            continue
        text = f.read_text(encoding="utf-8")
        lines = text.splitlines()
        for lineno, _ in grep(f, r"\bsys\.exit\s*\("):
            preceding = lines[max(0, lineno - 6) : lineno - 1]
            if any("__name__" in pl and "__main__" in pl for pl in preceding):
                continue
            fail(
                errors,
                f,
                lineno,
                7,
                "sys.exit() in a callable main() - return an int exit code instead so "
                "`belt eval` can chain run → score → aggregate without short-circuiting",
            )


# ── Principle 8: untrusted strings must pass through _safe.py helpers ──


# Attribute names whose values originate from agent stdout or LLM judge output.
# Conservative allowlist: extend only when adding a new attacker-controlled field.
_UNTRUSTED_ATTRS: tuple[str, ...] = (
    "reply_text",
    "reasoning",
    "agent_output",
    "agent_stdout",
    "raw_output",
    "judge_reasoning",
)
# Sinks whose first argument or appended item interprets formatting markup.
_MARKUP_SINKS: tuple[str, ...] = (
    "console.print",
    "lines.append",
    "summary.append",
    "panel_lines.append",
    "Text.from_markup",
    "Markdown(",
)


def check_principle_8_output_escaping(errors: list[str]) -> None:
    """In aggregator render sinks, untrusted attributes must be wrapped in
    ``rich_safe``/``md_safe`` before reaching markup-aware output."""
    for f in (SRC / "aggregator").glob("render_*.py"):
        text = f.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if not any(sink in line for sink in _MARKUP_SINKS):
                continue
            for attr in _UNTRUSTED_ATTRS:
                pattern = rf"\{{[^{{}}]*?\.{attr}[^{{}}]*?\}}"
                for m in re.finditer(pattern, line):
                    expr = m.group(0)
                    if "rich_safe" in expr or "md_safe" in expr:
                        continue
                    fail(
                        errors,
                        f,
                        lineno,
                        8,
                        f"untrusted '.{attr}' interpolated into a markup sink "
                        f"({expr!r}) without rich_safe()/md_safe() - wrap it",
                    )


# ── Principle 8 (cont.): every --allow-* flag is default-deny ──────────


def check_principle_8_default_deny(errors: list[str]) -> None:
    """Every ``add_argument("--allow-*", ...)`` must default to False. Any
    other default would silently lower a security boundary for users who
    never opt in. The escape sibling of Principle 8."""
    for f in SRC.rglob("*.py"):
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not (isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"):
                continue
            flags = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            allow_flag = next((flag for flag in flags if flag.startswith("--allow-")), None)
            if allow_flag is None:
                continue
            kwargs = {kw.arg: kw.value for kw in node.keywords}
            action = kwargs.get("action")
            default = kwargs.get("default")
            # action="store_true" implies default=False
            if isinstance(action, ast.Constant) and action.value == "store_true":
                continue
            if isinstance(default, ast.Constant) and default.value in (False, None):
                continue
            fail(
                errors,
                f,
                node.lineno,
                8,
                f"{allow_flag!r} must be default-deny - declare action='store_true' "
                f"(or default=False) so users never silently inherit the lowered boundary",
            )


# ── Principle 6: plugins import only from the public belt API ────


_PLUGIN_ROOTS: tuple[str, ...] = ("plugins", "examples/custom-agent")


def _walk_plugin_py_files() -> list[Path]:
    out: list[Path] = []
    for rel in _PLUGIN_ROOTS:
        root = ROOT / rel
        if not root.is_dir():
            continue
        out.extend(p for p in root.rglob("*.py") if not any(part.endswith(".egg-info") for part in p.parts))
    return out


def check_principle_6_plugin_public_api_only(errors: list[str]) -> None:
    """Plugin code must import belt symbols only via the top-level package.

    The published public API lives in :mod:`belt._public_api` and is
    re-exported lazily from ``belt/__init__.py``. Plugins (under
    ``plugins/`` and the ``examples/custom-agent/`` reference) write
    ``from belt import BaseExporter`` and never reach into internal
    paths like ``from belt.exporter.base import BaseExporter``.

    Both forms are flagged:

    * ``from belt.<sub> import X``
    * ``import belt.<sub>`` (or ``import belt.<sub> as alias``)

    Bare ``from belt import X`` and bare ``import belt`` pass.
    Internal helpers needed by a plugin must first be added to
    :data:`belt._public_api.PUBLIC_API` - that is what makes them
    part of the published contract.
    """
    try:
        from belt._public_api import PUBLIC_API  # noqa: F401  (used by error msg)
    except ImportError:
        # Package not importable from this checkout (rare); skip rather
        # than block the whole design check.
        return
    for f in _walk_plugin_py_files():
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod == "belt" or not mod.startswith("belt."):
                    continue
                names = ", ".join(alias.name for alias in node.names)
                fail(
                    errors,
                    f,
                    node.lineno,
                    6,
                    f"plugin imports {names!r} from internal module {mod!r}; "
                    f"use 'from belt import {names}' instead "
                    f"(extend belt._public_api.PUBLIC_API if a needed symbol is missing)",
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("belt."):
                        fail(
                            errors,
                            f,
                            node.lineno,
                            6,
                            f"plugin imports internal module {alias.name!r}; "
                            f"use 'from belt import <symbol>' instead "
                            f"(extend belt._public_api.PUBLIC_API if a needed symbol is missing)",
                        )


# ── Principle 6 (cont.): plugins iterate scorer payloads via the public API ─


_DIRECT_SCORE_INDEX = re.compile(
    r"\.\s*scores\s*(\.\s*get\s*\(\s*['\"](rules|llm)['\"]|\[\s*['\"](rules|llm)['\"]\s*\])"
)


def check_principle_6_no_direct_score_dict_access_in_plugins(errors: list[str]) -> None:
    """Plugin code must not literal-index ``score.scores[\"rules\"]/[\"llm\"]``.

    The two built-in scorer keys are an implementation detail of this
    repo's bundled scorers. Hard-coding them in a plugin (a) breaks any
    third-party scorer that registers a different key, and (b) bypasses
    :func:`belt.iter_dimension_feedback` - the public helper that
    handles every payload shape uniformly. Plugins reach into typed
    payload attributes (``payload.checks`` on
    :class:`belt.RulesPayload`) only after a typed
    ``isinstance`` check, never via stringly-typed lookups.

    The check is intentionally narrow: it flags only literal
    ``"rules"`` / ``"llm"`` string keys. Plugins that legitimately
    iterate ``score.scores.items()`` for cross-scorer work pass
    cleanly.
    """
    for f in _walk_plugin_py_files():
        try:
            text = f.read_text(encoding="utf-8")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            if _DIRECT_SCORE_INDEX.search(line):
                fail(
                    errors,
                    f,
                    lineno,
                    6,
                    "plugin literal-indexes score.scores['rules'/'llm'] - use "
                    "belt.iter_dimension_feedback(score) for cross-scorer iteration, "
                    "or guard typed payload access with isinstance(..., RulesPayload/LLMPayload)",
                )


# ── Convention: copyright header on every source file ─────────────────


_HEADER = "# (c) JFrog Ltd. (2026)"
# Roots that contain hand-authored Python source. Each .py under these
# (recursively) must start with the canonical header on line 1 OR line 2
# (line 1 may be a shebang).
_HEADER_ROOTS: tuple[str, ...] = ("src", "tests", "scripts", "examples", "plugins")
# Files that are auto-generated, vendored, or otherwise outside the
# hand-authored set. Paths are relative to ``ROOT``.
_HEADER_EXEMPT: frozenset[str] = frozenset()
# Directory names that disqualify any path containing them from the
# hand-authored set: virtualenvs, bytecode caches, and tool caches that
# can materialise inside a ``_HEADER_ROOTS`` entry (e.g. ``plugins/<name>/
# .venv/``).
_HEADER_SKIP_DIR_NAMES: frozenset[str] = frozenset(
    {".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".tox", "node_modules"}
)


def check_copyright_header(errors: list[str]) -> None:
    """Every hand-authored Python file must declare the JFrog Ltd. copyright
    header. This is JFrog's OSS convention and pairs with the Apache-2.0
    LICENSE at the repo root."""
    for root in _HEADER_ROOTS:
        root_dir = ROOT / root
        if not root_dir.is_dir():
            continue
        for f in root_dir.rglob("*.py"):
            if any(part in _HEADER_SKIP_DIR_NAMES for part in f.parts):
                continue
            rel = f.relative_to(ROOT).as_posix()
            if rel in _HEADER_EXEMPT:
                continue
            try:
                lines = f.read_text(encoding="utf-8").splitlines()[:2]
            except OSError:
                continue
            if any(line.strip() == _HEADER for line in lines):
                continue
            rel_path = f.relative_to(ROOT) if f.is_absolute() else f
            errors.append(
                f"  {rel_path}:1: Convention: copyright header missing - every "
                f"hand-authored .py under {list(_HEADER_ROOTS)} must start with "
                f"{_HEADER!r} on line 1 or 2 (line 1 may be a shebang). "
                f"Auto-generated files belong in `_HEADER_EXEMPT` in scripts/check_design.py."
            )


# ── Entry point ────────────────────────────────────────────────────────


_CHECKS = (
    check_principle_1_phase_independence,
    check_principle_2_entities_no_logic,
    check_principle_3_base_signatures,
    check_principle_5_no_isinstance_on_agents,
    check_principle_6_plugin_public_api_only,
    check_principle_6_no_direct_score_dict_access_in_plugins,
    check_principle_7_no_sys_exit_in_callable_main,
    check_principle_8_output_escaping,
    check_principle_8_default_deny,
    check_copyright_header,
)


def main() -> int:
    errors: list[str] = []
    for check in _CHECKS:
        check(errors)
    if errors:
        print("Design principle violations found:\n")
        for e in errors:
            print(e)
        print(f"\n{len(errors)} violation(s). See {DOC} for full rationale and fix guidance.")
        return 1
    print(f"Design principles: OK ({len(_CHECKS)} checks ran)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
