# (c) JFrog Ltd. (2026)

"""AST gate: no inline ``re.*`` calls on user-supplied scenario regexes.

The only module allowed to compile / search / match a user-supplied
scenario regex is :mod:`belt._regex_policy`. Every other module
consumes the cached :class:`re.Pattern` objects populated by
:meth:`belt.scenario.TurnExpectation.model_post_init`.

Without this gate, a future contributor could "helpfully" reintroduce
``re.search(expect.reply_pattern[i], ...)`` in a hot path - and with it
the silent-``False``-on-bad-regex behaviour the policy module exists to
prevent. The gate flags two shapes:

* Direct attribute / subscript access in the call: ``re.search(expect.reply_pattern, ...)``
  or ``re.compile(expect.tool_result_pattern[name])``.
* Local-name proxies: ``p = expect.reply_pattern[0]; re.search(p, ...)``
  - tracked per function via a single-pass assignment walker.

The gate is intentionally narrow: it does not try to track tainted
values across function boundaries. The cross-boundary case is covered
by Code Review + the test_design / phase-isolation gates: scenario
data lives on :class:`TurnExpectation` and that type does not leak its
field strings into framework code outside the scorer rules.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src" / "belt"

# Field names on ``TurnExpectation`` (and the corresponding cached
# ``PrivateAttr``s) that hold user-supplied regex source. Extend if a
# new regex-bearing scenario field lands.
_FORBIDDEN_TOKENS = (
    "reply_pattern",
    "tool_result_pattern",
    "_compiled_reply_patterns",
    "_compiled_tool_patterns",
)

_RE_FUNCS = frozenset({"compile", "search", "match", "fullmatch", "findall", "sub", "subn"})

# The single module allowed to call ``re.*`` directly on user input.
_EXEMPT_MODULES: frozenset[str] = frozenset({"_regex_policy.py"})


def _is_re_call(node: ast.Call) -> bool:
    """Detect ``re.<func>(...)`` (excludes ``re.escape`` / ``re.purge`` etc.)."""
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "re"
        and func.attr in _RE_FUNCS
    )


def _expr_mentions_forbidden(expr: ast.expr) -> bool:
    """``ast.unparse``-based check: does the expression text contain a forbidden token."""
    try:
        text = ast.unparse(expr)
    except Exception:
        return False
    return any(tok in text for tok in _FORBIDDEN_TOKENS)


class _UserRegexGate(ast.NodeVisitor):
    """Walks every function and flags violations.

    Per-function, builds a map of local names -> last RHS expression
    (as text) so a local-name proxy like ``p = expect.reply_pattern[0]``
    can be detected when ``p`` later appears as the first arg of
    ``re.search``.
    """

    def __init__(self, file: Path) -> None:
        self.file = file
        self.violations: list[tuple[int, str]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._walk_scope(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._walk_scope(node)
        self.generic_visit(node)

    def visit_Module(self, node: ast.Module) -> None:
        self._walk_scope(node)

    def _walk_scope(self, scope: ast.AST) -> None:
        # Names assigned from a forbidden expression: tracked so a later
        # re.X call on the bare name flags the same way the direct attr
        # access would.
        tainted: set[str] = set()
        for child in ast.walk(scope):
            if isinstance(child, ast.Assign):
                if _expr_mentions_forbidden(child.value):
                    for tgt in child.targets:
                        if isinstance(tgt, ast.Name):
                            tainted.add(tgt.id)
                continue
            if isinstance(child, ast.For) and _expr_mentions_forbidden(child.iter):
                if isinstance(child.target, ast.Name):
                    tainted.add(child.target.id)
                # ``for tool, p in expect.tool_result_pattern.items():``
                elif isinstance(child.target, ast.Tuple):
                    for elt in child.target.elts:
                        if isinstance(elt, ast.Name):
                            tainted.add(elt.id)
                continue
            if isinstance(child, ast.Call) and _is_re_call(child) and child.args:
                first = child.args[0]
                if _expr_mentions_forbidden(first):
                    self.violations.append(
                        (child.lineno, f"re.{child.func.attr}({ast.unparse(first)}) - direct user-regex access")
                    )
                    continue
                if isinstance(first, ast.Name) and first.id in tainted:
                    self.violations.append(
                        (
                            child.lineno,
                            f"re.{child.func.attr}({first.id}) - {first.id!r} was bound from a "
                            f"forbidden user-regex source earlier in this scope",
                        )
                    )


def _scan_file(file: Path) -> list[tuple[int, str]]:
    try:
        tree = ast.parse(file.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return []
    gate = _UserRegexGate(file)
    gate.visit(tree)
    return gate.violations


def test_no_inline_user_regex_calls_outside_policy_module() -> None:
    """Every ``re.*`` call on user-supplied scenario regexes must route through
    :mod:`belt._regex_policy`. The policy module is the only place flags and
    error semantics for user input are defined."""
    failures: list[str] = []
    for f in SRC.rglob("*.py"):
        if f.name in _EXEMPT_MODULES:
            continue
        for line, msg in _scan_file(f):
            rel = f.relative_to(REPO_ROOT)
            failures.append(f"  {rel}:{line}: {msg}")
    assert not failures, (
        "Inline ``re.*`` calls on user-supplied scenario regexes detected.\n"
        "Route through ``belt._regex_policy.compile_user_regex`` (or consume the cached\n"
        "``re.Pattern`` from ``TurnExpectation._compiled_*`` populated by ``model_post_init``).\n"
        "\n" + "\n".join(failures)
    )
