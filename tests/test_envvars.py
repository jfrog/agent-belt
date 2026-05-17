# (c) JFrog Ltd. (2026)

"""Centralisation, parity, and discipline invariants for the env-var registry.

Every ``BELT_*`` and ``_BELT_*`` name has exactly one source of
truth - ``belt.envvars`` for the public surface, ``belt._internal_envvars``
for internal forwarding. The literal must not appear anywhere else in
``src/``. These tests enforce both centralisation *and* parity with the
CLI surface and with the user-facing documentation: a typo or drift
between layers is caught here rather than at runtime.

What is enforced
~~~~~~~~~~~~~~~~
1. **Constant hygiene.** Every ``Final[str]`` constant in
   ``belt.envvars`` resolves to the ``BELT_*`` name advertised
   by its identifier (no copy-paste typos). Same invariant for the
   private ``belt._internal_envvars`` module.
2. **No inline literals.** Every ``"BELT_..."`` string in ``src/`` is
   reachable from ``envvars.ALL_NAMES``; every ``"_BELT_..."``
   string is reachable from ``_internal_envvars.ALL_INTERNAL_NAMES``.
3. **Allow-list correctness.** ``_redact.safe_environ`` is sourced
   from ``envvars.PUBLIC_ALLOW`` (single owner).
4. **Sorted registries.** ``ALL_NAMES`` and ``PUBLIC_ALLOW`` are
   alphabetically sorted so drift is visible in PR diffs.
5. **Flag-vs-env parity.** Every ``BELT_ALLOW_*`` constant has at
   least one matching ``--allow-*`` ``add_argument`` call in ``src/``.
6. **Doc coverage.** Every ``BELT_*`` constant appears verbatim in
   ``docs/glossary/CONFIGURATION.md`` so users can search for and
   discover every var the framework reads.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import pytest

from belt import _internal_envvars, envvars

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "belt"
_REPO_ROOT = _SRC_ROOT.parent.parent
_NAME_RE = re.compile(r'"(BELT_[A-Z0-9_]+)"')
_INTERNAL_NAME_RE = re.compile(r'"(_BELT_[A-Z0-9_]+)"')


# ── Section 1: constant hygiene ─────────────────────────────────────────────


class TestEnvvarsConstants:
    def test_constants_match_identifiers(self):
        """Every ``Final[str]`` value equals ``"BELT_" + identifier``."""
        for name in dir(envvars):
            if name.startswith("_") or name.upper() != name:
                continue
            value = getattr(envvars, name)
            if not isinstance(value, str) or not value.startswith("BELT_"):
                continue
            expected = f"BELT_{name}"
            assert value == expected, f"envvars.{name} = {value!r}, expected {expected!r}"

    def test_all_names_covers_every_constant(self):
        """``ALL_NAMES`` must enumerate every public ``Final[str]`` constant."""
        public_constants = {
            getattr(envvars, name)
            for name in dir(envvars)
            if name.upper() == name
            and not name.startswith("_")
            and isinstance(getattr(envvars, name), str)
            and getattr(envvars, name).startswith("BELT_")
        }
        missing = public_constants - envvars.ALL_NAMES
        assert not missing, f"envvars.ALL_NAMES is missing: {sorted(missing)}"


class TestInternalEnvvarsConstants:
    """Mirror the public hygiene test for ``_internal_envvars``.

    ``PREFIX`` is metadata about the namespace (the prefix itself), not
    a handoff variable name; both tests exclude it so the
    ``_BELT_<NAME>`` shape check and the registry-membership check
    apply only to actual handoff variables.
    """

    _METADATA_NAMES = {"PREFIX"}

    def test_prefix_constant_value_is_pinned(self):
        """The literal value of ``PREFIX`` is the security-critical
        boundary that ``build_subprocess_env`` uses to strip private
        handoff variables. If a future refactor accidentally changes
        the prefix to e.g. ``"BELT_"``, every existing
        ``_BELT_*`` variable would silently leak into child
        agents - and every other test in this module would still pass
        because they reference ``PREFIX`` symbolically. This test pins
        the literal so the regression is loud.
        """
        assert _internal_envvars.PREFIX == "_BELT_"

    def test_constants_match_identifiers(self):
        for name in dir(_internal_envvars):
            if name.startswith("_") or name.upper() != name or name == "ALL_INTERNAL_NAMES":
                continue
            if name in self._METADATA_NAMES:
                continue
            value = getattr(_internal_envvars, name)
            if not isinstance(value, str) or not value.startswith(_internal_envvars.PREFIX):
                continue
            expected = f"{_internal_envvars.PREFIX}{name}"
            assert value == expected, f"_internal_envvars.{name} = {value!r}, expected {expected!r}"

    def test_all_internal_names_covers_every_constant(self):
        constants = {
            getattr(_internal_envvars, name)
            for name in dir(_internal_envvars)
            if name.upper() == name
            and not name.startswith("_")
            and name != "ALL_INTERNAL_NAMES"
            and name not in self._METADATA_NAMES
            and isinstance(getattr(_internal_envvars, name), str)
            and getattr(_internal_envvars, name).startswith(_internal_envvars.PREFIX)
        }
        missing = constants - _internal_envvars.ALL_INTERNAL_NAMES
        assert not missing, f"_internal_envvars.ALL_INTERNAL_NAMES is missing: {sorted(missing)}"


# ── Section 2: no inline literals ───────────────────────────────────────────


class TestNoInlineLiterals:
    """Catch regressions where a contributor reintroduces an inline literal."""

    def test_every_public_literal_in_src_is_known(self):
        unknown: dict[str, list[str]] = {}
        for py in _SRC_ROOT.rglob("*.py"):
            if py.name == "envvars.py":
                continue
            text = py.read_text(encoding="utf-8")
            for match in _NAME_RE.finditer(text):
                name = match.group(1)
                if name in envvars.ALL_NAMES:
                    continue
                unknown.setdefault(str(py.relative_to(_SRC_ROOT.parent)), []).append(name)
        if unknown:
            lines = [f"{path}: {', '.join(sorted(set(names)))}" for path, names in sorted(unknown.items())]
            pytest.fail(
                "Inline BELT_* literals not in envvars.ALL_NAMES. "
                "Add them to envvars.py and use the constant instead:\n  " + "\n  ".join(lines)
            )

    def test_no_internal_literal_outside_internal_module(self):
        """``_BELT_*`` literals may only appear in ``_internal_envvars.py``.

        A typo'd internal name silently breaks the run -> score -> aggregate
        chain because the receiving phase reads ``""`` and falls through to
        a "no run" code path. Restricting literals to the source-of-truth
        module forces every reader to import a constant whose typo is a
        hard error.
        """
        offenders: dict[str, list[str]] = {}
        for py in _SRC_ROOT.rglob("*.py"):
            if py.name == "_internal_envvars.py":
                continue
            text = py.read_text(encoding="utf-8")
            for match in _INTERNAL_NAME_RE.finditer(text):
                name = match.group(1)
                offenders.setdefault(str(py.relative_to(_SRC_ROOT.parent)), []).append(name)
        if offenders:
            lines = [f"{path}: {', '.join(sorted(set(names)))}" for path, names in sorted(offenders.items())]
            pytest.fail(
                "Inline _BELT_* literals leaked outside _internal_envvars.py. "
                "Import the constant instead:\n  " + "\n  ".join(lines)
            )


# ── Section 3: allow-list correctness ───────────────────────────────────────


class TestPublicAllowSourcedFromEnvvars:
    def test_safe_environ_uses_envvars_public_allow_live(self, monkeypatch):
        """``safe_environ`` must read ``envvars.PUBLIC_ALLOW`` on every call.

        Caching a local alias of ``envvars.PUBLIC_ALLOW`` at module
        import time would silently desynchronise whenever a test
        monkeypatched the source. Reading the canonical attribute on
        every call keeps the public allow-list with exactly one home
        (``envvars.PUBLIC_ALLOW``); if a cached alias is reintroduced,
        this test fails because the patched name is not observed.
        """
        from belt._redact import safe_environ

        monkeypatch.setattr(envvars, "PUBLIC_ALLOW", frozenset({"BELT_TEST_LIVE_PROBE"}))
        out = safe_environ({"BELT_TEST_LIVE_PROBE": "ok", "OPENAI_API_KEY": "leaked"})
        assert out == {"BELT_TEST_LIVE_PROBE": "ok"}

    def test_debug_is_in_public_allow(self):
        """``BELT_DEBUG`` is a non-secret operator toggle.

        It is intentionally set in CI to enable verbose tracebacks. If it
        is redacted to ``"<set>"`` in ``run_meta.json``, reproducing a
        flaky run becomes harder than necessary. The opposite mistake
        (treating a secret as non-secret) is caught by
        ``_redact``'s secret-name regex.
        """
        assert envvars.DEBUG in envvars.PUBLIC_ALLOW


# ── Section 4: sorted registries ────────────────────────────────────────────


class TestRegistriesAreSorted:
    """``ALL_NAMES`` and ``PUBLIC_ALLOW`` are written in alphabetical order.

    ``frozenset`` itself has no order; we re-read the source file and
    inspect the literals so a contributor cannot quietly add a name out
    of order. The diff in a PR is then unambiguous.
    """

    @staticmethod
    def _literal_order(target: str) -> list[str]:
        """Return the source-order names inside ``frozenset({...})`` literal.

        Handles both ``Final[...] = frozenset(...)`` (parsed as
        ``ast.AnnAssign``) and bare ``= frozenset(...)`` (``ast.Assign``).
        """
        tree = ast.parse((_SRC_ROOT / "envvars.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            target_name: str | None = None
            value: ast.expr | None = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                tgt = node.targets[0]
                if isinstance(tgt, ast.Name):
                    target_name = tgt.id
                    value = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                target_name = node.target.id
                value = node.value
            if target_name != target or value is None:
                continue
            if not isinstance(value, ast.Call):
                pytest.fail(f"{target} is not assigned via frozenset(...)")
            if not value.args or not isinstance(value.args[0], ast.Set):
                pytest.fail(f"{target} expected frozenset({{...}}) literal")
            return [elt.id for elt in value.args[0].elts if isinstance(elt, ast.Name)]
        pytest.fail(f"Couldn't find assignment for {target}")
        return []  # pragma: no cover - pytest.fail raises

    def test_all_names_is_sorted_in_source(self):
        order = self._literal_order("ALL_NAMES")
        assert order == sorted(order), (
            f"envvars.ALL_NAMES is not alphabetised in source.\n" f"  saw:    {order}\n  sorted: {sorted(order)}"
        )

    def test_public_allow_is_sorted_in_source(self):
        order = self._literal_order("PUBLIC_ALLOW")
        assert order == sorted(order), (
            f"envvars.PUBLIC_ALLOW is not alphabetised in source.\n" f"  saw:    {order}\n  sorted: {sorted(order)}"
        )


# ── Section 5: flag <-> env parity ──────────────────────────────────────────


def _collect_long_flags() -> set[str]:
    """Return the set of every long flag declared via ``add_argument(...)``.

    Walks every ``*.py`` under ``src/belt`` with an AST so we don't
    rely on string heuristics. Both ``parser.add_argument("--foo", ...)``
    and ``parser.add_argument("-x", "--foo", ...)`` are picked up.
    """
    flags: set[str] = set()
    for py in _SRC_ROOT.rglob("*.py"):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                        flags.add(arg.value)
    return flags


class TestAllowFlagParity:
    """Every ``BELT_ALLOW_*`` constant has a matching ``--allow-*`` flag.

    The mapping is mechanical:
    ``BELT_ALLOW_FOO_BAR`` <-> ``--allow-foo-bar`` (lowercase, hyphens).
    """

    @staticmethod
    def _expected_flag(env_name: str) -> str:
        # ``BELT_ALLOW_FOO_BAR`` -> ``--allow-foo-bar``
        suffix = env_name[len("BELT_") :]  # ALLOW_FOO_BAR
        return "--" + suffix.lower().replace("_", "-")

    def test_every_allow_env_has_a_flag(self):
        flags = _collect_long_flags()
        missing: list[str] = []
        for env_name in sorted(envvars.ALL_NAMES):
            if not env_name.startswith("BELT_ALLOW_"):
                continue
            expected = self._expected_flag(env_name)
            if expected not in flags:
                missing.append(f"{env_name} (expected flag: {expected})")
        assert not missing, (
            "Every BELT_ALLOW_* env var must have a matching --allow-* flag "
            "on at least one CLI command.\n  " + "\n  ".join(missing)
        )


# ── Section 6: documentation coverage ───────────────────────────────────────


class TestDocCoverage:
    """Every public env var is mentioned by name in the user docs.

    Why this matters: a user trying to understand why their CI is doing
    something unexpected greps for ``BELT_FOO`` in the repo. If the
    only hit is a Python module, the user has to read code. The docs
    should always carry the user-facing name so the search bar is
    enough.

    The check is loose on purpose: we only assert the *literal name*
    appears somewhere in CONFIGURATION.md. Tone, table placement, and
    cross-referencing are reviewer concerns, not test concerns.
    """

    def test_every_public_envvar_in_configuration_md(self):
        doc_text = (_REPO_ROOT / "docs" / "glossary" / "CONFIGURATION.md").read_text(encoding="utf-8")
        missing = sorted(name for name in envvars.ALL_NAMES if name not in doc_text)
        assert not missing, (
            "Public env vars not documented in docs/glossary/CONFIGURATION.md. "
            "Add a row to the relevant table:\n  " + "\n  ".join(missing)
        )


# ── Section 7: forward_security_toggles ───────────────────────────────────


class _Args:
    """Minimal argparse-like namespace for forward_security_toggles tests."""

    def __init__(self, **flags: bool) -> None:
        for k, v in flags.items():
            setattr(self, k, v)


class TestForwardSecurityToggles:
    """The ``eval`` / ``run`` / ``score`` commands all funnel ``--allow-*``
    flags into the process env via this helper, so adapter / scorer
    plugins loaded in subprocess phases see them. A regression that
    drops a flag would silently relax a default-deny safety toggle."""

    @pytest.fixture(autouse=True)
    def _reset_env(self, monkeypatch: pytest.MonkeyPatch):
        for name in (
            envvars.ALLOW_FULL_ENV,
            envvars.ALLOW_ARBITRARY_AGENT,
            envvars.ALLOW_ARBITRARY_EXPORTER,
            envvars.ALLOW_ARBITRARY_SCORER,
            envvars.ALLOW_INSECURE_BASE_URL,
        ):
            monkeypatch.delenv(name, raising=False)

    def test_forwards_set_flags_into_env(self):
        args = _Args(allow_full_env=True, allow_arbitrary_agent=True)
        out = envvars.forward_security_toggles(args)
        assert envvars.ALLOW_FULL_ENV in out
        assert envvars.ALLOW_ARBITRARY_AGENT in out
        assert os.environ[envvars.ALLOW_FULL_ENV] == "1"
        assert os.environ[envvars.ALLOW_ARBITRARY_AGENT] == "1"

    def test_skips_unset_flags(self):
        args = _Args(allow_full_env=False, allow_arbitrary_agent=True)
        envvars.forward_security_toggles(args)
        assert envvars.ALLOW_FULL_ENV not in os.environ
        assert os.environ[envvars.ALLOW_ARBITRARY_AGENT] == "1"

    def test_missing_attributes_treated_as_false(self):
        # ``score`` parser does not declare ``--allow-full-env``; the
        # helper must not crash when that attribute is absent.
        args = _Args(allow_arbitrary_scorer=True)
        out = envvars.forward_security_toggles(args)
        assert envvars.ALLOW_FULL_ENV not in os.environ
        assert envvars.ALLOW_ARBITRARY_SCORER in out

    def test_flag_overrides_existing_env(self, monkeypatch: pytest.MonkeyPatch):
        # CLI flag wins over a shell value of "0" - matches the semantics
        # of the inline ``os.environ[X] = "1"`` assignments the helper
        # replaced.
        monkeypatch.setenv(envvars.ALLOW_FULL_ENV, "0")
        envvars.forward_security_toggles(_Args(allow_full_env=True))
        assert os.environ[envvars.ALLOW_FULL_ENV] == "1"

    def test_returns_empty_tuple_when_nothing_set(self):
        args = _Args()
        assert envvars.forward_security_toggles(args) == ()

    def test_no_flag_leaves_env_untouched(self, monkeypatch: pytest.MonkeyPatch):
        # When the flag is absent, a previously-exported env value stays.
        monkeypatch.setenv(envvars.ALLOW_FULL_ENV, "1")
        envvars.forward_security_toggles(_Args(allow_full_env=False))
        assert os.environ[envvars.ALLOW_FULL_ENV] == "1"

    def test_documented_toggles_are_wired(self):
        # Pin the contract: every entry in
        # :data:`envvars._SECURITY_TOGGLE_FLAG_TO_ENV` (the source of truth
        # for which flags get forwarded) round-trips its env name out of
        # :func:`envvars.forward_security_toggles`. Adding a new toggle
        # registers it in ``_SECURITY_TOGGLE_FLAG_TO_ENV`` and this test
        # picks it up automatically.
        args = _Args(**{flag: True for flag, _ in envvars._SECURITY_TOGGLE_FLAG_TO_ENV})
        out = set(envvars.forward_security_toggles(args))
        expected = {env for _, env in envvars._SECURITY_TOGGLE_FLAG_TO_ENV}
        assert out == expected
