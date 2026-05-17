# (c) JFrog Ltd. (2026)

"""Stdout / stderr routing contract.

Pin the convention enforced by :mod:`belt._ui`: UI output goes to
stderr; stdout is reserved for data (``--output markdown``,
``--output json``, future pipeable outputs). A pipeline like
``belt eval ... | jq`` must not have its stdout contaminated by status
banners.

A grep-level audit complements the runtime tests: any ``print(`` in
``src/belt/`` that isn't accompanied by ``file=sys.stderr`` and isn't
in the small allow-list of legitimate stdout writers is a regression.
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import pytest

from belt._ui import eprint

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src" / "belt"

# Files where bare ``print(...)`` is the intended stdout writer. Every
# entry should be paired with a comment in the source explaining why.
STDOUT_ALLOWLIST = {
    SRC / "commands" / "compare.py",  # --output markdown / --output json
    SRC / "commands" / "doctor.py",  # --json data dump
    SRC / "cli.py",  # `belt version` string
    SRC / "_ui.py",  # the eprint definition itself
}


class TestEprintHelper:
    def test_eprint_defaults_to_stderr(self) -> None:
        # Default: eprint writes to stderr. Capture by swapping the
        # interpreter's stderr for the duration of the call.
        buf = io.StringIO()
        original = sys.stderr
        sys.stderr = buf
        try:
            eprint("hello")
        finally:
            sys.stderr = original
        assert buf.getvalue() == "hello\n"

    def test_eprint_honors_file_override(self) -> None:
        # ``file=`` override still works (used by tests that want to
        # capture into a buffer rather than redirect a real fd).
        buf = io.StringIO()
        eprint("captured", file=buf)
        assert buf.getvalue() == "captured\n"


class TestStdoutAuditConvention:
    """Static check: every bare ``print(`` in ``src/belt/`` must be in
    the allow-list or pass ``file=sys.stderr``. Catches the next time
    someone adds a UI ``print(...)`` without going through ``eprint``.
    """

    def test_no_unannotated_stdout_prints(self) -> None:
        offenders: list[str] = []
        for py in SRC.rglob("*.py"):
            if py in STDOUT_ALLOWLIST:
                continue
            text = py.read_text(encoding="utf-8")
            for lineno, raw in enumerate(text.splitlines(), 1):
                line = raw.lstrip()
                if not line.startswith("print("):
                    continue
                if "file=sys" in raw:
                    continue
                # Look for the closing of the call within ~5 lines to
                # catch multi-line ``print(\n  "...",\n  file=sys.stderr,\n)``.
                ctx = "\n".join(text.splitlines()[lineno - 1 : lineno + 5])
                if "file=sys" in ctx:
                    continue
                offenders.append(f"{py.relative_to(REPO_ROOT)}:{lineno}: {raw.strip()}")
        assert not offenders, (
            "Bare print(...) found outside the stdout allow-list. "
            "Use eprint() from belt._ui for UI output, or add an "
            "explicit file=sys.stderr if you cannot import it.\n\n" + "\n".join(offenders)
        )


@pytest.mark.parametrize("argv", [["--version"], ["agent", "list"]])
def test_belt_subprocess_stdout_does_not_contain_ui(argv: list[str]) -> None:
    """Smoke test: ``belt --version`` writes only the version string to
    stdout; ``belt agent list`` writes nothing to stdout (the table is
    UI on stderr). Run as a subprocess so we see the real stream split,
    not the capsys patching that pytest applies in-process.
    """
    cmd = [sys.executable, "-m", "belt.cli", *argv]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, check=False)

    if argv == ["--version"]:
        # Version line is the one allowed stdout writer in cli.py.
        # Any additional non-empty line on stdout is a UI leak.
        non_empty = [ln for ln in result.stdout.splitlines() if ln.strip()]
        assert len(non_empty) <= 1, f"unexpected stdout lines: {non_empty!r}"
    elif argv == ["agent", "list"]:
        # The agent table is UI; stdout must be empty.
        assert result.stdout == "", f"agent list leaked to stdout: {result.stdout!r}"
