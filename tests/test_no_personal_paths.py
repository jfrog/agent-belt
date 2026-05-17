# (c) JFrog Ltd. (2026)

"""Examples must not contain absolute paths from a contributor's personal machine.

Hard-coded paths like ``/Users/dorr/...`` or ``C:\\Users\\me\\...`` mean the
example only runs on the author's laptop. They typically sneak in via a
``--workspace`` flag or a cached ``working_dir`` value. This test fails fast
in CI before such an example can land in main.

Acceptable:
- ``$HOME``, ``${HOME}``, ``~/``  (portable expansions)
- generic placeholders documented in prose: ``/path/to/repo``

Forbidden anywhere under ``examples/`` (including JSON, Python, Markdown):
- ``/Users/``  (macOS)
- ``/home/``   (Linux)
- ``C:\\Users\\``  (Windows)
- ``/private/var/``, ``/var/folders/`` (macOS sandbox temp dirs)
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"

_FORBIDDEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("macOS user dir", re.compile(r"/Users/[A-Za-z0-9._-]+")),
    ("Linux user dir", re.compile(r"/home/[A-Za-z0-9._-]+")),
    ("Windows user dir", re.compile(r"[Cc]:\\\\Users\\\\[A-Za-z0-9._-]+")),
    ("macOS private temp", re.compile(r"/private/var/folders/[A-Za-z0-9._-]+")),
]

# Files whose primary purpose is to discuss this very test or its forbidden
# patterns. They legitimately mention the strings as documentation.
_ALLOWLIST = {
    REPO_ROOT / "tests" / "test_no_personal_paths.py",
}

# Substring-level exceptions: canonical paths that are framework contracts,
# not personal paths. ``/home/agent`` is the fixed in-container ``$HOME`` set
# by ``DockerSandboxProvider`` (see ``_CONTAINER_HOME``); examples MUST be
# able to mention it by name so readers can grep for the docs from the
# code constant. Each entry must be a path that is guaranteed identical on
# every machine.
_PATH_EXCEPTIONS: frozenset[str] = frozenset(
    {
        "/home/agent",  # _CONTAINER_HOME in src/belt/runner/sandbox/docker.py
    }
)


def _candidate_files() -> list[Path]:
    files: list[Path] = []
    for p in EXAMPLES_DIR.rglob("*"):
        if not p.is_file():
            continue
        if any(part.startswith(".") and part != "." for part in p.relative_to(EXAMPLES_DIR).parts):
            continue  # skip hidden dirs (.git/, etc.)
        if p.suffix.lower() not in {".json", ".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".cfg"}:
            continue
        if p in _ALLOWLIST:
            continue
        files.append(p)
    return sorted(files)


@pytest.mark.parametrize("path", _candidate_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_no_personal_paths_in_examples(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    for label, pattern in _FORBIDDEN_PATTERNS:
        for m in pattern.finditer(text):
            if m.group(0) in _PATH_EXCEPTIONS:
                continue
            raise AssertionError(
                f"{path.relative_to(REPO_ROOT)} contains a hard-coded {label} path: {m.group(0)!r}. "
                f"Use ``$HOME``, ``~/``, or a relative path instead so the example runs on every machine."
            )
