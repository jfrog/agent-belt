# (c) JFrog Ltd. (2026)

"""Validates the bundled agent-skill at
``src/belt/.agents/skills/belt/SKILL.md``.

The skill ships inside the wheel under ``belt/.agents/skills/belt/``
and is the contract through which a consumer's AI agent (Cursor, Claude Code,
Codex, etc.) learns how to operate ``belt`` once it is wired into the
project's standard skills directory. If this file rots - broken doc links,
oversize body, missing frontmatter - consumer agents lose their grounding and
silently produce wrong output. The gate is structural, not stylistic: it covers
only the things Anthropic's `Skill authoring best practices`_ list as hard
constraints plus the repo-internal invariant that every linked
``docs/glossary/`` URL resolves to a real file.

.. _Skill authoring best practices: https://docs.anthropic.com/en/docs/agents-and-tools/agent-skills/best-practices
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_PATH = REPO_ROOT / "src" / "belt" / ".agents" / "skills" / "belt" / "SKILL.md"

# Anthropic's published ceiling is 500 lines, with the recommendation to stay
# closer to 300; we cap at 350 to keep slack while leaving room for legitimate
# expansion. Bumping this past 500 is a design decision that should fail loud.
MAX_BODY_LINES = 350

# Anthropic spec: name ≤64 chars, lowercase + digits + hyphens only, must not
# contain reserved words. Description ≤1024 chars, drives skill selection.
NAME_PATTERN = re.compile(r"^[a-z0-9-]+$")
RESERVED_NAME_TOKENS = ("anthropic", "claude")
MAX_NAME_LEN = 64
MAX_DESCRIPTION_LEN = 1024

GITHUB_DOC_LINK_PATTERN = re.compile(r"https://github\.com/jfrog/belt/blob/main/(\S+?)(?=[\s)\]])")


def _load() -> tuple[dict, str]:
    text = SKILL_PATH.read_text()
    if not text.startswith("---\n"):
        pytest.fail(f"SKILL.md must start with '---' YAML frontmatter delimiter, got: {text[:40]!r}")
    parts = text.split("---\n", 2)
    if len(parts) < 3:
        pytest.fail("SKILL.md frontmatter must be closed with a second '---' line")
    frontmatter = yaml.safe_load(parts[1]) or {}
    body = parts[2]
    return frontmatter, body


def test_skill_md_exists() -> None:
    assert SKILL_PATH.exists(), (
        f"SKILL.md must exist at {SKILL_PATH.relative_to(REPO_ROOT)} - it ships inside the wheel and is the "
        "contract through which consumer AI agents learn how to operate the belt CLI."
    )


def test_frontmatter_has_required_fields() -> None:
    frontmatter, _ = _load()
    assert "name" in frontmatter, "SKILL.md frontmatter missing required 'name' field"
    assert "description" in frontmatter, "SKILL.md frontmatter missing required 'description' field"


def test_name_field_matches_anthropic_spec() -> None:
    frontmatter, _ = _load()
    name = frontmatter["name"]
    assert isinstance(name, str) and name, "frontmatter 'name' must be a non-empty string"
    assert len(name) <= MAX_NAME_LEN, f"frontmatter 'name' exceeds {MAX_NAME_LEN} chars: {len(name)}"
    assert NAME_PATTERN.match(
        name
    ), f"frontmatter 'name' must match {NAME_PATTERN.pattern} (lowercase letters, digits, hyphens only); got: {name!r}"
    for token in RESERVED_NAME_TOKENS:
        assert token not in name, f"frontmatter 'name' contains reserved token {token!r}: {name!r}"


def test_name_matches_package() -> None:
    """The skill name must match the package name. If they ever diverge, either
    the package was renamed (in which case the skill should follow) or the
    skill is mis-titled (it must be the canonical entry point a consumer types)."""
    frontmatter, _ = _load()
    assert (
        frontmatter["name"] == "belt"
    ), f"frontmatter 'name' must be 'belt' to match the package; got {frontmatter['name']!r}"


def test_description_field_matches_anthropic_spec() -> None:
    frontmatter, _ = _load()
    description = frontmatter["description"]
    assert isinstance(description, str) and description, "frontmatter 'description' must be a non-empty string"
    assert (
        len(description) <= MAX_DESCRIPTION_LEN
    ), f"frontmatter 'description' exceeds {MAX_DESCRIPTION_LEN} chars: {len(description)}"
    # First-person leakage is the most common discoverability bug Anthropic calls out.
    lower = description.lower()
    for fragment in (" i can ", " i will ", "you can use this", "this skill lets you"):
        assert fragment not in lower, (
            f"frontmatter 'description' uses first/second person ({fragment!r}); must be third-person to keep "
            f"skill selection consistent with the surrounding system prompt."
        )


def test_body_within_line_budget() -> None:
    _, body = _load()
    line_count = body.count("\n")
    assert line_count <= MAX_BODY_LINES, (
        f"SKILL.md body is {line_count} lines, exceeding the {MAX_BODY_LINES}-line budget. Move depth into "
        f"docs/glossary/*.md (linked from SKILL.md), not into the body."
    )


def test_doc_links_resolve_to_existing_files() -> None:
    """Every ``https://github.com/jfrog/agent-belt/blob/main/<path>`` URL in the
    skill must resolve to a file that exists in the working tree. URLs are
    pinned to ``main`` so consumer-side reads land on the canonical doc, but
    that contract is only useful if the path is real *now* - a doc that has
    been renamed or deleted in this PR breaks the link silently."""
    _, body = _load()
    bad: list[str] = []
    for match in GITHUB_DOC_LINK_PATTERN.finditer(body):
        rel_path = match.group(1)
        # Strip a ``#anchor`` fragment before the on-disk existence check
        # - GitHub renders fragments against the *file*, so the file alone
        # is what must exist for the link to resolve.
        file_part = rel_path.split("#", 1)[0]
        full = REPO_ROOT / file_part
        if not full.exists():
            bad.append(rel_path)
    assert not bad, (
        "SKILL.md links to docs that don't exist in the working tree:\n  "
        + "\n  ".join(bad)
        + "\nUpdate the links or restore the missing files before merging."
    )
