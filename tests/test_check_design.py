# (c) JFrog Ltd. (2026)

"""Tests for ``scripts/check_design.py``.

The script is the mechanical guard for ``docs/glossary/ARCHITECTURE.md``.
If the script silently stops detecting violations, every PR after that point
ships unchecked. These tests pin each principle's detector against synthetic
violation/non-violation pairs so a refactor that breaks the detector fails
loudly in CI rather than 100 PRs later.

Each test pair follows the same shape:

    1. Build an in-memory snippet that *should* trigger the check.
    2. Build an in-memory snippet that *should not* trigger the check.
    3. Run the check function with a temporary ``SRC`` rooted at a
       fixture directory and assert detection.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "check_design.py"


def _load_check_design():
    """Load ``scripts/check_design.py`` as an importable module."""
    spec = importlib.util.spec_from_file_location("check_design", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def cd_module(monkeypatch, tmp_path):
    """Load ``check_design`` and redirect its ``SRC`` to an isolated tree."""
    mod = _load_check_design()
    src = tmp_path / "src" / "belt"
    src.mkdir(parents=True)
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "SRC", src)
    return mod


# ── Principle 1: phase independence ────────────────────────────────────


def test_principle_1_detects_cross_phase_import(cd_module):
    runner = cd_module.SRC / "runner"
    runner.mkdir()
    (runner / "leak.py").write_text("from belt.scorer.rules import scorer\n")
    errors: list[str] = []
    cd_module.check_principle_1_phase_independence(errors)
    assert any("Principle 1" in e and "leak.py" in e for e in errors)


def test_principle_1_clean_codebase_passes(cd_module):
    runner = cd_module.SRC / "runner"
    runner.mkdir()
    (runner / "ok.py").write_text("from belt.entities import TurnOutput\n")
    errors: list[str] = []
    cd_module.check_principle_1_phase_independence(errors)
    assert errors == []


# ── Principle 2: entities are data ─────────────────────────────────────


def test_principle_2_detects_entity_business_logic(cd_module):
    (cd_module.SRC / "entities.py").write_text("class Foo:\n" "    def compute_score(self):\n" "        return 42\n")
    errors: list[str] = []
    cd_module.check_principle_2_entities_no_logic(errors)
    assert any("Principle 2" in e and "compute_score" in e for e in errors)


def test_principle_2_allows_validators_and_dunders(cd_module):
    (cd_module.SRC / "entities.py").write_text(
        "class Foo:\n"
        "    def __init__(self): pass\n"
        "    def __repr__(self): return ''\n"
        "    @property\n"
        "    def name(self): return 'x'\n"
        "    @field_validator('x')\n"
        "    def check_x(cls, v): return v\n"
        "    def model_dump_safe(self): return {}\n"
    )
    errors: list[str] = []
    cd_module.check_principle_2_entities_no_logic(errors)
    assert errors == []


# ── Principle 3: base interface signature ──────────────────────────────


def test_principle_3_detects_extra_param_on_execute(cd_module):
    agent_dir = cd_module.SRC / "agent"
    agent_dir.mkdir()
    (agent_dir / "base.py").write_text(
        "class BaseAgentAdapter:\n"
        "    def execute(self, message, flags, model):\n"
        "        ...\n"
        "    def fetch_results(self, raw_output):\n"
        "        ...\n"
    )
    errors: list[str] = []
    cd_module.check_principle_3_base_signatures(errors)
    assert any("Principle 3" in e and "execute" in e for e in errors)


def test_principle_3_canonical_signature_passes(cd_module):
    agent_dir = cd_module.SRC / "agent"
    agent_dir.mkdir()
    (agent_dir / "base.py").write_text(
        "class BaseAgentAdapter:\n"
        "    def execute(self, message, flags):\n"
        "        ...\n"
        "    def fetch_results(self, raw_output):\n"
        "        ...\n"
    )
    errors: list[str] = []
    cd_module.check_principle_3_base_signatures(errors)
    assert errors == []


# ── Principle 5: no isinstance on concrete agents ─────────────────────


def test_principle_5_detects_isinstance_on_unknown_agent(cd_module):
    """The generic pattern must catch a NEW agent class - not just the seven
    legacy hardcoded names. This is the regression we're guarding against."""
    scorer = cd_module.SRC / "scorer"
    scorer.mkdir()
    (scorer / "branchy.py").write_text("if isinstance(agent, BrandNewAgentAdapter):\n" "    pass\n")
    errors: list[str] = []
    cd_module.check_principle_5_no_isinstance_on_agents(errors)
    assert any("Principle 5" in e and "branchy.py" in e for e in errors)


def test_principle_5_isinstance_on_non_agent_passes(cd_module):
    scorer = cd_module.SRC / "scorer"
    scorer.mkdir()
    (scorer / "ok.py").write_text("if isinstance(x, dict):\n    pass\n")
    errors: list[str] = []
    cd_module.check_principle_5_no_isinstance_on_agents(errors)
    assert errors == []


# ── Principle 7: no sys.exit() in callable main() ─────────────────────


def test_principle_7_detects_sys_exit_in_callable(cd_module):
    cmd = cd_module.SRC / "commands"
    cmd.mkdir()
    (cmd / "bad.py").write_text("import sys\n" "def main():\n" "    sys.exit(1)\n")
    errors: list[str] = []
    cd_module.check_principle_7_no_sys_exit_in_callable_main(errors)
    assert any("Principle 7" in e for e in errors)


def test_principle_7_sys_exit_under_dunder_main_block_passes(cd_module):
    cmd = cd_module.SRC / "commands"
    cmd.mkdir()
    (cmd / "ok.py").write_text(
        "import sys\n" "def main(): return 0\n" "if __name__ == '__main__':\n" "    sys.exit(main())\n"
    )
    errors: list[str] = []
    cd_module.check_principle_7_no_sys_exit_in_callable_main(errors)
    assert errors == []


# ── Principle 8: rich_safe / md_safe escaping ─────────────────────────


def test_principle_8_detects_unwrapped_untrusted_attr(cd_module):
    agg = cd_module.SRC / "aggregator"
    agg.mkdir()
    (agg / "render_terminal.py").write_text('console.print(f"reply: {turn.reply_text}")\n')
    errors: list[str] = []
    cd_module.check_principle_8_output_escaping(errors)
    assert any("Principle 8" in e and "reply_text" in e for e in errors)


def test_principle_8_wrapped_attr_passes(cd_module):
    agg = cd_module.SRC / "aggregator"
    agg.mkdir()
    (agg / "render_terminal.py").write_text('console.print(f"reply: {rich_safe(turn.reply_text)}")\n')
    errors: list[str] = []
    cd_module.check_principle_8_output_escaping(errors)
    assert errors == []


# ── Principle 8 (cont.): --allow-* flags default-deny ────────────────


def test_principle_8_detects_default_true_allow_flag(cd_module):
    cmd = cd_module.SRC / "commands"
    cmd.mkdir()
    (cmd / "leaky.py").write_text("parser.add_argument('--allow-insecure', default=True)\n")
    errors: list[str] = []
    cd_module.check_principle_8_default_deny(errors)
    assert any("Principle 8" in e and "--allow-insecure" in e for e in errors)


def test_principle_8_store_true_passes(cd_module):
    cmd = cd_module.SRC / "commands"
    cmd.mkdir()
    (cmd / "ok.py").write_text("parser.add_argument('--allow-insecure', action='store_true', default=False)\n")
    errors: list[str] = []
    cd_module.check_principle_8_default_deny(errors)
    assert errors == []


def test_principle_8_non_allow_flag_ignored(cd_module):
    cmd = cd_module.SRC / "commands"
    cmd.mkdir()
    (cmd / "ok.py").write_text("parser.add_argument('--workers', default=4, type=int)\n")
    errors: list[str] = []
    cd_module.check_principle_8_default_deny(errors)
    assert errors == []


# ── Principle 6: plugins import only public belt symbols ────────


def _stage_plugin_tree(cd_module, files: dict[str, str]) -> None:
    for rel, contents in files.items():
        path = cd_module.ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)


def test_plugin_public_api_detects_internal_from_import(cd_module):
    _stage_plugin_tree(
        cd_module,
        {"plugins/x/src/x/__init__.py": "from belt.exporter.base import BaseExporter\n"},
    )
    errors: list[str] = []
    cd_module.check_principle_6_plugin_public_api_only(errors)
    assert any("Principle 6" in e and "BaseExporter" in e and "internal" in e for e in errors)


def test_plugin_public_api_detects_internal_module_import(cd_module):
    _stage_plugin_tree(
        cd_module,
        {"plugins/x/src/x/foo.py": "import belt.scorer.llm.backend\n"},
    )
    errors: list[str] = []
    cd_module.check_principle_6_plugin_public_api_only(errors)
    assert any("Principle 6" in e and "belt.scorer.llm.backend" in e for e in errors)


def test_plugin_public_api_passes_top_level_import(cd_module):
    _stage_plugin_tree(
        cd_module,
        {"plugins/x/src/x/__init__.py": "from belt import BaseExporter, ExportContext\n"},
    )
    errors: list[str] = []
    cd_module.check_principle_6_plugin_public_api_only(errors)
    assert errors == []


def test_plugin_public_api_passes_bare_import(cd_module):
    _stage_plugin_tree(
        cd_module,
        {"plugins/x/src/x/foo.py": "import belt\n"},
    )
    errors: list[str] = []
    cd_module.check_principle_6_plugin_public_api_only(errors)
    assert errors == []


def test_plugin_public_api_scans_examples_custom_agent(cd_module):
    """The reference example is held to the same standard as real plugins."""
    _stage_plugin_tree(
        cd_module,
        {
            "examples/custom-agent/leaky_agent.py": "from belt.agent.base import BaseAgentAdapter\n",
        },
    )
    errors: list[str] = []
    cd_module.check_principle_6_plugin_public_api_only(errors)
    assert any("Principle 6" in e and "leaky_agent.py" in e for e in errors)


def test_plugin_public_api_ignores_egg_info(cd_module):
    """Auto-generated egg-info trees must not trigger the check."""
    _stage_plugin_tree(
        cd_module,
        {
            "plugins/x/src/x.egg-info/leaky.py": "from belt.agent.base import BaseAgentAdapter\n",
        },
    )
    errors: list[str] = []
    cd_module.check_principle_6_plugin_public_api_only(errors)
    assert errors == []


# ── Convention: copyright header ──────────────────────────────────────


def _stage_header_tree(cd_module, files: dict[str, str]) -> None:
    """Materialise files (path → contents) under ``ROOT`` and ensure
    ``check_copyright_header`` walks them, not the real repo."""
    for rel, contents in files.items():
        path = cd_module.ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)


def test_copyright_header_detects_missing(cd_module):
    _stage_header_tree(cd_module, {"src/belt/widget.py": "x = 1\n"})
    errors: list[str] = []
    cd_module.check_copyright_header(errors)
    assert any("Convention" in e and "widget.py" in e for e in errors)


def test_copyright_header_accepts_shebang_then_header(cd_module):
    _stage_header_tree(
        cd_module,
        {"scripts/foo.py": "#!/usr/bin/env python3\n# (c) JFrog Ltd. (2026)\nx = 1\n"},
    )
    errors: list[str] = []
    cd_module.check_copyright_header(errors)
    assert errors == []


def test_copyright_header_accepts_header_first_line(cd_module):
    _stage_header_tree(cd_module, {"tests/test_foo.py": "# (c) JFrog Ltd. (2026)\n"})
    errors: list[str] = []
    cd_module.check_copyright_header(errors)
    assert errors == []


def test_copyright_header_skips_virtualenv_and_cache_dirs(cd_module):
    """Pins ``_HEADER_SKIP_DIR_NAMES``: any path segment in that set
    disqualifies the file. Source files outside those segments stay
    subject to the header check."""
    _stage_header_tree(
        cd_module,
        {
            # Proves the scan runs (would error if the header were missing).
            "src/belt/widget.py": "# (c) JFrog Ltd. (2026)\nx = 1\n",
            # ``.venv`` segment skipped — realistic when a contributor
            # creates a venv inside a ``_HEADER_ROOTS`` entry.
            "src/belt/.venv/bin/activate_this.py": "x = 1\n",
            "src/belt/.venv/lib/python3.13/site-packages/_virtualenv.py": "x = 1\n",
            "src/belt/__pycache__/widget.cpython-313.pyc.py": "x = 1\n",
            "tests/.pytest_cache/CACHEDIR.tag.py": "x = 1\n",
        },
    )
    errors: list[str] = []
    cd_module.check_copyright_header(errors)
    assert errors == [], f"got errors: {errors}"


def test_copyright_header_exempt_set_is_consumed(cd_module):
    """Files declared in ``_HEADER_EXEMPT`` must be skipped by the header
    check. The set is currently empty (the previous ``_version.py`` entry
    went away with the move to ``importlib.metadata``-based versioning),
    but the exemption mechanism itself stays under test so future
    auto-generated additions wire up correctly.
    """
    if not cd_module._HEADER_EXEMPT:
        # ``pytest.skip`` (rather than a bare ``return``) so the no-op state
        # surfaces in test output. A silent pass would let the exemption
        # mechanism rot without anyone noticing.
        pytest.skip("_HEADER_EXEMPT is empty - no exempt files configured")
    fixtures = {rel: "x = 1\n" for rel in cd_module._HEADER_EXEMPT}
    _stage_header_tree(cd_module, fixtures)
    errors: list[str] = []
    cd_module.check_copyright_header(errors)
    assert errors == []


# ── End-to-end ─────────────────────────────────────────────────────────


def test_main_returns_zero_on_actual_repo():
    """The real codebase under ``main`` must pass every check."""
    mod = _load_check_design()
    assert mod.main() == 0
