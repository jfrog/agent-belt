# (c) JFrog Ltd. (2026)

# agent-belt - development and evaluation Makefile
#
# Quick start:
#   make install        # one-time setup (uv-based)
#   make check          # lint + test (same as CI)
#   make eval           # run full evaluation pipeline
#
# We default to uv (PEP 723 / PEP 735 native, fast, reproducible). Pip-based
# workflows still work via the ``install-pip`` target for users who haven't
# adopted uv yet.

.DEFAULT_GOAL := help

# Detect uv; fall back to python -m pip when not present so contributors
# without uv installed are not blocked.
UV := $(shell command -v uv 2>/dev/null)
PYTHON ?= python3

# All ``uv run``/``uv sync`` commands honor ``uv.lock``; the ``--locked`` flag
# in CI fails the build if pyproject and the lockfile drift.

# ── Development ──

.PHONY: help install install-pip lock lint format test check check-plugins clean build verify-wheel

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dev environment (uv preferred; falls back to pip)
ifeq ($(UV),)
	@echo "uv not found - falling back to pip-based install. Install uv from https://docs.astral.sh/uv/ for the recommended path."
	@$(MAKE) install-pip
else
	uv sync
	@echo ""
	@echo "  ✓ uv-managed .venv ready. Activate with:"
	@echo "      source .venv/bin/activate"
	@echo "  Or run any command via: uv run <cmd>"
endif

install-pip: ## Pip-based install fallback (no uv); editable + dev deps in the active venv
	$(PYTHON) -m pip install -e ".[dev]"

lock: ## Refresh uv.lock from pyproject.toml
ifeq ($(UV),)
	@echo "uv is required for ``make lock``; install from https://docs.astral.sh/uv/" && exit 1
else
	uv lock
endif

lint: ## Run linter checks (Python + Markdown)
ifeq ($(UV),)
	pre-commit run --files src/**/*.py tests/**/*.py **/*.md
else
	uv run pre-commit run --files src/**/*.py tests/**/*.py **/*.md
endif

format: ## Auto-format code (black + isort)
ifeq ($(UV),)
	black src/ tests/
	isort src/ tests/
else
	uv run black src/ tests/
	uv run isort src/ tests/
endif

test: ## Run unit tests
ifeq ($(UV),)
	$(PYTHON) -m pytest tests/ -v
else
	uv run pytest tests/ -v
endif
	@echo ""
	@echo "────────────────────────────────────────────────────────────"
	@echo "  Tests passed. Now verify the real CLI:"
	@echo "    uv sync && uv run belt agent list"
	@echo "    uv run belt agent info <agent>"
	@echo "    uv run belt eval examples/scenarios/ --modes rules,llm"
	@echo "  Unit tests can pass while the CLI is broken."
	@echo ""
	@echo "  Ask yourself:"
	@echo "    → Are there new tests worth adding?"
	@echo "    → Are there new scenarios worth adding?"
	@echo "────────────────────────────────────────────────────────────"

check: lint test ## Run all checks (lint + test) - same as CI

check-plugins: ## Install + test every plugin under plugins/ (kept separate from `check` so core CI stays fast)
	@# Editable-install each plugin into the active venv, then run its pytest
	@# suite from the SAME interpreter. Mixing ``uv pip install`` with ``$(PYTHON)
	@# -m pytest`` would land the plugin in ``.venv`` but run pytest against
	@# system python3, which doesn't see it - confusing failure if the venv is
	@# not pre-activated. Both install and pytest must branch together on UV.
	@for d in plugins/*/; do \
		if [ -f "$$d/pyproject.toml" ]; then \
			echo "==> $$d"; \
			if [ -n "$(UV)" ]; then \
				uv pip install -e "$$d" --quiet || exit 1; \
				uv run pytest "$$d/tests" -q || exit 1; \
			else \
				$(PYTHON) -m pip install -e "$$d" --quiet || exit 1; \
				$(PYTHON) -m pytest "$$d/tests" -q || exit 1; \
			fi; \
		fi; \
	done
	@echo ""
	@echo "  ✓ All plugins installed and tested."

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

build: clean ## Build wheel and sdist
ifeq ($(UV),)
	$(PYTHON) -m build
else
	uv build
endif

# Resolves every bundled showcase group's ``working_dir`` against the installed
# wheel layout. Same logic as ``runner/phases/setup_groups.py``:
# ``(group_dir / gc.working_dir).resolve()`` must point at an existing directory,
# otherwise the first user to run the group hits
# ``WorkspaceError: working_dir does not exist`` - the F3 ship-blocker / #233
# regression class. Exported here so the multi-line python body stays readable
# (Makefile string-escaping inside ``python -c`` chained with ``;`` is hostile).
define VERIFY_WORKING_DIRS
import json
from importlib.resources import files
from pathlib import Path

showcase = Path(str(files("belt") / "_bundled_examples" / "scenarios" / "showcase"))
assert showcase.is_dir(), f"BUG: bundled showcase missing in wheel: {showcase}"

resolved = []
broken = []
for group_dir in sorted(showcase.iterdir()):
    cfg_path = group_dir / "_config.json"
    if not (group_dir.is_dir() and cfg_path.is_file()):
        continue
    cfg = json.loads(cfg_path.read_text())
    working_dir = cfg.get("working_dir")
    if not working_dir:
        continue
    target = (group_dir / working_dir).resolve()
    resolved.append((group_dir.name, target))
    if not target.is_dir():
        broken.append((group_dir.name, str(target)))

assert not broken, f"BUG: bundled groups whose working_dir is missing from the wheel: {broken}"
print(f"OK: {len(resolved)} bundled group(s) with working_dir all resolve inside the wheel:")
for name, path in resolved:
    print(f"    {name}: {path}")
endef
export VERIFY_WORKING_DIRS

verify-wheel: build ## Build wheel, install into TWO fresh venvs (uv + pip), smoke-test every CLI surface
	@echo "==> Verifying wheel installs cleanly via classical pip"
	@rm -rf /tmp/belt-wheel-verify-pip
	@$(PYTHON) -m venv /tmp/belt-wheel-verify-pip
	@/tmp/belt-wheel-verify-pip/bin/pip install --quiet $(shell ls dist/*.whl | head -1)
	@$(MAKE) -s _verify-wheel-smoke VENV=/tmp/belt-wheel-verify-pip
	@rm -rf /tmp/belt-wheel-verify-pip
ifneq ($(UV),)
	@echo "==> Verifying wheel installs cleanly via uv pip"
	@rm -rf /tmp/belt-wheel-verify-uv
	@uv venv --quiet /tmp/belt-wheel-verify-uv
	@uv pip install --quiet --python /tmp/belt-wheel-verify-uv/bin/python $(shell ls dist/*.whl | head -1)
	@$(MAKE) -s _verify-wheel-smoke VENV=/tmp/belt-wheel-verify-uv
	@rm -rf /tmp/belt-wheel-verify-uv
endif
	@echo ""
	@echo "  ✓ Wheel install verified end-to-end (pip$(if $(UV), + uv,))."
	@echo "  Run this on any PR that touches packaging, path lookups, or bundled assets."

# Internal helper: run the full smoke-test matrix against ``$(VENV)``. Called
# once per fresh venv from ``verify-wheel``. Keeps the smoke logic in one place
# whether the venv was provisioned by classical pip or by ``uv pip install``.
.PHONY: _verify-wheel-smoke
_verify-wheel-smoke:
	@cd /tmp && $(VENV)/bin/belt --version
	@cd /tmp && $(VENV)/bin/belt --help > /dev/null
	@cd /tmp && $(VENV)/bin/belt agent list > /dev/null
	@# doctor exits non-zero when no agents/providers are configured on the host;
	@# the smoke check only needs to verify it produced valid JSON without crashing.
	@cd /tmp && $(VENV)/bin/belt doctor --json 2>/dev/null \
		| $(VENV)/bin/python -c "import sys, json; json.load(sys.stdin)" \
		|| { echo "FAIL: doctor --json did not produce valid JSON"; exit 1; }
	@for cmd in eval run score aggregate export compare view watch quickstart agent gc; do \
		cd /tmp && $(VENV)/bin/belt $$cmd --help > /dev/null \
			|| { echo "FAIL: $$cmd --help broken on wheel install"; exit 1; }; \
	done
	@echo "==> Verifying bundled scenarios are discoverable via importlib.resources"
	@cd /tmp && $(VENV)/bin/python -c "\
from importlib.resources import files; \
from pathlib import Path; \
p = Path(str(files('belt') / '_bundled_examples' / 'scenarios')); \
assert p.is_dir(), f'BUG: bundled scenarios missing in wheel: {p}'; \
qs = p / 'showcase' / 'correctness' / 'correctness_basic.json'; \
assert qs.is_file(), f'BUG: quickstart scenario missing in wheel: {qs}'; \
print('OK: bundled scenarios present in wheel:', p)"
	@echo "==> Verifying every bundled group's working_dir resolves inside the wheel"
	@# Three showcase groups (editing-workspace, sandboxed, sandboxed-offline) declare
	@# ``working_dir: "../../../fixtures/sample-project"``. If the fixture is missing
	@# from the wheel the group raises ``WorkspaceError: working_dir does not exist``
	@# the first time a user runs it - the regression class of #233 and the F3
	@# ship-blocker. Dry-run does not exercise workspace acquisition, so we resolve
	@# the working_dir path here directly: same logic as
	@# ``runner/phases/setup_groups.py`` ((group_dir / gc.working_dir).resolve()),
	@# no agent CLI required.
	@cd /tmp && $(VENV)/bin/python -c "$$VERIFY_WORKING_DIRS"
	@echo "==> Verifying full showcase tree loads via belt eval --dry-run (no agent invocation)"
	@cd /tmp && $(VENV)/bin/belt eval \
		$(VENV)/lib/python*/site-packages/belt/_bundled_examples/scenarios/showcase \
		--modes rules --dry-run > /dev/null
	@echo "==> Verifying bundled reply_pattern showcase loads end-to-end from the wheel"
	@# Bad regexes abort at scenario load with one aggregated error; either
	@# scenario regressing on the regex side fails this load step.
	@cd /tmp && $(VENV)/bin/belt eval \
		$(VENV)/lib/python*/site-packages/belt/_bundled_examples/scenarios/showcase/correctness \
		--scenarios reply_pattern,reply_pattern_multiline \
		--modes rules --dry-run > /dev/null \
		|| { echo "FAIL: bundled reply_pattern showcase scenarios did not load from the wheel"; exit 1; }
	@echo "==> Verifying __version__ matches the wheel filename's PEP 440 version"
	@# Two distinct fallbacks could mask a broken ``__version__``:
	@#   * ``importlib.metadata`` raises ``PackageNotFoundError`` -> ``0.0.0+unknown``
	@#   * ``hatch-vcs`` outside a git work tree -> ``0.0.0`` (per ``fallback-version``)
	@# A bare ``!= '0.0.0+unknown'`` check passes both fallbacks silently. Compare
	@# against the wheel filename instead - that's the value PyPI will see.
	@# Use ``re`` (stdlib) rather than ``packaging.utils`` because the smoke
	@# venv only has the wheel's runtime deps; ``packaging`` may not be
	@# transitively present.
	@cd /tmp && $(VENV)/bin/python -c "\
import re, belt; \
from pathlib import Path; \
wheels = list((Path('$(CURDIR)') / 'dist').glob('agent_belt-*.whl')); \
assert len(wheels) == 1, f'expected exactly one wheel in dist/, found: {wheels}'; \
m = re.match(r'agent_belt-(.+?)-py3-none-any\.whl', wheels[0].name); \
assert m, f'cannot parse wheel filename: {wheels[0].name}'; \
expected = m.group(1); \
assert belt.__version__ == expected, \
    f'BUG: belt.__version__ ({belt.__version__}) does not match wheel filename version ({expected})'; \
print('OK: belt.__version__ =', belt.__version__)"

# ── Evaluation ──

.PHONY: eval eval-score eval-dry-run

eval: ## Run full evaluation (run + score + aggregate)
	belt eval examples/scenarios/ $(ARGS)

eval-score: ## Re-score existing outcomes (OUTCOMES=path)
	belt score $(if $(OUTCOMES),--outcomes $(OUTCOMES)) $(ARGS)
	belt aggregate $(if $(OUTCOMES),--outcomes $(OUTCOMES)) $(ARGS)

eval-dry-run: ## List matching scenarios without executing
	belt eval examples/scenarios/ --dry-run $(ARGS)
