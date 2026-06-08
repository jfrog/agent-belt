# agent-belt - Guide for AI Coding Agents

This file is the single entrypoint. Read it top-to-bottom and you have
everything you need to make a non-trivial change to this repo.

## 1. What this is

A universal evaluation harness for headless CLI agents. It runs multi-turn
scenarios against any agent that has a CLI, scores the results with a
combination of rule-based checks and LLM judges, and aggregates them into
reports.

**The only public surface is the `belt` console script.** Everything
under `src/belt/{commands,runner,scorer,aggregator}/` is internal -
callers that aren't the CLI must go through documented agent/scorer
extension points.

## 2. Setup & first verification

```bash
uv sync                        # creates .venv, installs locked dev deps
uv run belt doctor             # checks Python, agents, providers, env
make check                     # lint + test - same as CI
```

`belt doctor` is the fastest way to sanity-check the install: it
verifies entry points resolve, agents are reachable, judge providers are
configured, and the active clone is the one your `belt` command
points at (a real foot-gun if you have multiple checkouts).

## 3. Where to start (by task)

| Task | Open this first |
|---|---|
| Add a new agent | `docs/glossary/PLUGGABILITY.md` (Authoring an agent), then `src/belt/agent/base.py` |
| Modify an existing agent's behaviour | `src/belt/agent/<agent>.py` (subclass `BaseAgentAdapter`) |
| Add a new scorer plugin | `docs/glossary/PLUGGABILITY.md` (Authoring a scorer), then `src/belt/scorer/base.py` |
| Add a new rule-based check | `src/belt/scorer/rules/scorer.py` (read neighbours first) |
| Change LLM judging or prompts | `src/belt/scorer/llm/scorer.py` |
| Add an LLM provider | `src/belt/scorer/llm/backend.py` (subclass `BaseJudgeBackend`) |
| Change scenario JSON shape | `src/belt/entities.py` + `docs/glossary/SCENARIOS.md` (reference appendix) |
| Add a CLI flag | `src/belt/commands/<cmd>.py` (argparse only) |
| Add a CLI subcommand | New `commands/<name>.py` (thin) + new module under the right phase library |
| Change run-phase pipeline | `src/belt/runner/phases/` |
| Change threshold/aggregation logic | `src/belt/aggregator/` |
| Add a new exporter (CSV / JUnit / vendor plugin) | `docs/glossary/PLUGGABILITY.md` (Authoring an exporter), then `src/belt/exporter/base.py` |
| Tune the export phase or chain on `eval` / `aggregate` | `src/belt/commands/export.py` (registry + dispatch) |
| Author a scenario | `docs/glossary/SCENARIOS.md` + `examples/scenarios/` |

For deeper navigation see [`docs/glossary/ARCHITECTURE.md`](docs/glossary/ARCHITECTURE.md)
("Where Things Live" table). Don't memorise the file tree - `tree src/belt -L 2`
is the canonical reference.

## 4. Design Principles

Skim [ARCHITECTURE.md → Design Principles](docs/glossary/ARCHITECTURE.md#design-principles)
before your first PR - this list is the working summary. Violating any
is a review block.

1. **Don't make phases import each other.** `runner/`, `scorer/`,
   `aggregator/`, and `exporter/` communicate through files on disk only. The
   contract is `entities.py`, not Python imports.
2. **Don't modify base-class signatures.** `BaseAgentAdapter`, `BaseScorer`,
   `BaseJudgeBackend` - extend by subclassing, never by widening the parent.
3. **Don't put behaviour into agent constructors.** Agents are thin
   plumbing; behaviour lives behind framework flags. If you find yourself
   wanting `agent_kwargs={"strict": True}`, add a flag.
4. **Don't `isinstance`-check entity subclasses.** Optional fields with
   safe defaults - both ends degrade gracefully.
5. **Don't raise raw exceptions from user-facing code paths.** Use a typed
   error from `errors.py` with actionable context.
6. **Don't ship a feature without updating the relevant doc** in
   `docs/glossary/`. The doc and the code change in the same PR.
7. **Don't bypass entry-point discovery** for agents or scorers. The
   `--allow-arbitrary-*` flags exist for exceptional cases only.
8. **Don't interpolate agent / judge / scenario text directly into Rich
   panels or Markdown.** Wrap in `rich_safe` or `md_safe` from
   `belt._safe`. Direct interpolation is an injection bug.
9. **Don't add a safety toggle that defaults to permit.** New behaviour
   gates are default-deny + an explicit `--allow-*` / `BELT_ALLOW_*`
   opt-in.
10. **Don't write `"BELT_*"` env-var names as string literals.**
    Import the constant from `belt.envvars` (or
    `_internal_envvars` for the private set).

## 5. Conventions

- **Read 2-3 neighbouring files before writing new code** - match local
  patterns rather than introducing new ones.
- **Namespace package layout.** Source lives in `src/belt/`; imports
  use the `belt.` prefix (`from belt.entities import ...`).
- **Pydantic for any data that crosses a phase or process boundary.**
  Plain dataclasses are fine for in-phase helpers.
- **Typed errors only.** Define new ones as subclasses of `BeltError`
  in `errors.py` with the `(message, hint)` shape.
- **Output integrity.** Anything written to terminal or markdown that
  contains user/agent input must go through `_safe.py` (`rich_safe`,
  `md_safe`). Never f-string user content directly into a Rich/Markdown
  template.
- **CLI commands are thin.** Each `commands/<name>.py` is argparse setup
  plus a `main() -> int`. Business logic stays in the matching phase library.
- **Lint config.** black + isort + ruff at 120-char line length, type
  hints on public interfaces. `pre-commit run --files <changed>` before
  pushing.

## 6. Test layout

`tests/` mirrors `src/belt/`. When you add a module, add the matching
test file in the mirrored path. CLI command tests live in `tests/commands/`.
Run a focused subset with `pytest tests/<path>` and the full suite with
`make test`.

## 7. Quality gate (before declaring "done")

1. `make check` - passes (lint + tests)
2. `belt doctor` - green for the agents/providers your change touches
3. For agent or CLI changes: `uv sync` then
   `uv run belt agent list` and `uv run belt agent info <name>` - entry points
   actually resolve. Unit tests can pass while the CLI is broken; this
   step catches it.
4. For behaviour changes: a new scenario or scorer test that exercises the
   new path. "Unverified code is not done."

## 8. End-to-end ownership

When you hit a blocker inside this project - tests failing, CLI broken,
config not loading, scorer erroring - diagnose and fix it. "It was already
broken on main" is not an acceptable handoff. The boundary is everything
under this repo plus its declared dependencies. Issues in an external
agent CLI, a cloud provider, or CI infrastructure are out of scope -
report and stop.

## 9. Durable claims (no exact counts that aren't auto-generated)

User-facing docs (`README.md`, `AGENTS.md`, `CONTRIBUTING.md`,
`docs/**/*.md`, `examples/**/*.md`, in-code comments, PR/commit
templates) **must not** assert exact numbers that change every release.
A claim that drifts on the next change is worse than no claim at all -
it actively misleads readers.

### Forbidden patterns

| Pattern | Why it drifts | Use instead |
|---|---|---|
| `1670 passed`, `62/62 checks`, `10/12 passed` | Test totals shift on every PR | Link to the CI status badge / "see CI" / "make check" |
| `Currently 7 agents are supported` | New agents land continuously | `belt agent list` (the canonical list) |
| `Depends on N packages` | `pyproject.toml` evolves | Link to [`pyproject.toml`](pyproject.toml) |
| `As of 2026-04-01, …` | Date stamps stale instantly | Drop the date or rephrase qualitatively |
| `~$0.42 per scenario` | Pricing changes per-model and per-month | Use a range (`a few cents`, `cents to dollars`) or omit |
| `Python 3.11.4 / pytest 8.0.1` | Versions in tooling output | Cite the project's declared minimum (e.g. `requires-python = ">=3.11"`) |

### Allowed exceptions

- **Auto-generated tables** that ship with the code path that produces
  them (e.g. AGENT-FEATURES.md is hand-maintained but its parity-critical
  rows are gated by `tests/agent/test_agent_parity.py`).
- **Configuration defaults** that are also constants in code
  (`SCHEMA_VERSION`, `EXAMPLE_LLM_MODEL`, `_DEFAULT_BASE_URL`). Cite the
  source-of-truth module so a reader can re-derive the value.
- **Schema versions** and **exit codes** - these are part of the
  contract and bumped deliberately.

### Reviewer checklist

When reviewing a doc PR, scan for: bare integer counts ("N tests", "N
agents", "N flags"), prices, version-specific costs, dated assertions,
and proper-noun-shaped lists that look like an exhaustive enumeration
(`(claude, gemini, codex)`). Replace with a link, a range, or a CLI
command that prints the live answer.

## 10. Reference docs

| Topic | Doc |
|---|---|
| Architecture, design principles, where-things-live (read first) | [ARCHITECTURE.md](docs/glossary/ARCHITECTURE.md) |
| Scenario authoring + JSON schema reference | [SCENARIOS.md](docs/glossary/SCENARIOS.md) |
| Scoring (rules + LLM judges, multi-judge, thresholds) | [SCORING.md](docs/glossary/SCORING.md) |
| Plugin architecture (agents, scorers, exporters) | [PLUGGABILITY.md](docs/glossary/PLUGGABILITY.md) |
| Layered configuration & env vars + security toggles + trust model | [CONFIGURATION.md](docs/glossary/CONFIGURATION.md) |
| Sandbox providers + kernel invariants, network policy | [SANDBOXING.md](docs/glossary/SANDBOXING.md) |
| Threat model, trust boundaries, controls, residual risk | [SECURITY-MODEL.md](docs/glossary/SECURITY-MODEL.md) |
| CLI subcommand index, workflows, progress modes | [CLI.md](docs/glossary/CLI.md) |
| CI integration, threshold gating, agent install/auth recipes | [CI.md](docs/glossary/CI.md) |
| On-disk artifacts, schema versioning, benchmark card | [OUTCOMES.md](docs/glossary/OUTCOMES.md) |
| Built-in agent feature matrix | [AGENT-FEATURES.md](docs/glossary/AGENT-FEATURES.md) |
