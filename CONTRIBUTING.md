# Contributing to agent-belt

## Getting Started

We use [uv](https://docs.astral.sh/uv/) for dependency management - it gives
you a reproducible, locked dev environment in one command. (Pip-based setup
still works as a fallback; see "Pip fallback" below.)

```bash
git clone https://github.com/jfrog/agent-belt.git
cd agent-belt
uv sync                              # creates .venv, installs locked dev deps
uv run pre-commit install            # enables gitleaks, black, isort on every commit
uv run make check                    # lint + test - should pass before you start
```

Then either `source .venv/bin/activate` once, or prefix every command with
`uv run` (`uv run belt --version`, `uv run pytest`, ...).

### Pip fallback

If you don't have uv installed:

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
pre-commit install
make check
```

The pip path uses the same `[project.optional-dependencies] dev` set that
`uv sync` reads via `[dependency-groups] dev`; we keep both in sync.

## Development Workflow

1. Create a branch from the latest `main` - this repository’s **default branch** on GitHub is `main` (not `master`).
2. Make your changes
3. Run `make check` (lint + test must pass)
4. Open a pull request **into** `main`

Apply a release-note label (`new feature`, `improvement`, `bug`,
`breaking change`, or `ignore for release`) so the auto-generated release
notes (see [`.github/release.yml`](.github/release.yml)) categorise your PR
correctly. Unlabeled PRs land under "Other Changes."

## Code Style

- **black** for formatting (120-char line length)
- **isort** with black profile
- **ruff** for linting
- Run `make format` to auto-format before committing

Pre-commit hooks enforce these on `git commit`.

## Testing

### Fast tests (no API keys, no agents)

Unit tests run entirely offline - no LLM keys or agent CLIs required.

```bash
make check                                   # lint + unit tests (CI runs this)
make test                                    # unit tests only
python -m pytest tests/scorer/test_rules.py  # specific file
python -m pytest tests/ -k "test_name"       # specific test
```

`make check` must pass before pushing. CI enforces this on every PR.

### Integration tests (requires agent CLIs + API keys)

Scenario runs execute real agent CLIs and cost real money. These are not part of `make check` -
run them manually when changing agents, the runner, or scoring logic.

```bash
belt agent list                                                # the canonical list of bundled agents
belt eval examples/scenarios/showcase --dry-run                # validate scenario parsing (free)
belt eval examples/scenarios/showcase --modes rules \
  --tags real-runnable --allow-external-working-dir            # real run (a few cents per scenario)
```

The bare `belt eval examples/scenarios/showcase` command intentionally
includes scenarios tagged `dry-run-only` (schema-coverage examples that
don't run cleanly against a generic CLI agent). See
[`examples/scenarios/showcase/README.md`](examples/scenarios/showcase/README.md)
for the recommended first-run command and per-group index.

**Prerequisites:** at least one of the agent CLIs reported by
`belt agent list` installed and authenticated. `belt doctor`
reports per-agent readiness and the auth signal it detected.

See [docs/glossary/SCENARIOS.md](docs/glossary/SCENARIOS.md) for scenario authoring.

## What runs where

A check is enforced in one of three places. Use this matrix to know what gates a PR.

| Check | Local (`make check`) | CI (every PR) | Frogbot PR scan (gated) |
|---|:---:|:---:|:---:|
| `black` formatting | ✅ | ✅ | - |
| `isort` import order | ✅ | ✅ | - |
| `ruff` linting | ✅ | ✅ | - |
| Unit tests (Python 3.13, recommended local default) | ✅ | ✅ | - |
| Unit tests (Python 3.11 / 3.12 / 3.14) | - (only your local version) | ✅ matrix | - |
| `build` (sdist + wheel can be packaged) | - | ✅ | - |
| `CLAssistant` (CLA signed) | - | ✅ | - |
| Frogbot SCA (vulnerable deps; High/Critical fails the PR) | - | - | ✅ |
| Frogbot secrets scan (leaked credentials in diff) | - | - | ✅ |

- **`make check`** is what you run before pushing. CI re-runs the same target across
  the `lint` + `test (3.11..3.14)` matrix, so a clean local run on the recommended
  3.13 toolchain is a strong signal CI will be green.
- **CI** runs on every push and PR. Branch protection on `main` requires every check above to pass.
- **Frogbot PR scan** runs on `pull_request_target` and is gated by the `frogbot` GitHub environment
  - a maintainer must approve external PRs before the scan executes. This prevents secret
  exfiltration from forked PRs. The scan fails the PR on High/Critical vulnerabilities.
- **Pre-commit hooks** (`gitleaks`, `black`, `isort`, `bandit`) run on `git commit` as an additional
  early-feedback layer; they overlap with `make check`.

## Credentialed CI workflows

Workflows that need LLM provider keys, real agent CLIs, or other secrets
are gated behind a maintainer-applied label so external PRs can't exfiltrate
credentials. The convention (proven on
[`jfrog/jfrog-cli`](https://github.com/jfrog/jfrog-cli/tree/master/.github/workflows)):

1. Trigger on `pull_request_target: types: [labeled]` (in addition to `push`).
2. Gate every job: `if: contains(github.event.pull_request.labels.*.name, 'safe to test') || github.event_name == 'push'`.
3. Checkout the PR head SHA: `ref: ${{ github.event.pull_request.head.sha }}`.
4. Pin every third-party action to a commit SHA, not a moving tag.

The `safe to test` label is auto-stripped after each run by
[`.github/workflows/removeLabel.yml`](.github/workflows/removeLabel.yml), so
each credentialed run requires a fresh maintainer review of the diff.

## Releases

Releases are tag-driven. Maintainers cut a release by pushing an annotated `v*` tag from `main`:

- `vX.Y.Z-rc1` (or any `v*-*` pre-release): builds, attests provenance, and publishes to **TestPyPI** for rehearsal.
- `vX.Y.Z` (clean PEP 440): builds, attests, publishes to **PyPI**, and creates a GitHub Release.

Release notes are generated automatically from PR labels (see
[`.github/release.yml`](.github/release.yml)) - apply at least one
release-note label (`new feature`, `improvement`, `bug`, `breaking change`,
or `ignore for release`) per PR. Operational details (Trusted Publisher
registration, GitHub Environments, branch-protection setup) live in a
private maintainer doc.

## Adding an Agent, Scorer, or Exporter

All three extension points are documented in
[docs/glossary/PLUGGABILITY.md](docs/glossary/PLUGGABILITY.md). For
third-party packages, register via the matching `belt.agents` /
`belt.scorers` / `belt.exporters` Python entry-point group -
no core changes needed.

## Project Layout

- `src/` - core library (agents, runner, scorer, aggregator)
- `tests/` - unit and integration tests
- `examples/` - scenario examples and custom agent template
- `docs/` - architecture, guides, schema reference
- `plugins/` - directory reserved for in-tree plugins, one vendor per
  subdirectory, each shipping as an independently-versioned PyPI package.
  Empty on the initial release; contributor-added plugins land here.

Import convention: `from belt.entities import TurnOutput` (full namespace).

### In-tree plugins (`plugins/`)

Each subdirectory is a vendor, importable as `belt_<vendor>`, published
as `belt-<vendor>`. Layout is flat (not split by extension type) - a
single plugin can register into `belt.agents`, `belt.scorers`, and/or
`belt.exporters` from the same package. When `plugins/` is non-empty,
CI installs every package under it alongside core and runs each plugin's
pytest suite (`make check-plugins`, dedicated `plugins` job in
`.github/workflows/test.yml`) - separate from the core matrix so plugin
failures don't block core PRs. When `plugins/` is empty (as on the
initial release), the `plugins` job short-circuits cleanly.

## Enable the Bundled `SKILL.md` for Your AI Coding Agent

agent-belt ships a `SKILL.md` inside the wheel that teaches AI coding
agents (Cursor, Claude Code, Codex, Gemini, Copilot, and others) how to
write scenarios, run evals, configure judges, register agents, and read
reports. After `pip install agent-belt`, the file lives at:

```text
<site-packages>/belt/.agents/skills/belt/SKILL.md
```

To make your project's agent discover it, symlink the bundled directory
into one of the standard skill locations your agent already scans
(`.agents/skills/`, `.cursor/skills/`, `.claude/skills/`, or
`.codex/skills/`):

```bash
python -c "import belt, pathlib; print(pathlib.Path(belt.__file__).parent / '.agents/skills/belt')" \
  | xargs -I {} ln -sfn {} .agents/skills/belt
```

Because the skill ships *inside* the wheel, it always matches the
installed agent-belt version - no separate update step.

## Reporting Issues

- **Bugs** - file an issue with steps to reproduce, expected vs. actual behavior
- **Security vulnerabilities** - see [SECURITY.md](SECURITY.md) for private reporting
- **Feature requests** - open an issue describing the use case

## Contributor License Agreement (CLA)

Before your first contribution can be merged, you must sign the
[JFrog Contributor License Agreement](https://jfrog.com/cla/). This is a one-time requirement per
contributor; once signed, it applies to all future contributions across JFrog open source projects.

## License

This project is licensed under the [Apache License 2.0](LICENSE) (SPDX: `Apache-2.0`). By
contributing, you agree that your contributions will be licensed under the same terms.
