# agent-belt

A seat belt for the agents you ship. Run reproducible multi-turn scenarios
against the *binary your users actually run* - Claude Code, Cursor, Codex,
Gemini, Copilot, opencode, Goose, or your own CLI. Score with rule checks,
workspace diffs, and a separate LLM judge. Pin variance down with
repeat trials, paraphrase families, and multi-judge consensus - so a
green check actually means something on a stochastic stack.

## Disclaimer

This tool is provided as-is with no warranty. Running untrusted agents
or scenarios can cause real, irreversible damage to your system. Each
scenario drives a real agent CLI that runs as your user - a malicious
scenario can modify dotfiles, SSH config, or git hooks; run destructive
or data-exfiltrating commands; or reach the network to pull
instructions, upload source, or push to external repos. Only run
scenarios you trust. We take no responsibility for damages.

## Before You Begin

You need **Python 3.13** (3.11+ supported) and at least one
CLI agent installed and authenticated (run it once interactively to sign
in). One agent is enough to get started.

## Install

```bash
pip install agent-belt
belt doctor   # verify agents, auth, and LLM scoring providers
```

For development setup, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Quick Start

```bash
belt quickstart              # auto-detects first available agent
belt quickstart claude-code  # or specify one
```

This validates the agent, runs a single scenario with rules-only scoring
(no API key needed), and prints next steps.

For more control over the bundled showcase (no source clone needed):

```bash
belt eval --bundled showcase --modes rules \
  --tags real-runnable --allow-external-working-dir                # whole runnable showcase
belt eval --bundled showcase --scorer-arg model=openai/gpt-5.4-mini  # + LLM judge (cloud)
belt eval --bundled showcase --scorer-arg model=ollama/gemma4        # + LLM judge (local, no API key)
belt eval --bundled showcase --modes rules --workers 3               # parallel
belt eval --bundled showcase --progress live --workers 3             # live TUI
```

`--bundled <NAME>` resolves to the scenarios shipped inside the wheel
(`belt/_bundled_examples/scenarios/<NAME>/`), so the same one-liner
works after `pip install agent-belt` without cloning the source repo.
Use `belt eval --bundled` to target the whole bundled tree, or
`belt eval <PATH>` to point at your own scenarios directory.

The showcase is a schema-coverage reference, not a green-CI baseline -
the unfiltered command intentionally includes `dry-run-only` scenarios
that document fields no generic CLI agent surfaces. See
[`examples/scenarios/showcase/README.md`](examples/scenarios/showcase/README.md)
for the full first-run guide and the per-group index.

See the [CLI guide](docs/glossary/CLI.md) for the subcommand index and
common workflows; `belt <cmd> --help` is the canonical flag reference.

For a real-world example, see [`examples/`](examples/README.md) - fixture
repos in Python, TypeScript, and Go with scenarios across L1-L4 difficulty
levels, including cross-agent comparison data.

## Why agent-belt

The eval ecosystem is real, and most of it solves a different problem.
Generic LLM-eval frameworks score a model's output on a single prompt.
Observability platforms (LangSmith, Langfuse, Braintrust) eval the
function you wrap with their SDK - not the binary your users
`npm install`'d. Trajectory frameworks build the agent loop themselves
and score it from the inside. Coding-agent benchmarks freeze a curated
dataset and score the field on someone else's repo.

agent-belt closes the obvious gap: take the binary your users actually
run, point it at a real workspace with your team's skills and MCP servers
wired up, hand it scenarios that mirror the use cases they drive it
through, repeat each one to measure reliability instead of luck, and
ship a verdict.

Three things are distinctive:

1. **The black-box CLI is the unit.** Not the model. Not a wrapper. Not
   a solver class. The thing you `pip install`'d, `brew install`'d, or
   `curl | bash`'d. agent-belt runs it as a subprocess; your MCP servers,
   skills, and auth stay untouched. The same harness drives any
   CLI agent - same scenarios, same scoring, same verdict format.
2. **Variance gets pinned down on three axes.** `--trials N` runs the
   same scenario k times and reports `pass^k` for k = 1, 3, 8. Tag a
   *family* of paraphrased scenarios and the aggregator gives you
   family-level pass rate. A multi-judge `--scorer-config` requires
   consensus before a verdict counts. Each axis on its own is a partial
   signal - together they're a regression detector for stochastic stacks.
3. **Your judge runs separately.** You pick its provider, model, persona,
   dimensions, and per-scenario instructions. It runs out-of-process from
   the agent under test - never as another skill inside the same Claude
   Code or Cursor session that's doing the work. OpenAI, Anthropic, Azure,
   `ollama/llama3.3` on your laptop, vLLM on your own GPUs. The plumbing
   is in the box.

## How agent-belt compares

| You want to evaluate | Use |
|---|---|
| A single LLM prompt's output (model-level, no agent loop) | DeepEval, Promptfoo, lm-evaluation-harness, OpenAI Evals |
| A function in your app you've wrapped with a vendor SDK | LangSmith, Langfuse, Phoenix, Braintrust |
| A research agent loop you've built inside a framework | Inspect AI |
| The field of coding agents on a frozen public benchmark | SWE-bench, Aider Polyglot, Terminal-Bench |
| **The CLI agent your users actually run, on your repo, against your scenarios** | **agent-belt** |

## Evaluate Your Own Use Cases

agent-belt sends inputs to your agent headlessly. The agent CLI must be
installed and authenticated on the host running agent-belt. Beyond that,
what the agent can do per scenario depends on where the agent discovers
its configuration:

- **Discoverable from the working directory** - skills, slash commands,
  plugins, and MCP servers that the agent auto-loads from the worktree
  root (e.g. Claude Code's `.claude/` and `.mcp.json`) can ship with the
  fixture. See
  [`examples/fixtures/folio/`](examples/fixtures/folio/) for a worked
  example exercising all four surfaces.
- **User-level only** - configuration the agent reads from a fixed user
  path (e.g. `~/.cursor/`, `~/.claude/`, OS keychains) must be set up
  out of band on the host running agent-belt.
- **Repository access** - the agent must be able to read and write the
  worktree agent-belt creates per scenario (default behavior).

### Create a Scenario

Scenarios live in a **group directory**. Each group has a `_config.json`
that holds shared settings - which agent to use, workspace isolation,
custom scoring dimensions - so individual scenarios stay focused on the
test case itself. This matters when you have many scenarios: instead of
repeating `"agent": "claude-code"` and `"working_dir": "../../my-repo"`
in every file, you set it once in `_config.json`.

```text
my-scenarios/
├── _config.json          # shared: agent, workspace, tags
├── fix_bug.json          # scenario 1
└── explain_arch.json     # scenario 2
```

`_config.json` - shared group configuration
([full schema](docs/glossary/SCENARIOS.md#2-group-config-_configjson)):

```json
{
  "agent": "claude-code",
  "working_dir": "../../path/to/your/repo",
  "default_tags": ["my-project"]
}
```

For a fully worked group with MCP servers, default skills, plugin
skills, plugin commands, and slash commands all wired into one fixture,
see
[`examples/scenarios/experience/programmatic-setup-claude/`](examples/scenarios/experience/programmatic-setup-claude/) -
seven runnable scenarios over a single `_config.json`. Run it with:

```bash
belt eval examples/scenarios/experience/programmatic-setup-claude --modes rules
```

See [Scenarios](docs/glossary/SCENARIOS.md) for the full authoring guide
and the field-by-field reference appendix.

## Documentation

The `docs/glossary/` directory holds the entire reference. One topic per
file, no duplication:

| Topic | Link |
|---|---|
| Architecture, design principles, where-things-live | [ARCHITECTURE.md](docs/glossary/ARCHITECTURE.md) |
| Scenario authoring + JSON schema reference | [SCENARIOS.md](docs/glossary/SCENARIOS.md) |
| Scoring (rules + LLM judges, multi-judge, thresholds) | [SCORING.md](docs/glossary/SCORING.md) |
| Layered configuration, env vars, security toggles | [CONFIGURATION.md](docs/glossary/CONFIGURATION.md) |
| CLI subcommand index, workflows, progress modes | [CLI.md](docs/glossary/CLI.md) |
| CI integration, threshold gating, agent install/auth recipes | [CI.md](docs/glossary/CI.md) |
| Plugin architecture (agents, scorers, exporters) | [PLUGGABILITY.md](docs/glossary/PLUGGABILITY.md) |
| On-disk artifacts, schema versioning, benchmark card | [OUTCOMES.md](docs/glossary/OUTCOMES.md) |
| Built-in agent feature matrix | [AGENT-FEATURES.md](docs/glossary/AGENT-FEATURES.md) |

agent-belt also ships a `SKILL.md` inside the wheel so an AI coding
agent on your machine can author scenarios, run `belt eval`, and
interpret a verdict without leaving your IDE. One-time symlink setup is
in [CONTRIBUTING.md](CONTRIBUTING.md#enable-the-bundled-skillmd-for-your-ai-coding-agent).

## Not for

- Benchmarking the LLMs themselves.
- Wrapping or replacing the agent loop. agent-belt invokes your
  *already-installed, already-authenticated* agent CLI as a subprocess.
- IDE integration. agent-belt is a headless CI-friendly harness; the
  agents it drives have their own IDE plugins.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, the test matrix, plugin
authoring, and what gates a PR.

## Release Notes

The release notes are available [here](https://github.com/jfrog/agent-belt/releases).

## License

Apache 2.0 - see [LICENSE](LICENSE) and [NOTICE](NOTICE) for third-party
attribution. Contributions require signing the
[JFrog CLA](https://jfrog.com/cla/); see [CONTRIBUTING.md](CONTRIBUTING.md).
