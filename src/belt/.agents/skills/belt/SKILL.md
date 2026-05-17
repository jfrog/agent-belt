---
name: belt
description: Operate the belt CLI to evaluate headless coding agents (Claude Code, Cursor, Codex, Gemini, and others) end to end. Use when the user asks to write or run eval scenarios, compare agents, score outputs with rules or LLM judges, register a new agent adapter, interpret reports or benchmark cards, or set up evals in CI. Also use when the user mentions agent-belt, scenario JSON, BELT_ env vars, llm_scorer_instruction, llm_scorer_evidence_files, TurnExpectation, or benchmark cards.
---

# agent-belt

`belt` is a CLI that evaluates headless coding-agent CLIs by running multi-turn scenarios against
them and scoring the results with rule-based checks plus optional LLM judges. The `belt` console
script is the only public surface - never import internals.

## 1. Verify install before doing anything

```bash
belt doctor
```

Checks Python, registered agents (auth + reachability), LLM scoring providers (cloud keys + Ollama),
and which `belt` clone the command resolves to. If `doctor` is unhappy, fix what it reports
before attempting anything else - most user-reported problems are solved by reading its output.

```bash
belt agent list           # registered agents (entry-point discovered)
belt agent info <name>    # capabilities of one agent (cli_options, env vars, fields it supports)
```

## 2. Run an evaluation

`belt eval` chains **run → score → aggregate** in one command. Start here.

```bash
belt quickstart                                                              # auto-detect, single rules-only
belt eval examples/scenarios/showcase --modes rules --tags real-runnable     # whole runnable showcase
belt eval my-scenarios/ --modes rules,llm --workers 3                        # rules + LLM judge, parallel
belt eval my-scenarios/ --dry-run                                            # list matched scenarios, no run
belt eval my-scenarios/ --modes rules --export junit:report.xml              # JUnit report for CI test reporters
```

**`--modes rules`** runs without any judge or API key - use it for fast feedback and CI smoke tests.
**`--modes llm`** (or `rules,llm`) requires an LLM judge configured (see §5).

Scenario filtering is path-relative to the directory passed as `<path>`; if the path itself is a group
(contains `_config.json`), `--scenarios` takes the bare scenario name. Tag filters use AND semantics
across the listed tags.

Subcommand index, common workflows, and progress modes:
[`docs/glossary/CLI.md`](https://github.com/jfrog/agent-belt/blob/main/docs/glossary/CLI.md).
The canonical per-flag reference is `belt <subcommand> --help`.

## 3. Write a scenario

A **group** is a directory containing `_config.json` plus one or more scenario JSON files. The group
config picks the agent and shared workspace settings; each scenario file is a single test case.

```text
my-scenarios/
├── _config.json
├── fix_bug.json
└── explain_arch.json
```

`_config.json` - shared, group-level only:

```json
{
  "agent": "claude-code",
  "default_tags": ["smoke"],
  "working_dir": "../../my-repo",
  "workspace_isolation": "git-worktree",
  "workspace_ref": "HEAD"
}
```

When the codebase under test lives in another repository, or you want to
mount a versioned payload (a skill, a corpus, a plugin bundle) into every
scenario's worktree, swap `working_dir` for `fixture_repo` + `resources`:

```json
{
  "agent": "claude-code",
  "fixture_repo": "https://github.com/example/target-repo.git",
  "fixture_ref": "v1.4.0",
  "resources": [
    {"kind": "file", "source": "../../skills/security-review", "dest": ".skills/security-review", "version": "0.3.1"}
  ]
}
```

The runner clones the repo once per group, installs each resource into
the per-scenario worktree before `agent.setup`, and writes
`resource_lock.json` (with the source SHA-256) alongside the scenario's
outcome artifacts. `fixture_repo` and `working_dir` are mutually
exclusive. Full reference:
[`SCENARIOS.md`](https://github.com/jfrog/agent-belt/blob/main/docs/glossary/SCENARIOS.md#9-external-fixture-repos-and-resources).

Scenario file (`fix_bug.json`) - the `name` field must match the filename without `.json`:

```json
{
  "name": "fix_bug",
  "description": "Fix the off-by-one in slice_window().",
  "tags": ["editing", "L2"],
  "turns": [
    {
      "message": "Fix the off-by-one in slice_window() in src/utils.py. Run pytest to verify.",
      "expect": {
        "no_errors": true,
        "has_reply": true,
        "tools_invoked": ["Edit"],
        "files_modified_any": ["src/utils.py"]
      }
    }
  ]
}
```

### 3.1 Pick the right `expect` field

`expect` is the **deterministic** half of scoring. Each field is independently optional - omit what
you don't need. Common patterns:

| Goal | Field |
|---|---|
| Correct answer appears in reply | `contains: ["132"]` |
| Forbidden phrases stay out | `not_contains: ["I cannot", "as an AI"]` |
| A specific tool was used | `tools_invoked: ["Edit"]` |
| A tool was called with specific args | `tool_args_contain: {"Read": {"path": "pyproject.toml"}}` |
| A tool returned specific content | `tool_result_contains: {"Read": "belt"}` |
| A tool returned content matching a regex | `tool_result_pattern: {"Read": "name\\s*=\\s*\"belt\""}` |
| A subset of tools allowed | `only_used_tools: ["Read", "Edit"]` |
| A file was modified | `files_modified_any: ["src/utils.py"]` (requires workspace isolation) |
| Latency budget | `max_total_seconds: 60.0` |
| Cost budget | `max_cost_usd: 0.10` (agent-specific - `belt agent info <name>` confirms support) |

Full list (semantics, defaults, agent-specific support) and the
authoring guide live in
[`SCENARIOS.md`](https://github.com/jfrog/agent-belt/blob/main/docs/glossary/SCENARIOS.md)
(reference appendix at §11-§13).

### 3.2 Multi-turn and per-scenario judge instructions

Multi-turn scenarios put more than one entry in `turns`. The agent sees the full conversation history.
A turn's `message` may reference prior `TurnOutput` fields via closed-set placeholders --
`{{prev.reply_text}}`, `{{prev.git_diff}}`, `{{prev.tool_sequence}}`, and `{{turn_N.<field>}}` for
explicit indices -- rendered before the agent sees the message; future-turn or unsupported-field
references fail fast. See
[`SCENARIOS.md#61-multi-turn-templating`](https://github.com/jfrog/agent-belt/blob/main/docs/glossary/SCENARIOS.md#61-multi-turn-templating).

To bias the LLM judge for a particular scenario, add `llm_scorer_instruction` at the scenario root -
it is appended to the default judge prompt for that scenario only.

By default the LLM judge sees a structured per-turn summary built from `TurnOutput` fields (reply,
tool sequence, metadata) - not the agent's raw NDJSON transcript. Set `llm_scorer_raw_transcript: true`
on a scenario to additionally append the raw CLI as a low-priority `## Raw CLI Output` section, only
when the evaluation genuinely depends on event-level inspection.

When the rubric or expected-findings document is too large for `llm_scorer_instruction` (capped at
10 000 chars) or must stay out of the agent's worktree, list it in `llm_scorer_evidence_files`. Each
path is resolved relative to the scenario JSON's directory, read into the judge prompt as
`<evidence_file path="...">…</evidence_file>` at priority 3 in the truncation order, and never copied
into the agent's workspace. Path traversal (`..`, absolute paths) and missing files fail the run.

### 3.3 Validate before running

```bash
belt eval my-scenarios/ --dry-run   # parse every scenario, report schema errors, list what would run
```

`--dry-run` is the fastest way to catch malformed JSON, unknown agent names, missing `working_dir`,
or invalid `expect` fields without paying for an actual run.

For real runs, the loader prints a one-line summary
(`Loaded N scenarios across G groups (M malformed skipped)`) and persists the
skipped count on `AggregatedResults.scenarios_skipped`. Pass `--strict` to
abort instead of running on a silently-shrunken fleet -- covers both agent
availability checks and scenario parse failures.

## 4. Choose a scorer

Two scoring modes; combine freely with `--modes rules,llm`:

| Mode | What it does | When to pick it |
|---|---|---|
| `rules` | Evaluates `expect`/`state_expect` deterministically. Zero API calls. | Default. Always run. CI smoke tests. Anything you can express as substring / tool / file / budget assertions. |
| `llm` | An LLM judge scores quality dimensions (correctness, helpfulness, etc.) plus any `llm_scorer_instruction` overrides per scenario. | When the right answer is a quality call, not a substring match. Worth the latency + spend; use rules for everything else. |

Full scoring model - dimensions, default judge prompt, single- vs multi-judge:
[`SCORING.md`](https://github.com/jfrog/agent-belt/blob/main/docs/glossary/SCORING.md).

## 5. Configure an LLM judge

When `--modes llm` is on, `llm.model` must come from one of three layers (CLI > env > yaml). There is
no built-in default; preflight fails with an explicit three-source error if none is set. Rationale:
silently routing every user to OpenAI hides genuine misconfiguration.

```bash
belt eval my-scenarios/ --scorer-arg model=openai/gpt-5.4-mini   # CLI flag, highest priority
export BELT_LLM_MODEL=ollama/gemma4                               # env var
# or set llm.model in belt.yaml (auto-discovered upward from cwd)
```

Provider credentials use the `BELT_` prefix - `BELT_OPENAI_API_KEY`,
`BELT_ANTHROPIC_API_KEY`, `BELT_AZURE_OPENAI_ENDPOINT` + `BELT_AZURE_OPENAI_API_KEY`,
etc. Ollama needs nothing if running on `localhost:11434`. Full provider matrix and the `judges.yaml`
multi-judge schema:
[`CONFIGURATION.md`](https://github.com/jfrog/agent-belt/blob/main/docs/glossary/CONFIGURATION.md).

The model identifier always carries the provider prefix: `openai/...`, `azure/...`, `anthropic/...`,
`ollama/...`. A bare model name fails preflight.

## 6. Register a new agent

belt discovers agents through a Python entry point - never via direct imports. Adding an agent
means writing a `BaseAgentAdapter` subclass and exposing it under the `belt.agents` group in
`pyproject.toml`.

Plugin architecture, entry-point conventions, and the per-extension
authoring guides (agents, scorers, exporters):
[`PLUGGABILITY.md`](https://github.com/jfrog/agent-belt/blob/main/docs/glossary/PLUGGABILITY.md).

Three constraints worth knowing before you write code:

- **`cli_options()` is the contract.** Whatever the adapter declares there is what `-X` accepts on
  the CLI. Empty `cli_options()` is a deliberate "this agent takes no per-call options" - pin
  behaviour through scenario `flags` or env vars listed in `required_env_vars()`, not through `-X`.
- **No behaviour in the constructor.** Adapters are thin plumbing; per-scenario behaviour goes
  through framework flags. If you find yourself wanting `agent_kwargs={"strict": True}`, add a flag
  instead.
- **`required_env_vars()`** is how the adapter declares external secrets. `belt doctor` reads
  this to flag missing vars; never bypass it with ad-hoc env reads.

## 7. Read the report

`belt eval` writes everything to a run directory under `outcomes/` (override with
`--outcomes-dir` or `BELT_OUTCOMES_DIR`). The aggregator produces:

- A console summary on stdout (passes, fails, scores per dimension, exit code).
- A `benchmark-card.json` capturing the full reproducibility manifest - exact agent versions, LLM
  judge config, scenario snapshot, environment.
- Per-turn artifacts: stream NDJSON, captured stdout, score breakdowns.

To view or compare runs:

```bash
belt view <run-dir>                       # pretty per-scenario breakdown
belt compare <run-dir-a> <run-dir-b>      # cross-run diff (e.g. agent A vs agent B)
```

On-disk layout, schema versioning rules, and the benchmark-card
reproducibility manifest:
[`OUTCOMES.md`](https://github.com/jfrog/agent-belt/blob/main/docs/glossary/OUTCOMES.md).

When a scenario's reply is an authentication / rate-limit / refusal / timeout
shape (the agent did not really run, as opposed to the agent ran and answered
wrong), `results.json` and `benchmark-card.json` carry an `agent_errors` block
with per-token counts plus a `vacuous_passes` field for scenarios whose rules
passed only because nothing meaningful happened. The token set is closed -
`authentication_failed`, `rate_limited`, `timeout`, `refused`, `unknown` - so
an automated consumer can branch on it (re-authenticate, back off, surface to
a human) without parsing free-form CLI output. Token reference and JSON
schema:
[`OUTCOMES.md §3 Agent runtime errors`](https://github.com/jfrog/agent-belt/blob/main/docs/glossary/OUTCOMES.md#3-agent-runtime-errors-error_type).

### 7.2 Export results to external destinations

`belt export` re-emits a completed run through a registered exporter so the
same artifacts can land in CI test reporters, BI pipelines, PR summaries, or
vendor dashboards. The same flags are accepted as a chain on `eval` and
`aggregate` (`--export NAME:PATH`, repeatable; `--export-config FILE.yaml`).

```bash
belt export <run-dir> --to junit:report.xml --to csv:results.csv
belt export --to-config exporters.yaml                  # latest run if <run-dir> omitted
```

`belt doctor` lists registered exporters; the live name is whatever the
registry resolves. New exporters register through the `belt.exporters`
entry-point group, mirroring the agent / scorer plugin model.

Authoring guide:
[`PLUGGABILITY.md → Authoring an exporter`](https://github.com/jfrog/agent-belt/blob/main/docs/glossary/PLUGGABILITY.md#9-authoring-an-exporter).

### 7.1 Threshold gating in CI

`--threshold <scorer>/<dimension>:<max-failures>` enforces a per-dimension failure budget. Repeat
the flag for multiple dimensions. With no `--threshold` flag, `eval`/`aggregate` always exit 0
(report only); with at least one, the process exits non-zero if any dimension exceeds its budget.
The exit code is the contract - wire it directly into your CI step.

```bash
belt eval my-scenarios/ --threshold rules/execution:0                       # zero rule failures allowed
belt eval my-scenarios/ --threshold rules/execution:0 --threshold rules/trajectory:10
belt eval my-scenarios/ --threshold llm/execution:0 --llm-fail-on low       # treat LLM "low" as failure
```

CI integration patterns and a worked GitHub Actions example:
[`CI.md`](https://github.com/jfrog/agent-belt/blob/main/docs/glossary/CI.md).

## 8. Critical DON'Ts

These are the mistakes that cost users (and AI agents) the most time. Each one is a real, recurring
failure mode.

- **Don't pass model selection through `-X` to a parameterless agent.** `claude-code`, `gemini` and
  similar adapters intentionally have empty `cli_options()`. Use the scenario's `flags` field (e.g.
  `"flags": ["--model", "X"]`) or set the agent-specific env var listed in `required_env_vars()`.
  `-X model=foo` against a parameterless agent is rejected with an explicit error - don't try to
  "fix" it by editing the adapter.
- **Don't import from `belt` internals.** The only public surface is the `belt` console
  script. Anything under `belt.{commands,runner,scorer,aggregator}` is internal and changes
  without notice. Plugins go through entry points and the documented base classes
  (`BaseAgentAdapter`, `BaseScorer`, `BaseJudgeBackend`).
- **Don't use a bare model name.** `openai/gpt-5.4-mini` is correct; `gpt-5.4-mini` fails preflight.
  Provider prefix is always required.
- **Don't mix read-only and editing scenarios in the same group.** `working_dir` is group-level;
  every scenario in a group inherits it. Split into sibling groups (`<agent>/read-only/` and
  `<agent>/editing/`) instead.
- **Don't bypass entry-point discovery.** `--allow-arbitrary-agent`, `--allow-arbitrary-scorer`,
  and `--allow-arbitrary-exporter` exist for genuinely exceptional cases; using them routinely
  defeats the trust and reproducibility model that the framework is built on.
- **Don't ignore `belt doctor`.** Almost every "it doesn't work" report is something `doctor`
  would have flagged in two seconds. Run it first.
- **Don't claim a value for a number that drifts.** Scenario counts, agent counts, test totals,
  prices - all of these change between releases. Cite the live source (`belt agent list`,
  `make check`, the relevant `pyproject.toml`) instead of pasting a frozen number into docs or
  scripts.
