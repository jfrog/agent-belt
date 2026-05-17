# Examples

Ready-to-run evaluation scenarios for belt. Two namespaces:

- [`scenarios/experience/`](scenarios/experience/) - full code-editing
  campaigns against realistic fixtures (L1-L4 difficulty levels, one folder
  per agent × fixture pair).
- [`scenarios/showcase/`](scenarios/showcase/) - one capability per group;
  every field of `TurnExpectation`, `StateExpectation`, and `GroupConfig`
  is demonstrated by at least one scenario.

## Prerequisites

You need **agent-belt** installed and **at least one agent CLI** available.
Check which agents are available on your system:

```bash
belt agent list
```

> **Note:** Editing scenarios call the agent's API and cost real money. L1
> read-only scenarios are cheaper but still use API calls. Only `--dry-run`
> is free.

## 1. Quick Start: Verify Your Setup

**Step 1 - Dry run (free, no API calls):**

```bash
belt eval examples/scenarios/showcase --dry-run
```

The output is a table listing every showcase scenario with its tags and turn
count. Use this to confirm belt can discover and parse the bundled
examples before spending API credits.

**Step 2 - First real run (one read-only scenario; a few cents):**

```bash
belt eval examples/scenarios/showcase --modes rules \
  --scenarios correctness/correctness_basic
```

Asks the agent (default: `cursor`) "what is 12 multiplied by 11?" and asserts
the answer contains `132`. Exit code is `0` only when every rule passes.

**Step 3 - Editing scenario with workspace isolation (a few cents):**

```bash
belt eval examples/scenarios/showcase --modes rules \
  --allow-external-working-dir \
  --scenarios editing-workspace/files_modified
```

Asks the agent to fix the divide-by-zero bug in
[`fixtures/sample-project/src/calculator.py`](fixtures/sample-project/src/calculator.py).
The agent edits files in an isolated git worktree - the fixture stays
pristine.

> **Why `--allow-external-working-dir`?** Editing scenarios reuse the shared
> [`examples/fixtures/`](fixtures/) repos, which live as siblings of
> `examples/scenarios/`. belt requires this flag whenever a group's
> `working_dir` resolves outside the scenarios root, as a safety guard
> against auto-init in unrelated directories.

## 2. Layout

| Subdirectory | What it holds |
|---|---|
| [`scenarios/experience/`](scenarios/experience/) | Experience-campaign groups - one per fixture × agent pair, scoped L1-L4 |
| [`scenarios/showcase/`](scenarios/showcase/) | One group per schema capability - the canonical reference for every expectation field |
| [`fixtures/`](fixtures/) | Self-contained codebases used by editing scenarios |
| [`scorer-config/`](scorer-config/) | Sample LLM scoring configurations (e.g. multi-judge panels) |
| [`custom-agent/`](custom-agent/) | Minimal installable agent plugin template |

## 3. Showcase: Schema Features by Example

Every field of `TurnExpectation`, `StateExpectation`, and `GroupConfig` is
demonstrated by at least one scenario in
[`scenarios/showcase/`](scenarios/showcase/). Use the table to find the
example for a specific field.

Scenarios tagged `real-runnable` execute end-to-end with `--modes rules`.
Scenarios tagged `dry-run-only` document fields the runtime can't reliably
force in CI (cost reporting on agents that don't track it, multi-agent
handoffs without a sub-agent harness, …); their `description` field
documents how to enable them manually.

### 3.1. `TurnExpectation` - correctness

| Field | Demonstrating scenario |
|---|---|
| `no_errors`, `has_reply`, `contains`, `not_contains` | [`correctness/correctness_basic.json`](scenarios/showcase/correctness/correctness_basic.json) |

### 3.2. `TurnExpectation` - tool trajectory

| Field | Demonstrating scenario |
|---|---|
| `tools_invoked`, `tools_invoked_any` | [`tool-trajectory/tools_basic.json`](scenarios/showcase/tool-trajectory/tools_basic.json) |
| `tools_invoked_in_order` | [`tool-trajectory/tools_in_order.json`](scenarios/showcase/tool-trajectory/tools_in_order.json) |
| `only_used_tools` | [`tool-trajectory/tools_only_used.json`](scenarios/showcase/tool-trajectory/tools_only_used.json) |
| `forbidden_tools` | [`tool-trajectory/tools_forbidden.json`](scenarios/showcase/tool-trajectory/tools_forbidden.json) |
| `tool_args_contain` | [`tool-trajectory/tool_args.json`](scenarios/showcase/tool-trajectory/tool_args.json) |
| `tool_result_contains`, `tool_result_pattern` | [`tool-trajectory/tool_result.json`](scenarios/showcase/tool-trajectory/tool_result.json) |

### 3.3. `TurnExpectation` - agent capabilities (dry-run-only)

| Field | Demonstrating scenario |
|---|---|
| `skills_invoked` | [`agent-capabilities/skills_invoked_dry_run.json`](scenarios/showcase/agent-capabilities/skills_invoked_dry_run.json) |
| `has_thinking` | [`agent-capabilities/has_thinking_dry_run.json`](scenarios/showcase/agent-capabilities/has_thinking_dry_run.json) |

### 3.4. `TurnExpectation` - budgets and latency

| Field | Demonstrating scenario |
|---|---|
| `max_llm_turns`, `max_tool_calls` | [`budgets-latency/budgets.json`](scenarios/showcase/budgets-latency/budgets.json) |
| `max_cost_usd` | [`budgets-latency/cost_budget_dry_run.json`](scenarios/showcase/budgets-latency/cost_budget_dry_run.json) (dry-run-only) |
| `max_total_seconds` | [`budgets-latency/latency_slo.json`](scenarios/showcase/budgets-latency/latency_slo.json) |
| `max_ttfe_seconds`, `max_ttft_seconds`, `max_ttlt_seconds` | [`budgets-latency/latency_streaming_dry_run.json`](scenarios/showcase/budgets-latency/latency_streaming_dry_run.json) (dry-run-only) |

### 3.5. `TurnExpectation` - error types (dry-run-only)

| Field | Demonstrating scenario |
|---|---|
| `error_type_is` | [`error-types/error_type_negative_dry_run.json`](scenarios/showcase/error-types/error_type_negative_dry_run.json) |

### 3.6. `TurnExpectation` - workspace edits

| Field | Demonstrating scenario |
|---|---|
| `files_modified_any`, `files_modified_exact`, `files_not_modified`, `git_diff_contains` | [`editing-workspace/files_modified.json`](scenarios/showcase/editing-workspace/files_modified.json) |

### 3.7. `StateExpectation`

| Field | Demonstrating scenario |
|---|---|
| `files_exist`, `files_contain`, `files_not_exist`, `capture_git_diff` | [`editing-workspace/state_files.json`](scenarios/showcase/editing-workspace/state_files.json) |

### 3.8. `GroupConfig`

| Field | Demonstrating `_config.json` |
|---|---|
| `working_dir`, `workspace_isolation`, `workspace_ref` | [`editing-workspace/_config.json`](scenarios/showcase/editing-workspace/_config.json) |
| `default_tags`, `llm_dimensions`, `llm_dimensions_extend_defaults` | [`group-config-fields/_config.json`](scenarios/showcase/group-config-fields/_config.json) |
| `fixture_repo`, `fixture_ref`, `resources` | [`external-fixture/_config.json`](scenarios/showcase/external-fixture/_config.json) |
| `DimensionDef.kind` (`ternary` / `binary`), `allow_inconclusive` | [`verdict-scales/_config.json`](scenarios/showcase/verdict-scales/_config.json) |

### 3.9. `Scenario` top-level

| Field | Demonstrating scenario |
|---|---|
| `llm_scorer_instruction`, multi-turn (`turns` length > 1) | [`correctness/multi_turn_with_judge.json`](scenarios/showcase/correctness/multi_turn_with_judge.json) |
| `llm_scorer_raw_transcript` (opt-in raw CLI in judge prompt) | [`correctness/llm_scorer_raw_transcript_optin.json`](scenarios/showcase/correctness/llm_scorer_raw_transcript_optin.json) |
| `llm_scorer_evidence_files` (judge-only rubric / ground-truth files) | [`correctness/llm_scorer_evidence_files.json`](scenarios/showcase/correctness/llm_scorer_evidence_files.json) (companion: [`llm_scorer_evidence_files.md`](scenarios/showcase/correctness/llm_scorer_evidence_files.md)) |

### 3.10. Run the whole real-runnable showcase

```bash
belt eval examples/scenarios/showcase \
  --modes rules \
  --tags real-runnable \
  --allow-external-working-dir
```

Every scenario tagged `real-runnable` returns green with `--modes rules` on
any agent the showcase config targets (default: `cursor`). To verify the
dry-run-only scenarios parse and list:

```bash
belt eval examples/scenarios/showcase --dry-run --tags dry-run-only
```

## 4. Experience Campaigns

Realistic codebases with intentional bugs at four difficulty levels. One
group per (fixture, agent) pair so agent-specific flags (e.g. Claude's
`--permission-mode acceptEdits`) live with the agent that needs them.

| Group | Fixture | Language | Agent | What it tests |
|---|---|---|---|---|
| [`tasktracker-claude/`](scenarios/experience/tasktracker-claude/) | Python CLI tool | Python | claude-code | File locking, CLI flags, formatters, data models |
| [`tasktracker-cursor/`](scenarios/experience/tasktracker-cursor/) | Python CLI tool | Python | cursor | Same coverage as tasktracker-claude - proves the cross-agent comparison story end-to-end |
| [`bookstore-api-claude/`](scenarios/experience/bookstore-api-claude/) | Express REST API | TypeScript | claude-code | SQL injection, auth, pagination, validation |
| [`urlshortener-claude/`](scenarios/experience/urlshortener-claude/) | URL shortener | Go | claude-code | Concurrency, URL validation, Dockerfile, graceful shutdown |
| [`programmatic-setup-claude/`](scenarios/experience/programmatic-setup-claude/) | MCP server + `.claude/` provisioning | Python | claude-code | MCP, skill, slash command, project-local plugin (with bundled command and skill) - all loaded from files in the workspace |

### 4.1. Difficulty levels

| Level | What the agent does | Workspace isolation | Cost class | Example |
|---|---|---|---|---|
| **L1** | Read files, explain / find bugs | No edits | low (cents) | "Find the concurrency bug in `storage.py`" |
| **L2** | Fix one bug in one file | git worktree | low (cents) | "Fix the off-by-one in `formatters.py`" |
| **L3** | Multi-file feature | git worktree | low-medium | "Add a `delete` command (cli + storage + tests)" |
| **L4** | 3-turn: investigate → fix → test | git worktree | medium | "Find all bugs, fix the worst, add a test" |

Cost classes are rough - actual cost depends on the model, prompt size, and
how much tool-using the agent does. Use `--modes rules` to skip LLM judges
when you only need rule-based scoring.

### 4.2. Run a campaign

```bash
# All L1 scenarios for tasktracker on Claude
belt eval examples/scenarios/experience/tasktracker-claude \
  --tags L1 --modes rules --allow-external-working-dir

# Cross-agent: same campaign, different agent
belt eval examples/scenarios/experience/tasktracker-cursor \
  --tags L1 --modes rules --allow-external-working-dir

# Single specific scenario
belt eval examples/scenarios/experience/tasktracker-claude \
  --scenarios l2_fix_formatter_bug --modes rules --allow-external-working-dir
```

> **Tag filter syntax:** `--tags` takes a single comma-separated value
> with **AND** logic (e.g. `--tags L2,bugfix` runs scenarios tagged with
> both `L2` and `bugfix`). To match EITHER `L1` OR `L2`, run two
> separate commands and merge the outcome dirs. Repeating the flag -
> `--tags L2 --tags bugfix` - is **not** AND: argparse keeps only the
> last occurrence (`bugfix`).

## 5. Custom Agent

The [`custom-agent/`](custom-agent/) directory is a complete,
installable agent plugin demonstrating the `BaseAgentAdapter` contract:

```bash
pip install -e examples/custom-agent
belt eval examples/custom-agent/scenarios --agent echo --modes rules
```

The bundled `multi_turn_with_expectations.json` exercises non-trivial
expectations (`tools_invoked`, `only_used_tools`, `tool_args_contain`,
`tools_invoked_in_order`, `max_total_seconds`, `not_contains`) so the
template proves more than minimum-viable plumbing.

See [PLUGGABILITY.md → Authoring an agent](../docs/glossary/PLUGGABILITY.md#7-authoring-an-agent)
for the full guide.

## 6. Understanding Results

After a run, results land under `outcomes/<timestamp>/`:

```text
outcomes/<run-id>/
├── results.json                    # summary: pass/fail, check counts
├── <group>/<scenario>/
│   ├── turn_0_cli.txt              # raw agent terminal output (one per turn)
│   ├── turn_0_output.json          # parsed TurnOutput per turn (tools, reply, timing, cost)
│   ├── turn_0_stream.ndjson        # live event stream per turn (when streaming is enabled)
│   └── score.json                  # ScenarioScore - all rule + LLM checks for the scenario
```

For the full artifact reference (including `run_meta.json` and the
`.manifest.json` concurrency manifest) see
[OUTCOMES.md](../docs/glossary/OUTCOMES.md).

The terminal also shows a result table whose row count matches the run
and ends with a one-line aggregate (`<passed>/<total> checks (<pct>%) · <elapsed>`).

When a check fails, the output names the check and links to the raw output
for debugging.

## 7. Further Reading

- [SCENARIOS.md](../docs/glossary/SCENARIOS.md) - scenario authoring guide and field-by-field schema reference
- [SCORING.md](../docs/glossary/SCORING.md) - scoring model and multi-judge config
- [CLI.md](../docs/glossary/CLI.md) - subcommand index and common workflows (`belt <cmd> --help` for flags)
