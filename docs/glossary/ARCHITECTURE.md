# Architecture

belt is a phased evaluation pipeline: **run** scenarios through an
agent, **score** the results, **aggregate** into a pass/fail report, and
optionally **export** the aggregated run to one or more downstream
destinations (CSV, JSONL, JUnit XML, Markdown, vendor plugins).

The first half of this document describes the **shape** of the system
(pipeline, layer map, key types, manifest). The second half lists the
**design principles** the shape depends on - eleven principles
enforced by `scripts/check_design.py` and the test suite.

## 1. Pipeline

```text
Scenario JSON ► Runner ► Outcome artifacts ► Scorer ► Scored results ► Aggregator ► Report ► Exporter ► registered destinations
                (agent)  (per-turn files)   (rules    (results.json)   (thresholds,         (post-aggregation;
                                               + LLM)                    comparison)         no LLM tokens)
```

The `belt eval` command chains run + score + aggregate (with
optional `--export`). Each phase can also run independently
(`belt run`, `belt score`, `belt aggregate`,
`belt export`).

## 2. Phases

### 2.1. Run (`runner/`)

The orchestrator (`runner/orchestrator.py`) drives a `BaseAgentAdapter`
through each scenario's turns:

1. `agent.setup(config)` - per-scenario initialization
2. `agent.execute(message, flags)` → raw CLI output - per-turn
3. `agent.fetch_results(raw_output)` → `TurnOutput` - normalize
4. `agent.teardown()` - per-scenario cleanup

Artifacts written per turn: `turn_N_cli.txt`, `turn_N_output.json`,
optionally `turn_N_state.json`. When streaming is enabled (default),
`turn_N_stream.ndjson` captures real-time agent events for live viewing
via `belt watch` or `--progress live`.

When `working_dir` is set in the group config, the orchestrator creates
isolated git worktrees via `WorkspaceManager` (`runner/workspace.py`)
and auto-captures `git_diff` and `files_modified` into `TurnOutput`
after each turn. See [SCENARIOS.md → Workspace isolation](SCENARIOS.md#6-workspace-isolation).

New agents are added by subclassing `BaseAgentAdapter` and registering
via the `belt.agents` entry-point group. See
[PLUGGABILITY.md → Authoring an agent](PLUGGABILITY.md).

### 2.2. Score (`scorer/`)

Two scorer implementations, both consuming `TurnOutput`:

- **RuleBasedScorer** (`scorer/rules/scorer.py`) - deterministic checks
  per turn against `TurnExpectation`. Checks: execution errors, error
  types, tool invocations, tool ordering, tool allowlists, tool
  argument assertions, thinking presence, response content, efficiency,
  cost budgets, workspace state (file existence/content), performance
  timing.
- **LLMScorer** (`scorer/llm/scorer.py`) - sends scenario + outputs +
  workspace evidence (git diff, file contents) to an LLM judge with a
  structured schema. Supports OpenAI, Azure OpenAI, Anthropic, and
  Ollama (local). Response caching avoids re-scoring identical inputs.

Both produce `ScorerResult(passed, data)`. Multi-judge scoring runs
multiple LLM judges with independent models and dimensions (configured
via YAML).

Custom scorers are added by subclassing `BaseScorer` and registering
via the `belt.scorers` entry-point group. See
[PLUGGABILITY.md → Authoring a scorer](PLUGGABILITY.md#7-authoring-a-scorer).

### 2.3. Aggregate (`aggregator/`)

Reads scored results and produces a structured terminal report with:

- **Scoring box** - pass count, mode, partial credit (checks passed / total)
- **Result table** - per-scenario rules + LLM dimension scores
- **Cost/timing** - total and per-scenario cost and wall-clock time
- **Reliability** - pass@{1,3,8} + pass^{3,8}, when `--trials N` was used
- **Failure details** - rule violations, LLM low scores, error context, evidence file paths
- **Threshold enforcement** - per-dimension failure budgets (`--threshold rules/execution:0`)

The `compare` subcommand does side-by-side cross-agent comparison with
cost/timing and per-scenario LLM dimension deltas.

### 2.4. Export (`exporter/`)

Reads the aggregator's `results.json` (typed as `AggregatedResults`)
and the per-scenario `score.json` files and writes the run to one or
more registered exporters. Built-ins ship as siblings of
`exporter/base.py`; vendor exporters register via the
`belt.exporters` entry-point group. The export phase is
**post-aggregation** and does not re-score, re-execute, or call any
LLM; it is a pure read of artifacts already on disk. Failures in one
exporter do not abort others; the exit code is non-zero only when every
requested exporter failed.

Exporters can be invoked standalone (`belt export <run-dir>`) or
chained on `eval` / `aggregate` via `--export NAME:PATH` /
`--export-config FILE`. See
[PLUGGABILITY.md → Authoring an exporter](PLUGGABILITY.md#8-authoring-an-exporter).

## 3. Key types

| Type | Module | Purpose |
|------|--------|---------|
| `BaseAgentAdapter` | `agent/base.py` | Interface for driving a CLI agent |
| `TurnOutput` | `entities.py` | Normalized output from one conversation turn (includes cost, thinking, tool sequence, error type, workspace files, git diff, files modified) |
| `Scenario` | `scenario.py` | Multi-turn conversation definition |
| `TurnExpectation` | `scenario.py` | Deterministic per-turn checks (cost, ordering, args, thinking, error type) |
| `StateExpectation` | `scenario.py` | Post-turn filesystem checks (files_exist, files_contain, files_not_exist, capture_git_diff) |
| `ScenarioResult` | `runner/entities.py` | Aggregated result across turns (includes `total_cost_usd`) |
| `ScoringStrategy` | `agent/scoring.py` | LLM judge dimensions and context |
| `JudgeVerdict` | `scorer/entities.py` | Structured LLM judge output |
| `JudgeConfig` | `scorer/entities.py` | LLM provider/model/temperature settings |
| `Resolution` / `EvidenceScope` | `scorer/entities.py` | Per-judge scoring granularity literals (`scenario` / `turn`, `isolated` / `cumulative`) |
| `TurnJudgeOverride` | `scenario.py` | Per-turn `instruction` / `dimensions` / `evidence_files` / `skip` override attached to `Turn.llm_judges[<judge_name>]` |
| `PerTurnLLMPayload` / `TurnVerdict` | `scorer/payloads.py` | Per-turn LLM judging on-disk payload (one verdict per scenario turn, rolled up by `iter_dimension_feedback` worst-of-turns) |
| `JudgeDef` / `ScorerConfigFile` | `scorer/config_schema.py` | Typed `--scorer-config` YAML (Pydantic `extra="forbid"`, reserved-name validator, per-judge `resolution` / `evidence_scope`) |

## 4. Layer map

```text
                        ┌──────────────────────┐
                        │   belt (CLI)    │  cli.py + commands/
                        └──────────┬───────────┘
                                   │
        ┌──────────────┬───────────┼───────────┬──────────────┐
        ▼              ▼           ▼           ▼              ▼
   ┌─────────┐    ┌─────────┐  ┌────────────┐  ┌────────────┐
   │ runner/ │─→  │ scorer/ │─→│ aggregator/│─→│  exporter/ │
   └────┬────┘    └────┬────┘  └────────────┘  └────────────┘
        │              │        outcome dir      results.json
        │ uses         │ uses   + score.json     + score.json
        ▼              ▼
   ┌─────────┐    ┌────────────┐
   │ agent/  │    │ scorer/llm │  (or scorer/rules)
   └─────────┘    └────────────┘
```

**Coupling rule:** runner / scorer / aggregator / exporter never import
each other. The contract between them is on disk - outcome files
written by the runner, read by the scorer; score files written by the
scorer, read by the aggregator; aggregated results written by the
aggregator, read by the exporter. Enforced mechanically by
`scripts/check_design.py` (see
[principle 1](#principle-1-the-four-phases-know-nothing-about-each-other)).

CLI command modules under `commands/` are thin argparse + dispatch
wrappers around the phase libraries. The `belt` console script is
the only public entry point - everything under `commands/` and the
phase packages is internal.

Tests mirror the source layout under `tests/` (e.g. `tests/commands/`,
`tests/agent/`). For a full file listing, run `tree src/belt -L 2`.

## 5. Run manifest (`manifest.py`)

The manifest solves a concurrency problem: multiple `belt eval`
processes can share the same outcomes directory simultaneously. Each
run registers its PID and per-group resources (agent name,
plugin-specific shared state) in a shared `.manifest.json` file.

### 5.1. How it works

```text
belt eval (PID 1234)          belt eval (PID 5678)
        │                                   │
        ├─ register_run(1234, ...)          ├─ register_run(5678, ...)
        │    ↓ file-locked write            │    ↓ file-locked write
        │   .manifest.json                  │   .manifest.json
        │                                   │
        ├─ run scenarios...                 ├─ run scenarios...
        │                                   │
        └─ unregister_run(1234)             └─ unregister_run(5678)
```

### 5.2. Concurrency safety

All manifest mutations are serialized through a `filelock` adjacent to
the JSON file (`.manifest.json.lock`). Before writing, the manifest
refreshes from disk to avoid stale overwrites. This makes concurrent
runs safe from corruption.

### 5.3. Orphan cleanup

If a process crashes without calling `unregister_run`, its resources
become orphans. On the next startup, `cleanup_orphans()` iterates all
registered runs and checks PID liveness via `os.kill(pid, 0)`:

- **PID alive** → skip (another eval is still running)
- **PID dead** → invoke the caller-provided `delete_fn` on each resource
  entry, then remove the run from the manifest

This handles agent-specific cleanup (e.g., deleting remote assistants
created during setup) without the manifest needing to know agent
internals.

## 6. Configuration layers

Precedence (highest wins):

1. CLI flags (`-S model=openai/gpt-5.4-mini`)
2. Environment variables (`BELT_LLM_MODEL`)
3. Config file (`belt.yaml`)
4. Package defaults

Full env-var inventory and credential routing:
[CONFIGURATION.md](CONFIGURATION.md).

---

## Design Principles

The following design principles are stable across the lifetime of the
project; change them only with broad consensus. Read before modifying
anything. Many are mechanically enforced by `scripts/check_design.py`
(the failure message points at the relevant anchor below).

## Principle 1: The Four Phases Know Nothing About Each Other

Runner → Scorer → Aggregator → Exporter is a one-way pipeline. Each
phase is independently invokable and reads only from the filesystem. No
phase may import another phase's implementation modules.

```text
runner/          scorer/             aggregator/          exporter/
orchestrator  →  rules/scorer.py  →  render_terminal.py → base.py
              →  llm/scorer.py       render_markdown.py    (plugins)
```

The coupling surface is exactly: outcome files written to disk +
`_BELT_*` environment variables for run-label forwarding. Nothing
more.

**Do:**

```python
# scorer reads from filesystem artifacts written by runner
turn_output = TurnOutput.model_validate_json(path.read_text())
```

**Do not:**

```python
# scorer importing runner internals
from belt.runner.orchestrator import run_scenario_turns  # WRONG
```

---

## Principle 2: Entities Are the Contract

All data crossing a phase boundary flows through Pydantic entities
defined in `entities.py`. No phase passes raw dicts, raw strings, or
agent-specific objects to another phase.

Key entities and their ownership:

| Entity | Written by | Read by |
|--------|-----------|---------|
| `TurnOutput` | Agent (`fetch_results`) | Rules scorer, LLM scorer |
| `ScenarioResult` | Runner orchestrator | Eval cmd (progress, handoff) |
| `ScenarioScore` | Scorer CLI | Aggregator |
| `GroupConfig` | Parser | Runner, Scorer |
| `Scenario` | Parser | Runner, Scorer |

Adding a field to an entity is fine and encouraged. Adding a field that
only one agent populates is also fine - use `Optional` with a safe
default. Do not add methods or business logic to entities.

---

## Principle 3: Extend via Abstractions, Never Modify the Framework Core

The framework has four extension points. Use them.

| Extension point | Abstract class | How to extend |
|----------------|---------------|---------------|
| New agent support | `BaseAgentAdapter` | Subclass + `belt.agents` entry point |
| New scoring mode | `BaseScorer` | Subclass + `belt.scorers` entry point |
| New LLM provider | `BaseJudgeBackend` | Subclass + add to `resolve_backend()` |
| New export destination | `BaseExporter` | Subclass + `belt.exporters` entry point |

`BaseAgentAdapter`, `BaseScorer`, `BaseJudgeBackend`, and
`BaseExporter` are **closed to modification**. If a new agent needs a capability those interfaces
don't have, add an optional method with a default - do not change the
interface contract of existing methods.

The only valid reason to touch `base.py` is to add an *optional
override* (with a default implementation) after verifying the pattern
is shared across at least two existing agents.

---

## Principle 4: Agents Are Thin Plumbing, Not Policy

An agent translates `execute(message, flags)` into a subprocess call
and parses the output back into `TurnOutput`. That is its entire job.

| Concern | Belongs to | Never belongs to |
|---------|-----------|-----------------|
| Model selection | Scenario `flags` | Agent |
| Working directory | Scenario `flags` | Agent |
| Execution mode | Scenario `flags` | Agent |
| Timeout | Scenario `flags` / framework | Agent |
| Headless safety flags | Agent (one exception - see below) | Framework |
| Output parsing | Agent | Framework |
| Session continuity | Agent (`--resume <id>`) | Framework |
| Evaluation dimensions | Agent (`scoring_strategy()`) | Framework |

The **only** exception to agent flag injection: flags required to
prevent interactive hangs that would block evaluation entirely (e.g.,
Cursor's `--force --approve-mcps`). These must be documented in the
agent's `display_info()` and are not a precedent for other injections.

---

## Principle 5: Graceful Degradation via Optional Fields, Not Branching

Agents have different capabilities. The framework accommodates this
through `Optional` fields on `TurnOutput` with safe defaults - **not**
through `isinstance` checks, capability flags, or agent-specific
branching in the framework.

**Do:**

```python
# TurnOutput field - only agents that support it populate it
cost_usd: float | None = None
thinking_text: str | None = None
tool_sequence: list[str] = []
```

```python
# rule scorer handles absence safely
if scenario.turns[i].expect.max_cost_usd is not None and output.cost_usd is not None:
    check cost_usd <= max_cost_usd
# if either is None, the check is silently skipped - no failure, no agent detection
```

**Do not:**

```python
# branching on agent type in the framework
if isinstance(agent, ClaudeCodeAgentAdapter):
    check_cost(output.cost_usd)  # WRONG - framework knows about concrete agent
```

```python
# capability flags
if output.supports_cost_tracking:  # WRONG - capability flags are isinstance in disguise
    check_cost(output.cost_usd)
```

**Choosing the default:** The default is the *most conservative
semantic value* - `None` for "absent," `0` for "zero events," `{}` for
"empty mapping," `False` for "feature not exercised." Never use a
sentinel like `"unknown"` or `-1`; rule scorers must short-circuit on
the absence, not pattern-match on a magic value.

---

## Principle 6: Public API as Plugin Boundary

Plugins (out-of-tree agents, scorers, exporters) import only from the
top-level `belt` package. Internal module paths
(`belt.exporter.base`, `belt.scorer.payloads`, …) are private
and may change without a major version bump.

```python
# Do
from belt import BaseExporter, RulesPayload, iter_dimension_feedback

# Don't
from belt.exporter.base import BaseExporter           # WRONG - internal path
from belt.scorer.payloads import RulesPayload          # WRONG - internal path
```

The contract is `belt._public_api.PUBLIC_API` - the dict mapping
each exported name to its source module, re-exported lazily from
`belt/__init__.py`. Adding a symbol to `PUBLIC_API` is what makes
it part of the published surface; until then, plugins must not depend
on it.

A second sub-rule: plugins do not literal-index
`score.scores["rules"]` or `score.scores["llm"]`. The two built-in
keys are an implementation detail of the bundled scorers; third-party
scorers register different keys. Plugins iterate via the public
`iter_dimension_feedback()` helper or guard typed payload access with
`isinstance(..., RulesPayload | LLMPayload)`.

---

## Principle 7: Errors Are User-Facing Communication

Exceptions that reach the terminal are part of the product UX. A raw
Python traceback is a bug - it exposes implementation details and
provides no actionable guidance.

**Rules:**

1. A top-level exception boundary in `cli.py` catches all unhandled
   exceptions and renders a one-line message with context (file,
   scenario, agent).
2. Known errors use the typed hierarchy in `errors.py` (`BeltError`
   and subtypes). Unexpected errors show `{ClassName}: {message}` + a
   debug hint.
3. `BELT_DEBUG=1` exposes full tracebacks for bug reports.
4. Never `sys.exit()` from a `main()` that another module calls - return
   an int exit code. `commands.eval` calls `commands.run.main()`,
   `commands.score.main()`, and `commands.aggregate.main()` in
   sequence; `sys.exit` short-circuits the chain.

**What this looks like in the terminal:**

```text
$ belt eval examples/scenarios/broken
ConfigError: invalid scorer config 'belt.yaml': temperature must be numeric, got 'hot'
  hint: set BELT_DEBUG=1 to see the full traceback
```

One line, names the file, names the field, names the value, points at
the next action. No stack frames in the user's face.

---

## Principle 8: Security Model - Default-Deny + Escape Untrusted Text

belt runs adversarial inputs (agent stdout, LLM judge reasoning,
scenario text from external repos) through markup-aware sinks and
network-bound subsystems. The security model has two halves:

**Default-deny for every safety toggle.** Every gate that lowers a
boundary is off by default and requires an explicit `--allow-*` CLI
flag or `BELT_ALLOW_*` environment variable to enable. No silent
acceptances, no "warn and continue."

```python
# Do
if not allow_insecure_base_url and parsed.scheme == "http" and not _is_loopback(parsed.host):
    raise ConfigError("LLM base_url uses http:// to a non-loopback host. ...")

# Don't
if parsed.scheme == "http" and not _is_loopback(parsed.host):
    logger.warning("insecure base_url, continuing anyway")  # WRONG - silent permit
```

The full toggle inventory lives in
[CONFIGURATION.md → Security toggles](CONFIGURATION.md#4-behaviour-gates----allow----default-deny),
and the full threat model this principle defends - trust boundaries,
control catalog, and accepted residual risk - is in
[SECURITY-MODEL.md](SECURITY-MODEL.md).

**Escape every untrusted string before it reaches a markup sink.**
Agent stdout, judge reasoning, scenario filenames, and any other
attacker-controlled text must pass through `_safe.py` (`rich_safe` for
Rich consoles, `md_safe` for Markdown step-summaries). A judge asked
to grade malicious agent text can return `[red]System failure[/red]`
or `</details><script>...` - without escaping, those tokens are
interpreted at the rendering layer and corrupt the output.

```python
from belt._safe import rich_safe, md_safe

# Do
console.print(f"[bold]Reply:[/bold] {rich_safe(turn.reply_text)}")
summary.append(f"- **{md_safe(scenario.name)}** - {md_safe(judge.reasoning)}")

# Don't
console.print(f"[bold]Reply:[/bold] {turn.reply_text}")        # WRONG - Rich injection
summary.append(f"- **{scenario.name}** - {judge.reasoning}")   # WRONG - Markdown injection
```

---

## Principle 9: One Source of Truth for Public Names

Public names - environment variables, schema versions, exit codes, file
templates - live in exactly one module. Other code imports the constant;
it does not re-spell the literal.

| Kind of name | Source of truth |
|---|---|
| Public env vars (`BELT_*`) | `envvars.py` |
| Internal forwarding env vars (`_BELT_*`) | `_internal_envvars.py` |
| Schema version, file templates, error patterns | `constants.py` |
| Typed error names | `errors.py` |
| Default judge parameters (temperature, seed, max_tokens) | `JudgeConfig` in `scorer/entities.py`. There is no built-in `llm.model` default - `config.py` raises `ConfigError` when none of CLI/env/yaml supplies one, and `constants.EXAMPLE_LLM_MODEL` is the example value rendered into help text. |

---

## Principle 10: Documentation Travels with Code

Every feature, behavioural change, or new convention updates the
relevant documentation in the same PR. Stale docs actively mislead -
new contributors and AI agents read them first and write code against
a phantom contract.

| Change | Update |
|--------|--------|
| New agent | [AGENT-FEATURES.md](AGENT-FEATURES.md) feature matrix |
| New scenario group | Group README if non-trivial |
| New scorer / dimension | [SCORING.md](SCORING.md), [PLUGGABILITY.md → Scorers](PLUGGABILITY.md#7-authoring-a-scorer), [CONFIGURATION.md](CONFIGURATION.md) |
| New CLI flag or command | `AGENTS.md` CLI section, [CLI.md](CLI.md) |
| New entity field | [SCENARIOS.md](SCENARIOS.md) reference appendix |
| New public symbol | `belt._public_api.PUBLIC_API` + [PLUGGABILITY.md](PLUGGABILITY.md) |
| Changed behaviour | Whatever doc describes the old behaviour |

This principle is last in numbering only. It is enforced by
doc-parity tests under `tests/test_*_doc_parity.py` and by the
mechanical drift guards described in each consolidated doc.

---

## Violation checklist

Before merging any PR, verify none of these are present:

- [ ] A scorer, aggregator, or exporter file imports from `runner/` (P1)
- [ ] A runner file imports from `scorer/`, `aggregator/`, or `exporter/` (P1)
- [ ] New business logic added to an entity (entities are data, not behaviour) (P2)
- [ ] `BaseAgentAdapter.execute()` or `fetch_results()` signature has changed (P3)
- [ ] A feature that only applies to one agent was added to `base.py` (P3)
- [ ] An agent stores policy choices (model name, mode, workspace) as instance state (P4)
- [ ] Framework code contains `isinstance(agent, SomeConcreteAgentAdapter)` or a `supports_*` capability flag (P5)
- [ ] A new field on `TurnOutput` or `ScenarioScore` is non-optional without a safe default (P5)
- [ ] A new field uses a sentinel value (`-1`, `"unknown"`) instead of `None`/`0`/`{}` (P5)
- [ ] A plugin imports from an internal module path instead of the top-level `belt` package (P6)
- [ ] A `main()` function uses `sys.exit()` instead of `return` (P7)
- [ ] A stdlib exception (`KeyError`, `TypeError`) escapes to the user without context (P7)
- [ ] An untrusted string (agent stdout, judge reasoning, filename) reaches a Rich/Markdown sink
  without `rich_safe` / `md_safe` (P8)
- [ ] A new safety boundary was wired with default-allow instead of default-deny + `--allow-*` opt-in (P8)
- [ ] A new env var was introduced as a literal string outside `envvars.py` / `_internal_envvars.py` (P9)
- [ ] A behavioural change or new feature has no corresponding documentation update (P10)
