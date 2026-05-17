# CLI

The `belt` console script is the only public entry point. This
document is the surface index - what each subcommand does and how the
common workflows compose. **`belt <subcommand> --help` is the
canonical flag reference**; that output is generated from the same
argparse definitions the runner uses, so it cannot drift.

```bash
belt --version           # print version
belt                     # list subcommands (same as --help)
belt <cmd> --help        # canonical flag reference for one subcommand
```

## 1. Subcommand index

```text
Pipeline (most common)
    eval         Full pipeline: run + score + aggregate (and optionally export)
    run          Execute scenarios only; write outcome artifacts
    score        Score outcome artifacts; write score.json per scenario
    aggregate    Aggregate scores into a report; threshold gating
    export       Emit a completed run to one or more registered destinations

Inspect & operate
    view         Browse results in the terminal after scoring
    watch        Live agent-output viewer (NDJSON tail)
    compare      Side-by-side comparison of two runs
    doctor       Setup verification (agents, auth, LLM providers)
    quickstart   Validate one agent + run the smallest possible scenario
    agent list   Inventory of registered agents (built-ins + plugins)
    agent info   Detailed info for one agent
    gc           Prune old run directories under outcomes/
```

## 2. Setup verification

Run these before anything else - they fail fast on missing agents,
broken auth, or absent LLM credentials.

### 2.1. `doctor` and `agent list`

```bash
belt doctor          # human-readable
belt doctor --json   # machine-readable
belt agent list      # inventory of registered agents
belt agent info <id> # details for a single agent
```

Checks belt version + Python version, every registered agent
(installed, on PATH, authenticated), LLM scoring providers (OpenAI,
Anthropic, Azure, Ollama), and registered exporters. Exit code is
`0` if at least one agent is ready, `1` if none are available.
Doctor also reports the live state of every security toggle
([CONFIGURATION.md → Security toggles](CONFIGURATION.md#4-behaviour-gates----allow----default-deny)).

### 2.2. Judge model preflight (automatic with `belt eval --modes llm`)

When `belt eval` is invoked with `--modes` that includes `llm`, the
preflight runs one cheap probe per configured judge model **before**
spawning the agent phase, in parallel across multiple judges:

- **OpenAI / OpenAI-compatible**: `GET /v1/models/{id}`
- **Azure OpenAI**: `GET /openai/deployments/{name}?api-version=…`
- **Anthropic**: `GET /v1/models/{id}`
- **Ollama**: `POST /api/show`

Outcomes:

- **2xx** → run proceeds.
- **401 / 403 / 404** → run aborts in <2s, before any agent subprocess.
  The message branches on the upstream `error.code`, so `403 +
  model_not_found` (project key without model access) gets a different
  hint than `404 + model_not_found` (typo). Multi-judge configs report
  every 4xx in one composite error.
- **5xx / 429 / timeout / network error** → run proceeds. These are
  transient; the per-scenario [judge infrastructure failure
  path](SCORING.md#251-judge-infrastructure-failures) handles them.

Knobs:

- `BELT_LLM_PREFLIGHT_TIMEOUT` (seconds, default `10`).
- `belt eval --dry-run` skips the network probe (offline-safe; still
  validates config shape).

### 2.3. `quickstart` - first run

```bash
belt quickstart              # auto-detect first available agent
belt quickstart claude-code  # specify agent
```

Validates the chosen agent is installed and authenticated, runs one
built-in scenario with rules-only scoring (no LLM key needed), and
prints next-step guidance.

## 3. The pipeline subcommands

`eval` is the day-to-day command. It chains `run → score → aggregate`
and, when given `--export`, finishes with `export`. Each phase is also
invokable on its own - useful when re-scoring a previous run with a
different judge model, or re-exporting an old run to a new destination.

```text
                         eval (chains the four below)
                          │
   run ──→ outcomes/ ──→ score ──→ score.json ──→ aggregate ──→ results.json
                                                   benchmark-card.json
                                                                    │
                                                              export ──→ external destinations
                                                              (no LLM tokens)
```

Phase isolation is a core design principle - see
[ARCHITECTURE.md → principle 1](ARCHITECTURE.md#principle-1-the-four-phases-know-nothing-about-each-other).

| Subcommand | What it does | Key inputs | Key outputs |
|---|---|---|---|
| `eval` | Run scenarios, score outcomes, aggregate into a report; optional export. | Scenarios root path | Run dir under `outcomes/`, exit code reflects thresholds |
| `run` | Run scenarios only. No scoring, no aggregation. | Scenarios root path | `turn_*` artifacts per scenario |
| `score` | Score the latest run (or `--run-dir`). | Run dir | `score.json` per scenario |
| `aggregate` | Aggregate scored outcomes into `results.json` + benchmark card; enforce thresholds. | Run dir | `results.json`, `benchmark-card.json[.md]` |
| `export` | Read `results.json` + `score.json` and write to registered exporters. Zero LLM cost. | Run dir + `--to`/`--to-config` | One file per exporter |

## 4. Common workflows

### 4.1. Quick validation (no API keys)

```bash
belt eval examples/scenarios/ --dry-run                          # list matched scenarios
belt eval examples/scenarios/ --modes rules --agent claude-code  # rules-only scoring
```

### 4.2. Development loop

```bash
belt eval my-scenarios/ --dry-run --scenarios my-group/my_new_scenario   # validate JSON
belt eval my-scenarios/ --scenarios my-group/my_new_scenario --modes rules
belt score --modes rules                                                  # re-score without re-running
belt score --modes llm --scorer-arg model=openai/gpt-5.4-mini             # add an LLM judge
```

### 4.3. CI gating

```bash
belt eval examples/scenarios/ \
  --agent claude-code \
  --modes rules,llm \
  --scorer-arg model=openai/gpt-5.4-mini \
  --workers 4 \
  --threshold rules/execution:0 \
  --threshold rules/trajectory:10 \
  --progress plain \
  --strict
```

The `--threshold MODE/DIMENSION:PERCENT` flag is repeatable. Exit code
is non-zero when any threshold breaches; threshold semantics are
detailed in [SCORING.md → Thresholds](SCORING.md#4-thresholds).
`--strict` makes any agent availability failure or scenario JSON parse
failure abort the run.

### 4.4. Scenario filtering

`--scenarios` filter paths are relative to the **path argument** (the
"scenarios root"). The runner discovers groups (directories that
contain `_config.json`) under that root; the filter then narrows the
run to one group, one scenario, or several of either.

```bash
belt eval examples/scenarios/ --scenarios showcase/correctness                                # one group
belt eval examples/scenarios/ --scenarios showcase/correctness/correctness_basic              # one scenario
belt eval examples/scenarios/ --scenarios "showcase/correctness,showcase/tool-trajectory"     # multiple groups
belt eval examples/scenarios/ --tags real-runnable                                            # by tag (AND logic)
```

When the path argument is itself a group (a directory containing
`_config.json` at its root), `--scenarios` takes the bare scenario
name; the redundant group prefix is also accepted and stripped:

```bash
belt eval examples/scenarios/experience/tasktracker-claude \
  --scenarios l2_fix_formatter_bug --modes rules
```

If a filter does not resolve, the error message names the scenarios
root and lists the available groups under it.

### 4.5. Re-scoring without re-running

`belt score` (and the `--modes`, `--scorer-arg`,
`--scorer-config` flags on it) is the cheap path when you've already
paid the cost of running the agent and want to try a different judge
model, a different rubric, or just add LLM scoring on top of an
earlier rules-only run.

```bash
belt score                              # latest run, defaults
belt score --run-dir outcomes/<id>      # specific run
belt score --modes llm --scorer-arg model=anthropic/claude-sonnet-4-5
belt score --scorer-config judges.yaml  # multi-judge
belt score --dry-run                    # preview judge prompt for the first scenario
```

### 4.6. Re-exporting without re-aggregating

`belt export` is post-aggregation: it reads `results.json` and
the per-scenario `score.json` files and writes the run to one or more
registered destinations. Re-exporting an old run costs zero LLM tokens.

```bash
belt export --to csv:results.csv                                       # latest run
belt export outcomes/20260322-143000 --to junit:report.xml --to markdown:summary.md
belt export --to-config exporters.yaml                                 # YAML-driven
```

The same chain is available on `belt eval` and `belt aggregate`
via `--export NAME:PATH` (repeatable) and `--export-config FILE`. Available exporter
names: see `belt doctor` under "Exporters". Plugin authoring:
[PLUGGABILITY.md → Authoring an exporter](PLUGGABILITY.md#8-authoring-an-exporter).

A reference YAML lives at
[`examples/exporter-config/exporters.yaml`](../../examples/exporter-config/exporters.yaml).

### 4.7. Cross-agent comparison

```bash
belt eval examples/scenarios/experience/tasktracker-claude --modes rules --allow-external-working-dir
cp outcomes/latest/results.json /tmp/claude.json

belt eval examples/scenarios/experience/tasktracker-cursor --modes rules --allow-external-working-dir
cp outcomes/latest/results.json /tmp/cursor.json

belt compare /tmp/claude.json /tmp/cursor.json \
  --label-a claude-code --label-b cursor --output markdown
```

`compare` also accepts run directories directly; `--output {terminal,markdown,json}`
controls the format.

### 4.8. Reliability (pass@k and pass^k)

```bash
belt eval examples/scenarios/ --trials 5 --modes rules
```

The aggregator reports pass@1, pass@{3,8} = 1-(1-p)^k and
pass^{3,8} = p^k in the output and in `results.json`.
See [SCORING.md → Reliability](SCORING.md#5-reliability-passk-and-passk).

## 5. Inspect & operate

### 5.1. `view` - terminal results browser

```bash
belt view                              # latest run
belt view outcomes/20260322-143000     # specific run
belt view --non-interactive            # print the summary table and exit
```

Drill-down detail per scenario shows rule check results, LLM dimension
scores with judge reasoning, per-turn cost/timing, tool calls, reply
text, and the tail of raw CLI output.

### 5.2. `watch` - live agent-output viewer

```bash
belt watch                                  # latest run
belt watch outcomes/20260322-143000         # specific run
belt watch --scenario claude-code           # filter by substring
belt watch -f                               # follow mode (long-lived)
```

For most use cases `--progress live` on `eval` / `run` is easier - it
shows progress bars and live output in one view. `watch` shines when
you want to tail a run from a separate terminal.

Stream files (`turn_N_stream.ndjson`) live in each outcome directory
and can also be tailed with `tail -f … | jq .`. Disable streaming with
`--no-stream` on `eval` / `run`.

### 5.3. `gc` - prune old run directories

```bash
belt gc                              # keep last 50 runs (default)
belt gc --dry-run                    # preview deletions
belt gc --keep-last 20 --older-than 30
belt gc --keep-last 0 --older-than 7  # purely age-based
```

Live runs (PID alive in the manifest) are always retained, even if
older than the `--older-than` threshold. The `--outcomes-dir` flag
overrides `$BELT_OUTCOMES_DIR` for one invocation.

## 6. Progress modes (`--progress`)

`belt run`, `belt score`, and `belt eval` accept
`--progress {rich,plain,live}`.

| Mode | What you see | When to use it |
|---|---|---|
| `rich` (default) | One progress bar per group + periodic per-scenario line, all written through Rich. | Interactive runs in a real terminal |
| `plain` | Newline-separated text, no ANSI styling, no live updates. | CI logs, log aggregators, anything that mangles cursor control sequences |
| `live` | Two-pane Rich `Live` display: progress bars on top, scrolling per-scenario event panel below (`--progress-live-lines N` overrides default height). | Watching multi-scenario runs as they happen; debugging tool calls |

All three modes share the same per-turn NDJSON stream
(`turn_*_stream.ndjson`) - `live` polls those files mid-run, while
`rich` and `plain` render from the runner's progress callbacks.

## 7. Verbosity (`-v`, `BELT_LOG_LEVEL`)

`belt eval` follows the Inspect AI / Promptfoo pattern: the terminal
is a compact scoreboard, the on-disk run is the canonical artifact.

| Invocation | Terminal level | What you see |
|---|---|---|
| `belt eval ...` (default) | `WARNING` | Results panel, one-line-per-failed-rule, post-run banner pointing at `belt view` and `eval.log`. |
| `belt eval -v ...` | `INFO` | Adds inline judge reasoning, trajectory diagnostics, and per-failure response tails. |
| `belt eval -vv ...` | `DEBUG` | Adds trace-level orchestrator output. |
| `BELT_LOG_LEVEL=DEBUG belt eval ...` | from env | Same effect as `-vv`; `-v` always wins over the env var. |

The transcript log at `<run_dir>/eval.log` always records `DEBUG`
regardless of the terminal level, so the full forensic copy is one
`tail` away whether you ran with `-v` or not.

**Scope:** `-v` is declared on `belt eval` only. The phase commands
(`belt run`, `belt score`, `belt aggregate`) honor the same terminal
verbosity but read it from `BELT_LOG_LEVEL` rather than a CLI flag.
Use `BELT_LOG_LEVEL=info belt aggregate ...` for standalone-phase runs;
`belt aggregate -v` will fail with `unrecognized arguments`.

## 8. Debugging a misbehaving run

When something goes sideways, start with these reads - in order:

1. **`belt doctor`** - agents, auth, LLM provider credentials,
   live state of every security toggle.
2. **`belt doctor --json | jq`** - same, machine-readable; useful
   in CI to fail-fast before paying for the eval.
3. **`belt view <run-dir>`** - drill into a single scenario's
   per-turn output (cost, tool calls, reply, raw CLI tail).
4. **`belt watch -f <run-dir>`** - tail a live run from a separate
   terminal when `--progress live` isn't enough.
5. **`outcomes/<run>/eval.log`** - full loguru output for the run
   (rotates per run dir).
6. **`outcomes/<run>/<group>/<scenario>/turn_*_stream.ndjson`** - raw
   per-turn agent stream, one JSON event per line. `jq` it.
7. **`outcomes/<run>/<group>/<scenario>/score.json`** - typed scorer
   payload; shape is in
   [OUTCOMES.md → Nested scorer payloads](OUTCOMES.md#21-nested-scorer-payloads-inside-scorejson).
8. **`outcomes/<run>/benchmark-card.json`** - full reproducibility
   manifest; `diff` two cards from two runs to find what changed.
9. **`BELT_DEBUG=1`** - full Python tracebacks on unexpected
   errors. Recorded in `run_meta.json`, safe to share.

Two runs disagreed and you don't know why? See
[OUTCOMES.md → Comparing two runs](OUTCOMES.md#6-comparing-two-runs)
and the benchmark card.

## 9. Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success or aggregate report-only |
| `1` | Threshold breach, no matching scenarios, invalid arguments, or fatal error |
