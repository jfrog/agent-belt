# Outcomes

What an `belt eval` run writes to disk, how it's versioned, and how
to compare runs.

The on-disk outcomes are belt's external contract - every other
tool (CI bots, dashboards, vendor exporters, the `view` and `compare`
subcommands) reads the same files. This document is the reference for
those readers.

## 1. Directory layout

```text
outcomes/
└── 20260322-143000/                # run directory (timestamp)
    ├── run_meta.json               # run-time provenance (input side)
    ├── run_fixtures.json           # per-group fixture SHAs (post setup_groups)
    ├── benchmark-card.json         # reproducibility manifest (post-aggregate)
    ├── benchmark-card.md           # human-readable card (GITHUB_STEP_SUMMARY)
    ├── .manifest.json              # concurrent-run manifest
    ├── eval.log                    # loguru output
    ├── <group>/<scenario>/         # one dir per scenario
    │   ├── _runtime_info.json      # agent CLI path/version/auth (post setup)
    │   ├── turn_0_cli.txt          # raw CLI output
    │   ├── turn_0_output.json      # TurnOutput (Pydantic)
    │   ├── turn_0_state.json       # thread state (optional)
    │   ├── turn_0_stream.ndjson    # live event stream (optional)
    │   └── score.json              # ScenarioScore (Pydantic)
    └── results.json                # aggregated results
```

## 2. Versioned artifacts

Every file in the table below carries a `schema_version`; the count
itself is intentionally not maintained inline so adding a new versioned
artifact does not require a doc edit.

| File | Model | Writer | Reader(s) |
|------|-------|--------|-----------|
| `turn_N_output.json` | `TurnOutput` | `runner/orchestrator.py` | scorer, aggregator |
| `score.json` | `ScenarioScore` | `commands/score.py` | aggregator, exporter |
| `run_meta.json` | raw dict | `commands/run.py` | scorer (scenarios root), benchmark card |
| `results.json` | `AggregatedResults` | `commands/aggregate.py` | exporter, external consumers, benchmark card |
| `benchmark-card.json` | `BenchmarkCard` | `commands/aggregate.py` | exporter, external consumers, PR comments |

The two unversioned sidecars (`_runtime_info.json`, `run_fixtures.json`)
are internal inputs to the benchmark card; their fields are absorbed
into the versioned card and never surface independently to external
readers.

### 2.1. Nested scorer payloads inside `score.json`

`score.json` carries an outer `schema_version` for the `ScenarioScore`
envelope (currently `"1"`) and an **inner** `schema_version` per scorer
under `scores`:

```jsonc
{
  "schema_version": "1",          // ScenarioScore envelope
  "scores": {
    "rules": { "schema_version": "rules.v1", "checks": [/* ... */] },
    "llm":   { "schema_version": "llm.v1",   "dimensions": {/* ... */} }
  }
}
```

The inner `schema_version` is the discriminator for the typed payload
contract defined in `belt.scorer.payloads`. Built-in shapes are
`rules.v1` and `llm.v1`; the LLM payload carries per-dimension
verdicts on either the ternary (`high`/`medium`/`low`) or binary
(`pass`/`fail`) scale, optionally extended with `inconclusive` - see
[SCORING.md → §2.5](SCORING.md#25-verdict-scales-binary--ternary--inconclusive).
Third-party scorers register their own shapes (see
[PLUGGABILITY.md → Authoring a scorer](PLUGGABILITY.md#7-authoring-a-scorer)).
Missing or unregistered `schema_version` raises a `ValueError` at parse
time - the framework never guesses at unknown shapes.

When the LLM judge backend fails for infrastructure reasons (rate-limit,
timeout, network, parse failure), `score.json` writes a non-verdict
`llm.v1` payload with `judge_errored: true` plus `judge_error_type`
(`rate_limited` / `timeout` / `auth_failed` / `other`) and an empty
`dimensions` map. Downstream consumers (the aggregator's `judge_errors`
block in `results.json`, vendor eval runners) key off `judge_errored`
to partition the run, never on string parsing of grader evidence.
`results.json` also carries a top-level `judge_errors` field mirroring
the shape of `agent_errors`, and the `stats.task_quality` partition
exposes both `env_failed_agent` and `env_failed_judge` so the operator
can attribute infra blame to the right side of the wall.

## 3. Agent runtime errors (`error_type`)

`TurnOutput.error_type` carries a stable, framework-defined token for
*runtime* failures - the agent didn't really run, as opposed to the
agent ran and answered wrong. Adapters classify into this taxonomy in
their `fetch_results`; the aggregator rolls per-scenario tokens into
`results.json` and `benchmark-card.json` under an `agent_errors` block.

The token set is the canonical source of truth for downstream
consumers. Vendor-specific labels (Anthropic's `"overloaded"`,
Gemini's `"QuotaError"`, OpenCode's `"AuthError"`) are normalised into
these tokens by `belt.agent.error_types.normalize_error_type` -
no vendor strings leak past the agent boundary.

| Token | Bucket | Meaning |
|-------|--------|---------|
| `authentication_failed` | environmental | Credentials missing, expired, or rejected (401, "Not logged in", "/login"). |
| `rate_limited` | environmental | Provider throttled the request (429, quota exceeded). |
| `timeout` | environmental | Per-turn deadline / gateway timeout. |
| `model_unavailable` | environmental | Credentials worked but the requested model is unknown or not entitled to the project (`model_not_found`, "does not have access to model X"). Fix is a model swap or entitlement request, not re-authentication. |
| `refused` | task | Model declined the request ("I can't help with that"). |
| `unknown` | task | Adapter detected `has_error=true` but no pattern matched. |

Constants live in `belt.entities` (see `AUTHENTICATION_FAILED`,
`RATE_LIMITED`, `TIMEOUT`, `MODEL_UNAVAILABLE`, `REFUSED`, `UNKNOWN`,
and the `ERROR_TYPES` set). The bucket partition lives in the same module
(`ENVIRONMENTAL_ERROR_TYPES`, `TASK_ERROR_TYPES`); the two sets are
disjoint and together cover `ERROR_TYPES` exactly.

Adding a token is additive (no schema bump) but requires placing it in
exactly one bucket; an `environmental` error means "the agent could
not run because of something outside its decision-making" (auth,
throttle, deadline), while a `task` error means "the agent ran but
behaved badly" (refused, classifier-unmatched). Renaming a token is a
breaking schema change. Documentation parity is enforced by
`tests/test_error_types_doc_parity.py`; the partition invariant by
`tests/aggregator/test_task_quality_split.py`.

### 3.1. `agent_errors` block in `results.json` and `benchmark-card.json`

`AggregatedResults.agent_errors` (and the equivalent
`BenchmarkCard.agent_errors`) is `null` when no scenario reported
`has_error=true` on any turn. Otherwise, the block carries:

```jsonc
{
  "scenarios_with_errors": 1,        // count of scenarios with >= 1 erroring turn
  "scenarios_total": 1,              // total scenarios in the run
  "vacuous_passes": 0,               // scenarios that passed AND had agent errors
  "by_error_type": {                 // canonical-token tally
    "authentication_failed": 1
  },
  "remediation": "Re-authenticate the Claude Code CLI: run `claude login`.",
  "task_quality": {                  // present when env_failed > 0 (see below)
    "attempted": 1,
    "env_failed": 1,
    "env_failed_agent": 1,           // agent-axis environmental failures
    "env_failed_judge": 0,           // judge-axis environmental failures
    "completed": 0,
    "passed": 0,
    "task_failed": 0,
    "pct": null
  },
  "per_scenario": [                  // results.json only; the card omits this
    {
      "scenario": "showcase/correctness/correctness_basic",
      "passed": false,
      "vacuous_pass": false,
      "error_types": ["authentication_failed"],
      "first_reply_text": "Not logged in · Please run /login"
    }
  ]
}
```

A non-zero `vacuous_passes` means at least one scenario's rules passed
while the agent errored on a turn - the rules pass is untrustworthy.

#### Task quality vs environmental health (`task_quality` sub-block)

`task_quality` partitions the run into "agent ran cleanly" vs "agent
was blocked by the environment". It is present whenever at least one
scenario hit an environmental error (auth, rate-limit, timeout) - the
single-axis "M/N scenarios failed" headline is misleading in that
case because it conflates "the agent did the wrong thing" with "the
provider had a bad minute". When every error in the run is in the
task bucket (`refused`, `unknown`), the field is absent and the
existing single-axis headline already tells the right story.

| Field | Meaning |
|-------|---------|
| `attempted` | Total scenarios in the run. |
| `env_failed` | Scenarios with at least one environmental error on either axis (`env_failed_agent + env_failed_judge`). |
| `env_failed_agent` | Scenarios where the agent CLI hit an environmental error (auth, rate-limit, timeout, model-unavailable). |
| `env_failed_judge` | Scenarios where the LLM judge backend hit its own infra failure. Agent-axis wins on overlap. |
| `completed` | `attempted - env_failed` - scenarios where the agent ran cleanly enough to produce a verdict. |
| `passed` | Of `completed`, how many passed rules. Vacuous passes (rules pass + env error) are NOT counted. |
| `task_failed` | `completed - passed`. |
| `pct` | `passed / completed` rounded to one decimal, or `null` when `completed == 0`. |

`passed / completed` is the number a CI dashboard can defensibly
publish: it excludes scenarios the agent never got to attempt because
of transient external failures. The bottom-line headline rendered by
`belt aggregate` switches to a three-part split when this block is
present - `<P>/<C> task quality (<%>) - <E> environmental failures -
<F> agent task failures` - replacing the single-axis "M/N scenarios
failed" line. The benchmark card mirrors the same fields under
`agent_errors.task_quality`, including both axis counters.

### 3.2. `judge_errors` block in `results.json` and `benchmark-card.json`

Structurally parallel to `agent_errors`: `null` when every scenario's
LLM judge produced a verdict, otherwise a typed block summarising
judge-axis environmental failures (rate-limit, timeout, auth, network
from the judge backend - not the agent). Surfaced as a sibling rather
than folded into `agent_errors` so downstream tooling can attribute
environmental failures to the right backend (provider key vs judge
key, agent CLI auth vs judge backend auth) without re-deriving the
split from raw artifacts.

```jsonc
{
  "scenarios_with_errors": 1,
  "scenarios_total": 4,
  "by_error_type": {
    "rate_limited": 1
  }
}
```

`BenchmarkCard.judge_errors` carries the same rolled-up shape; the
per-scenario detail (`per_scenario`, with vendor error messages)
appears in `results.json` only.

## 4. Scenario load failures (`scenarios_skipped`)

`AggregatedResults.scenarios_skipped` is the count of scenario JSON
files that failed to parse during the loader phase (typo in a field,
malformed JSON, forbidden extra key under `extra="forbid"`). Always
present; defaults to `0`.

```jsonc
{
  "total": 12,                       // scenarios that ran and produced score.json
  "passed": 11,
  "failed": 1,
  "scenarios_skipped": 2,            // scenarios dropped by the parser
  "overall_pass": false
  // ... rest of AggregatedResults
}
```

A non-zero `scenarios_skipped` means the run executed on a smaller fleet
than the author intended. Use `--strict` on `belt run` /
`belt eval` to make any non-zero value abort the run with the list
of offending files. The count is threaded from the loader phase through
`run_meta.json`; old runs without that field load as `0`.

## 5. Version contract

- `schema_version` is a string (currently `"1"`).
- The canonical value lives in `belt.constants.SCHEMA_VERSION`.
- **Missing field** (`null` / absent): optional in the v1 contract.
  Readers log a warning and assume v1.
- **Matching version**: proceed silently.
- **Mismatched version**: readers log a warning but still attempt to
  parse. This keeps old tooling usable against newer artifacts for
  minor changes.

## 6. Comparing two runs

Diff two runs on disk:

```bash
diff outcomes/run-A/results.json outcomes/run-B/results.json
belt compare outcomes/run-A outcomes/run-B
```

Cross-agent comparison surfaces per-scenario LLM dimension deltas plus
cost/timing. For a deeper "what made these runs differ?" investigation,
read the benchmark card.

## 7. Benchmark card (reproducibility manifest)

Every run also writes a **benchmark card** alongside the artifacts
above. The card answers a single question:

> Two runs disagreed. What was different?

The answer lives in the card: belt version, host runtime, the
user's exact command, scenario file SHAs, fixture branch + SHA, agent
CLI path and version, judge backends, scoring config, and pass/fail
summary.

### 7.1. Files

Both files are written by `belt aggregate` (the final phase of
`belt eval`):

- `benchmark-card.json` - machine-readable, validated by the
  [`BenchmarkCard`](../../src/belt/benchmark_card/entities.py)
  Pydantic schema. Stable contract; consumed by external tooling, CI
  artifacts, and PR bots.
- `benchmark-card.md` - human-readable rendering of the same data,
  suitable for `$GITHUB_STEP_SUMMARY` and PR comments. Best-effort
  formatting; the JSON is the source of truth.

When `$GITHUB_STEP_SUMMARY` is set in the environment, the Markdown
card is appended to the existing aggregator summary so a single click
in the PR's CI tab reveals full provenance alongside the pass/fail
story.

### 7.2. Sections

The card is organised into eight sections; each maps to one Pydantic
model in
[`benchmark_card/entities.py`](../../src/belt/benchmark_card/entities.py).

| Section | Schema | Sourced from |
|----|----|----|
| `belt` | `BeltProvenance` | `importlib.metadata.version` + git SHA (editable installs only) |
| `host` | `HostProvenance` | `platform` + `importlib.metadata` for declared dependencies |
| `invocation` | `Invocation` | argv + parsed argparse namespace + sanitised env (env-var allow-list) |
| `scenarios` | `ScenarioSelection` | run-time tags/groups filter + SHA-256 of every scenario JSON |
| `fixtures` | `FixtureProvenance` | per-group git SHA + dirty count, captured after group setup |
| `agents` | `AgentProvenance` | per-group `runtime_info()`: CLI binary path, CLI version, auth signals, `-X` args |
| `scoring` | `ScoringConfig` | `--modes`, thresholds, and the LLM judge backends actually consulted |
| `runtime` | `RuntimeConfig` | `--workers`, `--trials`, streaming on/off, scenario delay |
| `cost_timing` + `summary` | `CostTimingSummary` + `ScoreSummary` | aggregator's `results.json` |

### 7.3. Pipeline

The card is assembled from on-disk inputs written earlier in the run:

```text
commands/run.py            ──→ run_meta.json                 # static run-time provenance
runner/phases/setup_groups ──→ run_fixtures.json             # per-group git SHA + dirty count
runner/orchestrator.py     ──→ <g>/<s>/_runtime_info.json    # agent CLI identity
commands/score.py          ──→ <g>/<s>/score.json            # judge backends used
commands/aggregate.py      ──→ benchmark-card.json           # final artifact
                              benchmark-card.md
```

[`build_card()`](../../src/belt/benchmark_card/build.py) is the
only consumer that reads all sources together. Any individual input
can be missing; the card falls back to a deterministic default rather
than aborting the aggregate.

### 7.4. Adding agent runtime info

When you write a new agent (see
[PLUGGABILITY.md → Authoring an agent](PLUGGABILITY.md#6-authoring-an-agent)),
override `runtime_info()` to populate the `agents` section. The
detailed contract - flat shape, `_capture_cli_version()` helper, how
the framework projects flat dicts into the persisted two-level
shape - lives with the agent-authoring guide.

The on-disk shape is:

```json
{
  "group": "showcase",
  "agent": {
    "name": "myagent",
    "adapter_class": "MyAgentAdapter",
    "args": {"model": "gpt-4"},
    "auth_signals": ["env:MYAGENT_API_KEY"]
  },
  "cli": {
    "binary_path": "/usr/local/bin/myagent",
    "version": "1.2.3"
  }
}
```

The sidecar is intentionally **unversioned**: it is an internal input
absorbed into the versioned card and never surfaces independently to
external readers.

### 7.5. Secret hygiene

Every user-controlled string in the card is filtered through
`belt._redact` before persistence. That module is the single
source of truth for secret redaction across the codebase: the
secret-name regex is defined exactly once, and every `key=value`
parser is centralised so a future shape (e.g. a new flag) cannot
diverge from the others.

- Environment variables (`safe_environ`): only the documented
  allow-list passes through; secret-shaped names degrade to `"<set>"`;
  `*_BASE_URL` is redacted to `scheme://host[:port]`.
- `argv` (`scrub_argv`): `-X KEY=VALUE` and `--agent-arg KEY=VALUE` are
  scanned for secret-shaped keys; matching values are replaced with
  `<redacted>`. Every shape argparse accepts is handled (`-Xk=v`,
  `-X k=v`, `--agent-arg k=v`, `--agent-arg=k=v`); a regression of
  any single shape is gated by `tests/test_redact.py`.
- `agent.args` (`scrub_dict` via `_redact.safe_agent_args`):
  secret-shaped option names OR options whose declared `env_var`
  matches the secret regex are replaced with `"<set>"`.

The card never holds a raw API key, password, or token. If a future
agent adds a new credential surface, the redactor's input contract is
the place to extend - never the card schema.

### 7.6. Reading a card programmatically

```python
from pathlib import Path

from belt.benchmark_card import BenchmarkCard, build_card, load_results_for_card

run_dir = Path("outcomes/20260101-120000-abcdef")
results = load_results_for_card(run_dir)
card = build_card(run_dir, results)

print(card.belt.git_sha, "vs", other_card.belt.git_sha)
print(card.scenarios.scenario_files == other_card.scenarios.scenario_files)
```

The schema is validated on construction (Pydantic), so a malformed card
fails loudly rather than silently dropping fields.
