# CI

belt is built to gate merges on agent quality. This doc covers the
patterns that make that work in practice - headless behaviour, exit
codes that block PRs, cost-aware modes, and standard test reports that
plug into any CI vendor.

## 1. What CI gets you

- **Headless by default.** `--progress plain` produces line-buffered,
  non-interactive output. `belt doctor --json` is machine-readable
  for fail-fast preflight.
- **Exit codes are merge gates.** Threshold breaches return non-zero;
  `dorny/test-reporter` (or any JUnit consumer) marks PR checks red.
- **Reliability metrics that test runners don't have.** `--trials N`
  produces pass@k and pass^k - the agent-quality signal that boolean
  test runners can't express.
- **Cost-aware execution.** Run rules-only (deterministic, free) on
  every PR; run rules + LLM judges (expensive, full signal) on nightly.
- **Portable reports.** JUnit XML is consumed by every major CI
  vendor's native test publisher. Markdown chains into your job's
  step summary. CSV / JSONL chain into a data warehouse.
- **Agent-comparison matrices.** Run the same scenarios against N
  agents in parallel; compare results across runs to track regressions
  per agent, per release.

## 2. Quick start

One canonical GitHub Actions job covering the patterns most projects
need: pip + agent caching, run cancellation on PR superseding,
preflight `doctor` check, threshold-gated eval, JUnit + step summary,
and per-turn outcomes uploaded as an artifact.

```yaml
name: Eval
on:
  pull_request:
    branches: [main]

concurrency:
  group: eval-${{ github.ref }}
  cancel-in-progress: true

jobs:
  eval:
    runs-on: ubuntu-latest
    permissions:
      contents: read

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
          cache: pip

      - name: Install belt and your agent CLI
        run: |
          pip install agent-belt
          curl -fsSL https://cursor.com/install | bash
          echo "$HOME/.local/bin" >> "$GITHUB_PATH"

      - name: Verify setup
        env:
          # Auth env vars for the agent + LLM judge.
          CURSOR_API_KEY: ${{ secrets.CURSOR_API_KEY }}
          BELT_OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: belt doctor --json | jq -e '.agents_ready > 0'

      - name: Run eval
        env:
          CURSOR_API_KEY: ${{ secrets.CURSOR_API_KEY }}
          BELT_OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          belt eval examples/scenarios/ \
            --agent cursor \
            --tags smoke \
            --modes rules,llm \
            --threshold rules/execution:0 \
            --progress plain \
            --export junit:report.xml \
            --export markdown:summary.md

      - name: Upload outcomes (per-turn detail for post-mortem)
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: belt-outcomes-${{ github.sha }}
          path: outcomes/
          retention-days: 14
```

Wire `report.xml` into your CI's JUnit publisher and append
`summary.md` to your job's step summary using your vendor's standard
mechanism. Run `belt doctor` to list every registered exporter.

### 2.1. Skip the eval when nothing relevant changed

Eval runs are slow and cost money. Use `dorny/paths-filter` (or
GitHub's built-in `paths:` filter) to skip the job when no relevant
files changed.

```yaml
jobs:
  changes:
    runs-on: ubuntu-latest
    outputs:
      should_run: ${{ steps.filter.outputs.should_run }}
    steps:
      - uses: actions/checkout@v4
      - uses: dorny/paths-filter@v3
        id: filter
        with:
          filters: |
            should_run:
              - 'src/**'
              - 'examples/scenarios/**'
              - '.github/workflows/eval.yml'

  eval:
    needs: changes
    if: needs.changes.outputs.should_run == 'true'
    # ...
```

### 2.2. Parameterize manual re-runs

`workflow_dispatch` with `inputs:` lets contributors trigger an eval
from the GitHub UI with custom tags / modes / thresholds - no YAML
edit, no commit. Invaluable when triaging a regression or
experimenting with a new threshold.

```yaml
on:
  workflow_dispatch:
    inputs:
      tags:
        description: "Comma-separated tags (AND logic)."
        default: "smoke"
      modes:
        description: "Scorers to run."
        default: "rules,llm"
        type: choice
        options: ["rules,llm", "rules", "llm"]
      thresholds:
        description: "Comma-separated dimension thresholds. Empty = report only."
        default: "rules/execution:0"

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      # ... checkout / install / doctor ...
      - run: |
          THRESHOLD_FLAGS=""
          for t in $(echo "${{ github.event.inputs.thresholds }}" | tr ',' ' '); do
            THRESHOLD_FLAGS="$THRESHOLD_FLAGS --threshold $t"
          done
          belt eval examples/scenarios/ --agent cursor \
            --tags "${{ github.event.inputs.tags }}" \
            --modes "${{ github.event.inputs.modes }}" \
            --progress plain \
            $THRESHOLD_FLAGS
```

## 3. Threshold gating

`--threshold MODE/DIMENSION:PERCENT` is repeatable. Any breach makes
the run exit non-zero, which fails the GitHub check and blocks the
merge.

```bash
belt eval examples/scenarios/ \
  --threshold rules/execution:0    \  # zero execution failures
  --threshold rules/trajectory:10  \  # at most 10% trajectory regressions
  --llm-fail-on low,medium             # any low/medium LLM verdict fails
```

Threshold semantics: [SCORING.md → Thresholds](SCORING.md#4-thresholds).

| Exit code | Meaning |
|---|---|
| `0` | All thresholds passed (or report-only without thresholds) |
| `1` | Threshold breach, no matching scenarios, invalid arguments, or fatal error |
| `130` | Cancelled by SIGINT (`Ctrl-C`, run cancellation) |

## 4. Cost-aware patterns

LLM judges cost real money per scenario. The patterns below let you
get full signal where it matters without paying for it on every PR.

### 4.1. Rules-only on PR, full on nightly, tags everywhere

PR jobs run rules (deterministic, no API key needed) over a small
`smoke` subset. A nightly schedule runs rules + LLM judges over the
full `production` suite. Use `--tags` (AND-logic intersection) to keep
all scenarios in one directory and select the subset per workflow.

```yaml
# PR check: cheap, fast, deterministic, narrow scenario subset
- run: belt eval examples/scenarios/ --tags smoke
         --modes rules --threshold rules/execution:0

# Nightly: full LLM signal over the full suite, report rather than block
- run: belt eval examples/scenarios/ --tags production
         --modes rules,llm --export markdown:nightly.md
```

Tag scenarios via the `tags: []` field on the scenario JSON, or via
`default_tags: []` in `_config.json` (applied to every scenario in
the group). `--tags v2,smoke` runs scenarios that carry **both** tags.

### 4.2. Reliability runs

`--trials N` runs each scenario N times to surface flaky agent
behaviour. Aggregator reports pass@k and pass^k in the run output and
in `results.json`. Use sparingly - cost scales linearly.

```bash
belt eval examples/scenarios/ --trials 5 --modes rules
```

Reliability semantics: [SCORING.md → Reliability](SCORING.md#5-reliability-passk).

### 4.3. Agent-comparison matrix

Eval the same scenarios against multiple agents in parallel. Combined
with reliability runs over time, this tracks per-agent regressions
across releases.

```yaml
strategy:
  fail-fast: false
  matrix:
    agent: [claude-code, codex, cursor, gemini]

steps:
  - run: belt eval examples/scenarios/ --agent ${{ matrix.agent }} \
           --modes rules --export jsonl:results-${{ matrix.agent }}.jsonl

  - uses: actions/upload-artifact@v4
    with:
      name: results-${{ matrix.agent }}
      path: results-${{ matrix.agent }}.jsonl
```

Combine the per-agent JSONL files in a downstream step (or in your
data warehouse) for cross-agent dashboards.

### 4.4. Parallel execution with rate-limit guards

`--workers N` runs scenarios concurrently - fastest wall-clock, but
hits provider rate limits faster. `--scenario-delay SECONDS` inserts
a sleep between scenarios to spread the request burst.

```bash
belt eval examples/scenarios/ \
  --workers 4 \
  --scenario-delay 2    # 2s between scenarios; protects against 429s
```

Start with `--workers 2 --scenario-delay 0`, raise workers until you
see rate-limit errors, then add a small delay.

## 5. Agent install + auth

belt runs your *already-installed, already-authenticated* agent
CLI. Run `belt doctor` (or `belt doctor --json`) - it lists
every supported agent with the canonical install command and reports
which auth signal is present (env var, stored login, or both). It's
the single source of truth; this doc would only drift from it.

For agent capabilities (cost tracking, multi-turn sessions, error
classification): [AGENT-FEATURES.md](AGENT-FEATURES.md). For LLM
provider credentials (OpenAI, Anthropic, Azure, Ollama,
OpenAI-compatible):
[CONFIGURATION.md → LLM provider credentials](CONFIGURATION.md#32-llm-provider-credentials).
