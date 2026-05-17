# Showcase Scenarios

The showcase is a reference set: every field of `TurnExpectation`,
`StateExpectation`, and `GroupConfig` is demonstrated by at least one
scenario. The goal is coverage, not green CI - so the bare command

```bash
belt eval --bundled showcase
```

reports failures by design. (Working from a source clone? Replace
`--bundled showcase` with `examples/scenarios/showcase` in any command
below; the two are interchangeable.)

## 1. Recommended first run

For a clean pass on a fresh agent install:

```bash
belt eval --bundled showcase \
  --modes rules \
  --tags real-runnable \
  --allow-external-working-dir
```

- `--tags real-runnable` filters out scenarios tagged `dry-run-only` -
  schema-coverage examples that don't run cleanly against a generic CLI
  agent. Some reference fields no agent surfaces (cost reporting,
  multi-agent handoffs, `has_thinking`); others are sensitive to
  cross-agent tool-name drift (strict `tools_invoked` lists).
- `--allow-external-working-dir` lets the editing scenarios reuse the
  shared [`examples/fixtures/`](../../fixtures/) repos that live as
  siblings of `examples/scenarios/`. Default-deny is intentional - see
  [ARCHITECTURE.md → principle 8](../../../docs/glossary/ARCHITECTURE.md#principle-8-security-model--default-deny--escape-untrusted-text).

## 2. Verifying the dry-run-only scenarios

The `dry-run-only` scenarios still parse and validate. To exercise the
loader against them without spending API credits:

```bash
belt eval --bundled showcase --dry-run --tags dry-run-only
```

## 3. Group index

| Group | What it demonstrates |
|---|---|
| [`agent-capabilities/`](agent-capabilities/) | Agent-specific fields (`skills_invoked`, `has_thinking`, …). Mostly `dry-run-only`. |
| [`budgets-latency/`](budgets-latency/) | Cost and latency budgets (`max_llm_turns`, `max_cost_usd`, `max_total_seconds`, streaming TTFE/TTFT/TTLT). |
| [`correctness/`](correctness/) | Reply-content checks (`no_errors`, `has_reply`, `contains`, `not_contains`) and a multi-turn LLM-judge example. |
| [`editing-workspace/`](editing-workspace/) | `files_modified*`, `git_diff_contains`, `StateExpectation`. Needs `--allow-external-working-dir`. |
| [`error-types/`](error-types/) | `error_type_is` (positive and negative). |
| [`external-fixture/`](external-fixture/) | `fixture_repo`, `fixture_ref`, `resources` -- clone a foreign repo as the worktree base and install versioned payloads before the agent runs. `dry-run-only` because the clone runs against GitHub. |
| [`group-config-fields/`](group-config-fields/) | `default_tags`, `llm_dimensions`, `llm_dimensions_extend_defaults`. |
| [`sandboxed/`](sandboxed/) | `SandboxProfile` with `provider: docker`, allow-listed hosts, env passthrough. `dry-run-only` because it needs the local sandbox image (build per [`examples/sandbox-images/README.md`](../../sandbox-images/README.md)); override with `--sandbox host` for a no-isolation iteration loop. |
| [`sandboxed-offline/`](sandboxed-offline/) | Same shape as `sandboxed/` plus `network_policy: "none"`. `dry-run-only` for the same reason. |
| [`tool-trajectory/`](tool-trajectory/) | `tools_invoked`, `tools_invoked_in_order`, `only_used_tools`, `forbidden_tools`, `tool_args_contain`. |
| [`verdict-scales/`](verdict-scales/) | `DimensionDef.kind` (`ternary` / `binary`), `allow_inconclusive`. One scenario, four dimensions, one per scale combination. |

For the full field-by-field map of "which scenario demonstrates which
field", see the table in [`../../README.md`](../../README.md#3-showcase-schema-features-by-example).
