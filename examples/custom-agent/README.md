# Custom Agent - Plugin Template

A complete, installable belt agent plugin. Copy this directory, rename a
few identifiers, and you have your own agent registered with belt's
entry-point discovery.

The bundled `EchoAgentAdapter` returns each input message verbatim and
synthesizes a single `echo` tool call per turn - enough surface area to
exercise tool-trajectory and latency expectations without spending API
credits.

## 1. Layout

| File | Purpose |
|---|---|
| [`echo_agent.py`](echo_agent.py) | The `EchoAgentAdapter` - `BaseAgentAdapter` subclass, the only behaviour file you need to replace |
| [`pyproject.toml`](pyproject.toml) | Declares the package and registers the adapter under the `belt.agents` entry-point group |
| [`scenarios/echo/`](scenarios/echo/) | Sample scenarios that target `--agent echo`: a single-turn smoke test and a multi-turn group exercising non-trivial expectations |

## 2. Install and Run

```bash
pip install -e examples/custom-agent
belt agent list                  # confirms `echo` appears
belt eval examples/custom-agent/scenarios --agent echo --modes rules
```

`--modes rules` skips LLM judges - the bundled scenarios only need rule-based
checks, so the run is free.

## 3. Fork It as Your Own Agent

Three identifiers turn this template into a new agent. Pick a name (e.g.
`myagent`) and apply it consistently:

| Where | Change |
|---|---|
| `pyproject.toml` `[project] name` | `belt-echo` → `belt-myagent` |
| `pyproject.toml` `[project.entry-points."belt.agents"]` | `echo = "echo_agent:EchoAgentAdapter"` → `myagent = "myagent:MyAgentAdapter"` |
| `pyproject.toml` `[tool.setuptools] py-modules` | `["echo_agent"]` → `["myagent"]` |
| `echo_agent.py` filename + `EchoAgentAdapter` class name | rename to match the entry point |

Then implement the four `BaseAgentAdapter` methods (`setup`, `execute`,
`fetch_results`, `teardown`) against your real CLI. The
[`EchoAgentAdapter`](echo_agent.py) shows the universal `TurnOutput` fields
the runner consumes - populate the ones your CLI exposes; defaults are safe
for the rest.

## 4. Verify

```bash
pip install -e .
belt agent info myagent          # shows display_info() output
belt eval scenarios --agent myagent --modes rules --dry-run
```

`--dry-run` confirms scenarios discover and parse without invoking the agent.
Drop the flag once the adapter is wired up.

## 5. Further Reading

- [PLUGGABILITY.md → Authoring an agent](../../docs/glossary/PLUGGABILITY.md#6-authoring-an-agent) -
  full `BaseAgentAdapter` contract, optional capabilities, streaming, error types.
- [PLUGGABILITY.md → Discovery](../../docs/glossary/PLUGGABILITY.md#4-discovery) -
  how entry-point discovery works and when to use `--allow-arbitrary-agent`.
- [AGENT-FEATURES.md](../../docs/glossary/AGENT-FEATURES.md) - capability
  matrix for the bundled agents; aim to fill the same columns where your
  CLI supports them.
