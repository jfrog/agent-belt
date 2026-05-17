# Folio - the bookstore SaaS demo fixture

Folio is a small fictional bookstore that ships **a domain MCP server +
a customer-support skill + a slash command + a plugin (command + skill)**
to the coding agent its customers use. The fixture is consumed by the
scenario group under
`examples/scenarios/experience/folio/`.

Running the eval answers the launch-post question
*"is what we ship to our customers' agents still doing the right thing
this week?"* across four distinct Claude Code customization surfaces in
one ~4-minute run.

## 1. What's inside

```text
folio/                  # the running service (FastAPI + FastMCP)
  api.py                #   REST endpoints (incl. POST /api/admin/reset)
  mcp_server.py         #   MCP tools (mirrors REST surface)
  server.py             #   uvicorn entry point - serves both
  db.py                 #   SQLite, re-seeded on every startup or reset()
  reset.py              #   `python -m folio.reset` -> calls /api/admin/reset
data/                   # deterministic seed data
  seed.sql              #   30 books, 5 customers
  orders_seed.json      #   7 orders with policy-graded ages
.mcp.json               # Claude MCP wiring (HTTP transport, vendor-pinned path)
.claude/                # Claude Code wiring
  settings.json         #   project MCP enabled, tool allow-list
  skills/bookstore-assistant/SKILL.md         # tier-1/2/3 refund ladder skill
  commands/refund.md                          # /refund <order_id> <reason>
  plugins/folio-ops/.claude-plugin/plugin.json # plugin manifest
  plugins/folio-ops/commands/lookup.md        # /folio-ops:lookup command
  plugins/folio-ops/skills/inventory-audit/SKILL.md # plugin skill
pyproject.toml          # local package: fastapi + uvicorn + mcp
tests/                  # service-level tests (pure HTTP / pure MCP)
```

## 2. Running Folio

The Folio server is HTTP-only - we simulate the remote SaaS the agents
would connect to in production. Folio is its own self-contained uv
project (`pyproject.toml` here pins fastapi + uvicorn + mcp); it does
NOT borrow from the agent-belt venv. **Start it before running any
`belt eval` against this fixture.**

```bash
cd examples/fixtures/folio
uv sync                          # one-time, creates ./.venv with the right deps
uv run python -m folio.server
#  -> Folio is live on http://127.0.0.1:8765
#     REST: /api/health, /api/books?q=…, /api/orders/{id}, …
#     MCP : /mcp/   (Streamable HTTP transport - what agents connect to)
```

Override host/port via flag or env: `--host 0.0.0.0 --port 9000`,
`FOLIO_HOST`, `FOLIO_PORT`.

Re-seeds on every start, so order #1003 is always 90 days post-delivery
no matter when you run the demo.

## 3. Smoke test

```bash
# REST
curl localhost:8765/api/health
curl 'localhost:8765/api/books?q=Stephenson&limit=5'
curl localhost:8765/api/orders/1003       # 90 days old -> escalation territory

# REST + MCP roundtrip in one go (run from this directory)
uv run python -m folio.tests.smoke_mcp
```

## 4. Running the eval

Assumes the LLM judge model and provider credentials are already set in
the environment (`BELT_LLM_MODEL` + the provider's `BELT_*` vars - see
[CONFIGURATION.md](../../../docs/glossary/CONFIGURATION.md)).

```bash
# Terminal 1 - the SaaS. Run from examples/fixtures/folio/:
cd examples/fixtures/folio
uv sync && uv run python -m folio.server

# Terminal 2 - INITIALIZE before every eval invocation.
#
# Folio holds its state in-memory and the MCP endpoint is shared by every
# client that connects (IDE agents, ad-hoc curl, a previous eval run).
# Any of them can mutate orders 1001 / 1002 / 1003 between server start
# and your eval; if they do, scenarios L2 / L3 / L4 see the order already
# in its terminal state ("refunded" / "credited" / "escalated") and skip
# the action under test. The init command wipes the DB and re-seeds the
# deterministic snapshot.
cd examples/fixtures/folio
uv run python -m folio.reset            # or: curl -X POST http://127.0.0.1:8765/api/admin/reset

# Terminal 2 - the eval. MUST run from the agent-belt repo root (the
# scenario paths below are relative to that root). --allow-external-
# working-dir is required: the scenarios point at the Folio fixture,
# which lives outside the scenarios directory.
cd <agent-belt-repo-root>
belt eval examples/scenarios/experience/folio --modes rules,llm \
    --allow-external-working-dir
```

### What the scenarios prove

The ten scenarios (L1 - L10) each pin one Claude Code customization
surface so the eval can tell, from artifacts alone, that the right
surface loaded:

| Scenario | Surface | Pinned by |
|---|---|---|
| L1 catalog_search | MCP server | `tool_invoked(mcp__folio__search_books)` |
| L2 refund_within_policy (tier-1) | Top-level skill | `skill_invoked(bookstore-assistant)` + `mcp__folio__refund_order` |
| L3 store_credit_tier (tier-2) | Top-level skill | `skill_invoked(bookstore-assistant)` + `mcp__folio__issue_store_credit` |
| L4 escalate_out_of_policy (tier-3) | Top-level skill | `skill_invoked(bookstore-assistant)` + `mcp__folio__escalate_to_human` |
| L5 partial_stock_guard | Skill + MCP composition | `forbidden_tools` + `tools_invoked_any` |
| L6 unknown_order_no_hallucination | Grounding | `forbidden_tools(refund_order, ...)` + reply assertion |
| L7 place_then_refund_multiturn | Multi-turn state continuity | turn 0 `place_order` + turn 1 `refund_order` |
| L8 slash_command_refund | Project slash command | `reply_pattern(REFUND\|...\|tag=folio-cmd-v1)` (marker exists only in `.claude/commands/refund.md`) |
| L9 plugin_command_lookup | Plugin slash command | `reply_pattern(LOOKUP\|...\|tag=folio-ops-v1)` (marker exists only in plugin command file) |
| L10 plugin_skill_inventory | Plugin skill | `skill_invoked(folio-ops:inventory-audit)` |

The `tag=folio-cmd-v1` / `tag=folio-ops-v1` markers are deliberately
unique strings - they exist nowhere in the workspace except inside the
command files for that surface. Seeing them in the reply, combined with
the corresponding MCP call in the trajectory, is hard proof that the
agent loaded and executed the right primitive.

### What "failure" means in this demo

Two scenarios are expected to surface honest model variance rather than
plumbing bugs:

- **L2** - tier-1 refund: the agent sometimes over-explains alternatives
  (mentions store credit or escalation paths even though the skill is
  explicit that tier-1 should refund and stop). The `not_contains` guards
  catch it.
- **L7** - multi-turn refund: the agent reliably invokes the skill on
  turn 0 (place_order) but sometimes does not re-invoke it on turn 1
  (refund_order). The `skills_invoked` assertion on the second turn
  catches it.

Both failures are *the point*: this is the kind of cross-turn / cross-policy
regression a hand-test on a single agent in isolation would miss, and which
re-running the same group week-over-week will surface as the model evolves.

## 5. The graded refund policy (what's actually being scored)

The `bookstore-assistant` skill encodes a three-tier ladder. Each
scenario lands on a specific tier so the eval can prove the agent
picked the right tool:

| Tier | Elapsed time since `delivered_at` | Tool                  | Example order |
| ---- | --------------------------------- | --------------------- | ------------- |
| 1    | ≤ 30 days                         | `refund_order`        | #1001 (5 days post-delivery)  |
| 2    | 31-60 days                        | `issue_store_credit`  | #1002 (45 days post-delivery) |
| 3    | > 60 days, damage, fraud, etc.    | `escalate_to_human`   | #1003 (90 days post-delivery) |

The forbidden_tools assertion on every refund scenario catches the
classic sycophantic-agent failure: silently refunding outside the
self-serve window because the customer asked nicely.

## 6. Reseeding

`db.py` rebuilds the SQLite DB in memory on every uvicorn startup. To
reset state mid-eval, kill the server and restart it.
