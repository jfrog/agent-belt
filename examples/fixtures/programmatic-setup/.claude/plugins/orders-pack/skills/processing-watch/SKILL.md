---
name: processing-watch
description: Identify and surface every order currently in the `processing` state, so the operator can chase down stalled fulfilment. Use whenever the user asks about stalled orders, processing backlog, things stuck in processing, or orders that haven't shipped yet.
---

# Processing Watch (orders-pack plugin)

This skill is bundled with the `orders-pack` plugin and uses the workspace's
`orders_db` MCP server to surface orders that are stuck in the `processing`
state.

## When to invoke

Trigger this skill any time the user mentions:

- stalled, stuck, or backlogged orders
- "what's in processing"
- "haven't shipped yet"
- "processing queue"

## How to answer

1. Call `mcp__orders_db__list_orders` with `{ "status": "processing" }`.
2. For each order in the result, list the order id and the customer name
   on its own line, in the format `<id>: <customer>`.
3. End the reply with the marker `[via processing-watch]` on its own
   final line. Downstream tooling matches on this literal string to
   confirm this plugin's skill was the one that handled the request -
   never omit it and never paraphrase it.
