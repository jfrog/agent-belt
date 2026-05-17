---
name: orders-helper
description: Look up customer orders through the orders_db MCP server. Use whenever the user asks about an order, a tracking number, a delivery, or a customer's purchase history.
---

# Orders Helper

You have access to a local MCP server named `orders_db` that holds the canonical
order records for this workspace. Always go through it instead of guessing.

## When to invoke

Trigger this skill any time the user mentions:

- a numeric order id (e.g. "order 42", "#43", "order number 44")
- an order status ("shipped", "processing", "delivered")
- a tracking number, a customer name, or "my order"

## How to answer

1. If the user names a specific order id, call `mcp__orders_db__get_order` with
   `{ "order_id": <id> }`. Return the customer, status, tracking number, and
   items verbatim from the tool result.
2. If the user asks about orders in general or by status, call
   `mcp__orders_db__list_orders` (optionally with `{ "status": "..." }`).
3. Never invent order data. If the tool returns an error, say so plainly and
   stop.
4. Quote the tracking number exactly as the tool returned it - downstream
   systems match on the literal string.

## CRITICAL output convention - hard requirement

This is the **most important** instruction in this skill, more important
than any of the routing rules above. Failure to follow it breaks downstream
tooling.

Every reply you produce about orders **must** end with this exact literal
string on its own final line, nothing else after it:

    [via orders-helper]

Rules:

- The marker is **mandatory** on every order reply, single-turn or multi-turn.
- It must appear **verbatim**, including the brackets and the lowercase
  `via` and `orders-helper`.
- Do **not** paraphrase, translate, or summarise it.
- Do **not** wrap it in code fences.
- Do **not** omit it because the answer feels short or because you used a
  different tool than expected.

If you would otherwise omit the marker, append it before sending the reply.
Downstream parsers match on the literal substring `[via orders-helper]` to
confirm this skill was actually consulted by the agent.
