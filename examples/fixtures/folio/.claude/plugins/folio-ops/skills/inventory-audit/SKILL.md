---
name: inventory-audit
description: Identify Folio books with low on-hand stock so the operator can replenish before the title goes out of stock. Use whenever the user asks about books running low, low stock, restock list, what's almost gone, stock audit, or which titles need reordering.
---

# Inventory Audit (folio-ops plugin)

This skill is bundled with the `folio-ops` plugin and uses the workspace's
Folio MCP server to surface books whose on-hand stock has dropped to a
low-water mark.

## When to invoke

Trigger this skill any time the user mentions:

- books running low
- low stock / what's almost gone
- stock audit / inventory audit
- which titles need reordering

## How to answer

1. Call `mcp__folio__search_books` with an empty query and `limit=50`
   to retrieve every book.
2. Filter the result to books with `stock_qty < 10`.
3. List each low-stock book on its own line in the format
   `<isbn> | <title> | stock=<qty>`, sorted by `stock_qty` ascending.
4. End the reply with the marker `[via inventory-audit]` on its own
   final line. Downstream tooling matches on this literal string to
   confirm this plugin's skill handled the request - never omit it,
   never paraphrase it.
