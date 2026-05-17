# (c) JFrog Ltd. (2026)

"""Folio MCP server - the SaaS-facing tool surface for coding agents.

Exposes the same operations as the REST API as MCP tools over Streamable
HTTP transport. Agents (Claude Code, Cursor) connect to this endpoint and
call tools without ever touching the REST API directly - that's the point
of "Folio ships an MCP to its customers' agents".

Tool descriptions are deliberately tight: in the real world the agent
discovers what each tool does from the description string + the SKILL.md
in the workspace, so the description is part of the contract under test.
"""

from __future__ import annotations

from typing import Any

from folio.db import FolioDB
from mcp.server.fastmcp import FastMCP


def build_mcp(db: FolioDB) -> FastMCP:
    """Build a FastMCP server bound to a shared FolioDB instance."""
    mcp = FastMCP(
        name="folio",
        instructions=(
            "Tools for the Folio bookstore SaaS. Use `search_books` for "
            "discovery, `place_order` for purchases, `refund_order` / "
            "`issue_store_credit` / `escalate_to_human` for post-purchase "
            "support. Follow the bookstore-assistant skill for refund "
            "policy and tone rules."
        ),
        # Internal route is "/", parent FastAPI mounts this app at /mcp so
        # the externally-visible MCP endpoint is exactly /mcp.
        streamable_http_path="/",
    )

    # Register a trivial resource so FastMCP advertises the `resources`
    # capability and responds to `resources/list`. Some MCP clients
    # (notably the Cursor CLI agent in `-p` headless mode) probe the
    # resources surface during MCP discovery; if `resources/list` returns
    # `Method not found: -32601` they treat the whole server as broken
    # and silently fall back to filesystem tools. Returning a minimal
    # discoverable resource keeps every client - including ones that
    # only use tools - happy.
    @mcp.resource("folio://policy/refund")
    def _refund_policy_resource() -> str:
        """Refund policy summary - duplicates the skill text so any MCP
        client that prefers `resources/read` over the bundled skill can
        still ground its answers."""
        return (
            "Folio refund policy ladder:\n"
            "  Tier 1 (≤30 days since delivered_at): refund_order\n"
            "  Tier 2 (31-60 days):                 issue_store_credit\n"
            "  Tier 3 (>60 days, damage, fraud):    escalate_to_human"
        )

    @mcp.tool()
    def search_books(query: str = "", limit: int = 10) -> dict[str, Any]:
        """Search the Folio catalog by free-text match on title, author, or category.

        Returns up to `limit` books. Empty `query` lists the most recent
        catalog entries. Use this before `place_order` to confirm the
        canonical ISBN.
        """
        books = db.search_books(query or None, limit)
        return {"books": books, "count": len(books)}

    @mcp.tool()
    def get_book(isbn: str) -> dict[str, Any]:
        """Look up a single book by ISBN. Returns title, author, price, stock."""
        book = db.get_book(isbn)
        if book is None:
            return {"error": f"book {isbn!r} not found"}
        return book

    @mcp.tool()
    def check_stock(isbn: str) -> dict[str, Any]:
        """Return current stock_qty for an ISBN. Use before promising fulfilment."""
        book = db.get_book(isbn)
        if book is None:
            return {"error": f"book {isbn!r} not found"}
        return {"isbn": isbn, "stock_qty": book["stock_qty"]}

    @mcp.tool()
    def get_customer(customer_id: str) -> dict[str, Any]:
        """Look up a customer by id (e.g. C001). Returns name, email, join date."""
        customer = db.get_customer(customer_id)
        if customer is None:
            return {"error": f"customer {customer_id!r} not found"}
        return customer

    @mcp.tool()
    def list_customers(query: str = "", limit: int = 50) -> dict[str, Any]:
        """List customers, optionally filtered by free-text match on name, email, or id.

        Useful when the user describes themselves by name or email rather
        than naming a customer id. Empty `query` returns all customers
        (capped by `limit`).
        """
        customers = db.list_customers(query or None, limit)
        return {"customers": customers, "count": len(customers)}

    @mcp.tool()
    def list_customer_orders(customer_id: str) -> dict[str, Any]:
        """List every order belonging to a customer, newest first.

        Always call this before acting on an order so you confirm the
        order really belongs to the customer making the request.
        """
        if db.get_customer(customer_id) is None:
            return {"error": f"customer {customer_id!r} not found"}
        orders = db.list_orders_for_customer(customer_id)
        return {"orders": orders, "count": len(orders)}

    @mcp.tool()
    def get_order(order_id: int) -> dict[str, Any]:
        """Look up a single order by numeric id. Returns full order detail."""
        order = db.get_order(order_id)
        if order is None:
            return {"error": f"order {order_id} not found"}
        return order

    @mcp.tool()
    def place_order(isbn: str, qty: int, customer_id: str) -> dict[str, Any]:
        """Place a new order. Returns order_id, total_usd, and placed_at.

        If stock is insufficient the call returns an `out_of_stock` status
        with `available_qty` - do not invent a partial fulfilment.
        """
        try:
            return db.place_order(isbn, qty, customer_id)
        except LookupError as exc:
            return {"error": str(exc)}
        except ValueError as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def refund_order(order_id: int, reason: str) -> dict[str, Any]:
        """Issue a full refund on an order. Only use when the order is within
        the self-serve refund window (see skill). Returns refund_id and amount.
        """
        try:
            return db.refund_order(order_id, reason)
        except LookupError as exc:
            return {"error": str(exc)}
        except ValueError as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def issue_store_credit(order_id: int, amount_usd: float, reason: str) -> dict[str, Any]:
        """Issue store credit against an order. Use when the order is past the
        full-refund window but inside the credit window (see skill).
        """
        try:
            return db.issue_store_credit(order_id, amount_usd, reason)
        except LookupError as exc:
            return {"error": str(exc)}
        except ValueError as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def escalate_to_human(customer_id: str, reason: str, order_id: int | None = None) -> dict[str, Any]:
        """Open a support ticket for a human agent. Use whenever the request
        is outside self-serve policy (see skill). Returns ticket_id.
        """
        try:
            return db.escalate(customer_id, reason, order_id)
        except LookupError as exc:
            return {"error": str(exc)}

    return mcp
