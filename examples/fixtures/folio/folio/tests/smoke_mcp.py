# (c) JFrog Ltd. (2026)

"""Smoke test against a running Folio instance.

Hits both the REST surface and the MCP endpoint to prove they answer.
Useful for the launch-demo screencast and for sanity-checking a new
checkout before wiring an agent.

Usage (run from ``examples/fixtures/folio/`` after ``uv sync``)::

    # Terminal A
    uv run python -m folio.server

    # Terminal B
    uv run python -m folio.tests.smoke_mcp [--host 127.0.0.1] [--port 8765]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import httpx


async def _exercise_mcp(url: str) -> None:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(f"  ✓ MCP initialized; {len(tools.tools)} tools registered")
            for tool in tools.tools:
                print(f"      - {tool.name}")

            print("\n  → search_books(query='Stephenson', limit=3):")
            result = await session.call_tool("search_books", {"query": "Stephenson", "limit": 3})
            payload = json.loads(result.content[0].text)
            for book in payload["books"]:
                print(f"      {book['title']} ({book['author']}) - {book['stock_qty']} in stock")

            print("\n  → get_order(order_id=1003)  # 90 days post-delivery; Tier 3 territory:")
            result = await session.call_tool("get_order", {"order_id": 1003})
            print("      " + result.content[0].text.replace("\n", "\n      "))


def _exercise_rest(base: str) -> None:
    with httpx.Client(base_url=base, timeout=5.0) as client:
        health = client.get("/api/health").json()
        print(f"  ✓ REST /api/health -> {health}")

        books = client.get("/api/books", params={"q": "Tolkien", "limit": 3}).json()
        print(f"  ✓ REST /api/books?q=Tolkien -> {books['count']} hit(s)")

        order = client.get("/api/orders/1001").json()
        print(
            f"  ✓ REST /api/orders/1001 -> {order['status']} order for "
            f"{order['customer_id']} (placed {order['placed_at']})"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"
    mcp_url = f"{base}/mcp/"

    print(f"Smoke-testing Folio at {base}")
    print()
    print("REST surface:")
    try:
        _exercise_rest(base)
    except httpx.HTTPError as exc:
        print(f"  ✗ REST not reachable: {exc}", file=sys.stderr)
        return 1
    print()
    print(f"MCP surface ({mcp_url}):")
    try:
        asyncio.run(_exercise_mcp(mcp_url))
    except Exception as exc:  # noqa: BLE001 - this is a CLI smoke script
        print(f"  ✗ MCP roundtrip failed: {exc}", file=sys.stderr)
        return 1
    print()
    print("✓ Folio is alive and answering on both REST and MCP.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
