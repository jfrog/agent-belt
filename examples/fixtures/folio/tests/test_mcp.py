# (c) JFrog Ltd. (2026)

"""End-to-end MCP test: client → uvicorn → FastMCP → FolioDB.

Boots a uvicorn process on a free port, opens a real Streamable HTTP
MCP client against it, and exercises a representative tool path.
"""

from __future__ import annotations

import asyncio
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

_FIXTURE_ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_health(port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/api/health", timeout=1.0)
            if r.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    raise RuntimeError(f"Folio server did not come up on :{port} within {timeout}s")


@pytest.fixture(scope="module")
def folio_server():
    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "folio.server",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(_FIXTURE_ROOT),
    )
    try:
        _wait_for_health(port)
        yield port
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_mcp_lists_tools_and_calls_search(folio_server: int) -> None:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async def _run() -> dict:
        async with streamablehttp_client(f"http://127.0.0.1:{folio_server}/mcp/") as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = sorted(t.name for t in tools.tools)
                search = await session.call_tool("search_books", {"query": "Stephenson", "limit": 5})
                return {"tool_names": names, "text": search.content[0].text}

    result = asyncio.run(_run())
    expected = {
        "check_stock",
        "escalate_to_human",
        "get_book",
        "get_customer",
        "get_order",
        "issue_store_credit",
        "list_customer_orders",
        "list_customers",
        "place_order",
        "refund_order",
        "search_books",
    }
    assert expected.issubset(set(result["tool_names"]))
    assert "Snow Crash" in result["text"]
    assert "Anathem" in result["text"]
