# (c) JFrog Ltd. (2026)

"""Combined uvicorn entry point: REST API + MCP HTTP on one port.

  - REST  -> http://HOST:PORT/api/...
  - MCP   -> http://HOST:PORT/mcp  (Streamable HTTP transport)
  - Health -> http://HOST:PORT/api/health

Pure HTTP makes this look like a real remote SaaS to the coding agent.
The DB is re-seeded on every startup so scenarios are deterministic.
"""

from __future__ import annotations

import argparse
import contextlib
import os

import uvicorn
from fastapi import FastAPI
from folio.api import build_router
from folio.db import FolioDB
from folio.mcp_server import build_mcp


def build_app() -> FastAPI:
    """Wire FolioDB + REST + MCP into a single FastAPI app."""
    db = FolioDB()
    mcp = build_mcp(db)
    mcp_app = mcp.streamable_http_app()

    # FastMCP's streamable_http_app() returns a Starlette app whose
    # session_manager startup lives in `router.lifespan_context`. Mounted
    # sub-apps don't get their lifespan run automatically, so the parent
    # FastAPI must forward it.
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        async with mcp_app.router.lifespan_context(app):
            yield

    app = FastAPI(title="Folio", version="0.1.0", lifespan=lifespan)
    app.include_router(build_router(db), prefix="/api")
    app.mount("/mcp", mcp_app)
    return app


app = build_app()


def main() -> int:
    parser = argparse.ArgumentParser(description="Folio bookstore SaaS demo server.")
    parser.add_argument("--host", default=os.getenv("FOLIO_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("FOLIO_PORT", "8765")))
    parser.add_argument("--log-level", default=os.getenv("FOLIO_LOG_LEVEL", "info"))
    args = parser.parse_args()

    uvicorn.run(
        "folio.server:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
