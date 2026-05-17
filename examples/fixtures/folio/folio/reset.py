# (c) JFrog Ltd. (2026)

"""CLI helper: reset the running Folio server's in-memory DB.

Folio is in-memory and stateful across the process lifetime, so any
MCP client that calls a mutating tool (refund / credit / escalate /
place_order) shifts the DB out of seed state. Run this before every
eval invocation so scenarios start from a known snapshot:

    python -m folio.reset
"""

from __future__ import annotations

import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_DEFAULT_BASE_URL = "http://127.0.0.1:8765"


def main() -> int:
    base_url = os.environ.get("FOLIO_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")
    url = f"{base_url}/api/admin/reset"
    req = Request(url, method="POST")
    try:
        with urlopen(req, timeout=5) as resp:
            body = resp.read().decode("utf-8")
    except HTTPError as e:
        print(f"reset failed: HTTP {e.code} {e.reason}", file=sys.stderr)
        return 1
    except URLError as e:
        print(
            f"reset failed: cannot reach {url} ({e.reason}). " f"Is `uv run python -m folio.server` running?",
            file=sys.stderr,
        )
        return 1
    print(f"Folio reset OK ({url}): {body}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
