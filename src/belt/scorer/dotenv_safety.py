# (c) JFrog Ltd. (2026)

"""Owner-checked dotenv loading for the scoring phase.

The scorer reads scoring config (model selection, base URLs, API keys) from
a ``.env`` file in cwd. A hostile cwd would let an attacker inject
``OPENAI_BASE_URL=https://evil`` and silently redirect judge calls. We
refuse to load the file unless we own it and it isn't group/world-writable.

Set ``BELT_NO_DOTENV=1`` to skip the load entirely (sandboxed CI).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from belt import envvars
from belt.constants import ENV_FILE

_dotenv_banner_emitted = False


def load_dotenv_safely() -> None:
    """Load ``ENV_FILE`` only when owned by the invoking user and not world-writable.

    The ownership check (``st.st_uid == euid``) and the
    group/world-writable check (``st.st_mode & 0o022``) are POSIX
    concepts and are no-ops on Windows: ``os.geteuid`` is absent there,
    so the ownership branch is gated by ``hasattr(os, "geteuid")``, and
    the mode bits Windows fills in via ``stat`` are not enforceable as
    Unix permissions. On Windows the file is loaded as long as it
    exists - this matches every other dotenv loader in the Python
    ecosystem and reflects that NTFS ACLs, not POSIX mode bits, are the
    sanctioned mechanism there.
    """
    global _dotenv_banner_emitted

    if envvars.is_truthy(envvars.NO_DOTENV):
        if not _dotenv_banner_emitted:
            logger.info("Skipping dotenv load ({}=1)", envvars.NO_DOTENV)
            _dotenv_banner_emitted = True
        return

    env_path = Path(ENV_FILE)
    if not env_path.exists():
        return

    try:
        st = env_path.stat()
    except OSError as exc:
        logger.warning("Could not stat dotenv file {}: {} - skipping load", env_path, exc)
        return

    if hasattr(os, "geteuid"):
        euid = os.geteuid()
        if st.st_uid != euid:
            logger.warning(
                "Refusing to load dotenv {} (owner uid={} ≠ euid={}); set {}=1 "
                "to silence this once you have moved or fixed the file.",
                env_path,
                st.st_uid,
                euid,
                envvars.NO_DOTENV,
            )
            return
    if st.st_mode & 0o022:
        logger.warning(
            "Refusing to load dotenv {} (mode 0o{:o} is group/world writable); chmod 0600 {} to enable loading.",
            env_path,
            st.st_mode & 0o777,
            env_path,
        )
        return

    if not _dotenv_banner_emitted:
        logger.info("Loading dotenv from {}", env_path)
        _dotenv_banner_emitted = True
    load_dotenv(env_path, override=False)
