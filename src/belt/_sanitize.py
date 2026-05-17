# (c) JFrog Ltd. (2026)

"""Stream sanitisation primitives.

Single source of truth for ANSI-escape and ASCII-control stripping
across belt. Every site that consumes untrusted bytes from an
agent CLI and forwards them into a markup-aware sink (Markdown card
cells, Rich panels, version capture, stderr logs) routes through the
helpers in this module. Concentrating both the regexes and the
sanitisation policy here means a new escape shape (e.g. OSC hyperlinks,
xterm window-title sequences) is handled in exactly one place; callers
no longer roll their own ``re.compile(r"\\x1b\\[...")`` and silently
miss the cases the canonical regexes cover.

Two flavours of helper, distinguished by name on purpose (mirroring
:mod:`belt._redact`):

- ``strip_*`` - **pure transform**. Removes one class of bytes and
  returns the result. No policy, no defaults, easy to compose.
- ``sanitize`` - **curated pipeline**. Applies the strip helpers in the
  order an arbitrary agent stream needs to be safe for Markdown / Rich
  sinks. The default for "this string came from an external CLI and
  is about to render somewhere".

The regexes are kept narrow on purpose. ``strip_controls`` does **not**
touch high-Unicode (so an agent named ``Goose 1.4 ⚡`` survives), and
``strip_ansi`` does **not** assume the input is well-formed UTF-8 (a
lone ESC byte from a malformed sequence is preserved for ``strip_controls``
to remove on the next pass).
"""

from __future__ import annotations

import re

# Strip ANSI CSI sequences as a complete unit so the residual ``[31m...[0m``
# body is removed alongside the ESC byte itself. The character classes are
# the broadest the ECMA-48 grammar admits: ``[0-?]`` parameter bytes,
# ``[ -/]`` intermediate bytes, ``[@-~]`` final byte.
_ANSI_CSI_RE: re.Pattern[str] = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

# OSC (Operating System Command): ``\x1b]...BEL`` or ``\x1b]...\x1b\``.
# Matters because terminal hyperlinks (OSC 8) and xterm window-title
# sequences (OSC 0/2) are common in modern CLI output and would otherwise
# render as visible noise in a Markdown card.
_ANSI_OSC_RE: re.Pattern[str] = re.compile(r"\x1b\][^\x07]*(?:\x07|\x1b\\)")

# ASCII C0 control bytes (0x00-0x1F) plus DEL (0x7F). Covers raw
# newlines/tabs that would split a Markdown table row, NUL bytes, and
# lone ESC bytes left behind by malformed sequences. Kept narrow on
# purpose - high-Unicode characters are preserved.
_ASCII_CONTROL_RE: re.Pattern[str] = re.compile(r"[\x00-\x1f\x7f]")

# ``\x1b[2K`` (CSI Erase In Line, mode 2) clears the current line. CLIs
# that draw a spinner overwrite the same line by emitting this escape
# repeatedly; the *last* segment after the final erase is what the user
# would see on a real terminal. Exposed as a constant so the simulation
# in :func:`strip_ansi` is greppable.
ERASE_LINE: str = "\x1b[2K"


def strip_ansi(text: str, *, simulate_erase: bool = False) -> str:
    """Remove ANSI CSI and OSC escape sequences from ``text``.

    When ``simulate_erase`` is True, a pre-pass keeps only the segment
    after the final ``\\x1b[2K`` (erase-line) so spinner / progress
    output resolves to the line the user would actually see on a real
    terminal. ``\\r`` (carriage return) is also stripped in this mode
    because spinners commonly pair the two.

    Returns the input unchanged if no escapes are present.
    """
    if simulate_erase:
        if ERASE_LINE in text:
            text = text.split(ERASE_LINE)[-1]
        text = text.replace("\r", "")
    text = _ANSI_CSI_RE.sub("", text)
    text = _ANSI_OSC_RE.sub("", text)
    return text


def strip_controls(text: str) -> str:
    """Remove ASCII C0 (``\\x00``-``\\x1f``) and DEL (``\\x7f``) bytes.

    Used after :func:`strip_ansi` (or via :func:`sanitize`) to remove
    the residue of malformed escapes plus any embedded newlines or NUL
    bytes that would let an attacker break out of a Markdown table cell
    or Rich panel.
    """
    return _ASCII_CONTROL_RE.sub("", text)


def sanitize(text: str, *, simulate_erase: bool = False) -> str:
    """Apply the full strip pipeline: ANSI escapes then ASCII controls.

    The canonical entry point for any string that crosses an
    untrusted-bytes boundary on its way to a markup-aware sink. Pass
    ``simulate_erase=True`` for streams that contain spinner output
    (e.g. ``cursor`` ``--version`` or ``codex`` doctor probes).
    """
    return strip_controls(strip_ansi(text, simulate_erase=simulate_erase))


__all__ = [
    "ERASE_LINE",
    "sanitize",
    "strip_ansi",
    "strip_controls",
]
