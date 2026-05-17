# (c) JFrog Ltd. (2026)

"""Output-escaping helpers for markup-aware sinks.

belt renders evaluation artefacts into several surfaces that
interpret formatting or formula markup:

* a Rich ``Panel`` / ``Console`` (terminal): square-bracket Rich markup
  is parsed by :func:`rich.text.Text.from_markup` and by
  ``Console.print``.
* a GitHub Step Summary (CI): rendered as Markdown by GitHub Actions.
* a CSV file opened in a spreadsheet: leading ``=`` / ``+`` / ``-`` /
  ``@`` / ``\\t`` / ``\\r`` cells are interpreted as formulas.
* a JUnit XML report consumed by CI test reporters: ``&`` / ``<`` /
  ``>`` are treated as XML metacharacters.
* a structured viewer (TUI / web): currently delegates to Rich for
  terminal output.

Several values reaching those sinks originate from agent stdout, LLM
judge ``reasoning`` strings, agent ``--version`` output, fixture
working-dir paths, scenario tags, rule-check ``details``, and
free-form scenario filter strings - all of which are
attacker-controllable. A judge prompted to grade malicious agent
text can produce ``[red]System failure[/red]``,
``</details><script>...``, ``=cmd|'/c calc'!A1``, or
``\\`# Inject heading`` and have it interpreted at the rendering layer.

This module is the single source of truth for every per-sink escape
rule used by a persisted or terminal-bound output. ANSI and
ASCII-control stripping are layered on top via
:mod:`belt._sanitize` (called from inside the Markdown helpers)
so the per-sink helper takes care of everything between "untrusted
bytes" and "safe to print".

Helpers, distinguished by sink:

- :func:`rich_safe` - Rich console / panel / table.
- :func:`md_safe` - Markdown **prose** (outside code spans).
- :func:`md_inline` - Markdown **inline-code span** (table cells,
  ``\\`...\\``` spans). Wraps the value in backticks and neutralises the
  characters that would close the span or split a table row.
- :func:`csv_safe` - CSV cell value. Neutralises spreadsheet formula
  injection by prefixing risky leading characters with a single quote.
- :func:`xml_safe` - XML text content (e.g. JUnit ``<failure>`` body).
  Escapes ``&``, ``<``, ``>``.
"""

from __future__ import annotations

from xml.sax.saxutils import escape as _xml_escape

from rich.markup import escape as _rich_escape

from belt._sanitize import strip_controls

# OWASP-recommended set: cells starting with any of these characters
# trigger formula evaluation in major spreadsheet apps (Excel, Sheets,
# LibreOffice, Numbers). Tab + CR are included because Excel reuses
# them as field-spec hints.
_CSV_RISKY_LEADING_CHARS: frozenset[str] = frozenset(("=", "+", "-", "@", "\t", "\r"))

# Markdown control characters we want to neutralise. We escape the small
# set that lets a malicious string break out of an inline ``- ...``
# bullet list, end a code span, or open an HTML comment / tag. The
# escaped form is kept readable ("\\*" rather than HTML entity) so the
# fix is visible in the rendered Step Summary.
_MD_CHARS = ("\\", "`", "*", "_", "{", "}", "[", "]", "(", ")", "#", "+", "-", ".", "!", "|", "<", ">")


def rich_safe(value: object) -> str:
    """Return ``str(value)`` with Rich markup neutralised.

    Replaces ``[`` with ``\\[`` so :func:`rich.text.Text.from_markup`
    and ``Console.print`` treat the string as literal text.
    """
    if value is None:
        return ""
    return _rich_escape(str(value))


def md_safe(value: object) -> str:
    """Return ``str(value)`` safe to embed in GitHub-flavoured Markdown prose.

    Escapes the punctuation set that GFM treats as control characters
    and collapses runs of whitespace so multi-line judge reasoning
    cannot break out of an enclosing list item or table cell. ASCII
    control bytes are stripped first (defence against embedded NUL or
    raw escape sequences from agent stdout).
    """
    if value is None:
        return ""
    text = strip_controls(str(value))
    out_chars = []
    for ch in text:
        if ch in _MD_CHARS:
            out_chars.append("\\" + ch)
        elif ch in ("\n", "\r", "\t"):
            out_chars.append(" ")
        else:
            out_chars.append(ch)
    return " ".join("".join(out_chars).split())


def md_inline(value: object, *, empty: str = "`-`") -> str:
    """Return ``str(value)`` wrapped in a Markdown inline-code span.

    The result is always of the form ```` `...` ````. Inside a backtick
    code span GFM treats most punctuation as literal, so the only
    characters that need handling are the ones that would *close the
    span* (backtick) or *split the table row* (pipe, ASCII controls).

    - Backticks become single quotes (preserves intent without leaking
      content as Markdown).
    - Pipes are escaped (``\\|``) so a value embedded in a table cell
      cannot start a new column.
    - All ASCII C0 / DEL bytes (incl. embedded newlines) are stripped
      via :func:`belt._sanitize.strip_controls`.

    ``value`` of ``None`` or ``""`` collapses to ``empty`` (default
    ```-```) so callers can interpolate without a literal-empty
    backticks edge case.
    """
    if value is None or value == "":
        return empty
    text = strip_controls(str(value))
    text = text.replace("`", "'").replace("|", "\\|")
    return f"`{text}`"


def csv_safe(value: object) -> str:
    """Return ``str(value)`` safe to write as a CSV cell.

    Spreadsheet applications (Excel, Sheets, LibreOffice, Numbers) treat
    any cell beginning with ``=``, ``+``, ``-``, ``@``, ``\\t`` or ``\\r``
    as a formula. A scenario tag, an LLM ``reasoning`` snippet, or a
    rule-check ``details`` field exfiltrated from agent output can
    therefore execute when the file is double-clicked - even though the
    CSV itself is structurally well-formed.

    Following OWASP guidance, we prefix every cell that starts with one
    of those characters with a single quote so the spreadsheet renders
    the literal value. The trade-off is that a legitimately-negative
    numeric value also gains the prefix; for the columns the framework
    emits (cost / time / counts) those are always non-negative, so the
    trade-off is purely defensive.
    """
    if value is None:
        return ""
    text = str(value)
    if text and text[0] in _CSV_RISKY_LEADING_CHARS:
        return "'" + text
    return text


def xml_safe(value: object) -> str:
    """Return ``str(value)`` safe to embed as XML text content.

    Escapes ``&``, ``<`` and ``>`` via the stdlib ``xml.sax.saxutils``
    rules. The output is intended for plain text content - never wrapped
    in a CDATA section - so a ``]]>`` sequence in the input is naturally
    neutralised by the standard ``>`` -> ``&gt;`` substitution.
    """
    if value is None:
        return ""
    return _xml_escape(str(value))


__all__ = ["csv_safe", "md_inline", "md_safe", "rich_safe", "xml_safe"]
