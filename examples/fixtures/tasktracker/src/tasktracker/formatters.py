# (c) JFrog Ltd. (2026)

"""Output formatting - table and JSON."""

import json


def format_table(tasks):
    if not tasks:
        return "No tasks found."

    headers = ["ID", "Title", "Project", "Status"]
    rows = []
    for t in tasks:
        status = "done" if t.done else "pending"
        rows.append([str(t.id), t.title, t.project, status])

    # BUG: off-by-one in column width calculation - uses len(header) instead of
    # max(len(header), max(len(cell))) so long values overflow columns
    col_widths = [len(h) for h in headers]

    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    separator = "  ".join("-" * w for w in col_widths)

    lines = [header_line, separator]
    for row in rows:
        lines.append("  ".join(val.ljust(w) for val, w in zip(row, col_widths)))

    return "\n".join(lines)


def format_json(tasks):
    data = [t.to_dict() for t in tasks]
    return json.dumps(data, indent=2)
