# (c) JFrog Ltd. (2026)

"""Cross-agent comparison - reads two results.json files and produces a side-by-side report.

Usage:
    belt compare results_a.json results_b.json
    belt compare results_a.json results_b.json --output markdown
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

from belt._ui import eprint

# Ternary verdicts have a natural ordinal that lets ``compare`` produce
# numeric deltas (``high`` - ``low`` = +2). Binary verdicts deliberately
# do not appear here: ``pass``/``fail`` are nominal, not ordinal, and
# the renderer falls back to "score unchanged" when a delta can't be
# computed.
SCORE_RANK = {"low": 0, "medium": 1, "high": 2}


def load_results(path: Path) -> dict:
    """Load a results.json file.

    Raises ``BeltError`` if the file can't be read or parsed.
    """
    try:
        return json.loads(path.read_text())
    except Exception as e:
        from belt.errors import BeltError

        raise BeltError(f"Failed to load {path}: {e}") from e


def _extract_label(path: Path) -> str:
    """Derive a short label from the results path."""
    parts = path.parts
    for i, p in enumerate(parts):
        if p == "outcomes" and i + 1 < len(parts):
            return "/".join(parts[i + 1 :]).removesuffix("/results.json")
    return path.stem


def _scenario_key(s: dict) -> str:
    return f"{s.get('group', '')}/{s.get('scenario_name', '')}"


def _iter_llm_dim_scores_from_scenario(scenario: dict) -> Iterable[tuple[str, str, str | None]]:
    """Yield (scorer_name, dimension, score_token) from a serialised scenario.

    Operates at the dict level (no Pydantic) so :command:`belt compare`
    can read older / cross-version ``results.json`` files without
    payload-class drift breaking it.

    Handles both LLM payload shapes:

    - scenario-level: ``{"dimensions": {dim: {"score": ...}}}``
    - per-turn:       ``{"turns": [{"dimensions": {dim: {"score": ...}}}]}``
      For per-turn, applies worst-of-turns so each dim contributes one
      row, matching ``iter_llm_verdicts``.
    - any scorer name (multi-judge ``--scorer-config`` writes
      ``judge_a``/``judge_b``) is walked - not just the literal
      ``"llm"`` key.
    """
    for scorer_name, payload in (scenario.get("scores") or {}).items():
        if not isinstance(payload, dict):
            continue
        sv = payload.get("schema_version") or ""
        if not sv.startswith(("llm.v", "per_turn_llm.v")):
            # Cross-version: also accept payloads that have a
            # ``dimensions`` or ``turns`` shape without an explicit
            # version (older serialisations).
            if "dimensions" not in payload and "turns" not in payload:
                continue
        if "turns" in payload and isinstance(payload.get("turns"), list):
            # Worst-of-turns rollup per dim. Rank mirrors
            # _WORST_OF_TURNS_RANK in scorer.payloads; keep it in-file to
            # avoid importing the Pydantic stack just for a dict walk.
            rank = {"inconclusive": -1, "fail": 0, "low": 0, "medium": 1, "high": 2, "pass": 2}
            worst: dict[str, str] = {}
            for tv in payload["turns"]:
                if not isinstance(tv, dict):
                    continue
                for dim, vd in (tv.get("dimensions") or {}).items():
                    if not isinstance(vd, dict):
                        continue
                    token = vd.get("score")
                    if not token:
                        continue
                    prev = worst.get(dim)
                    if prev is None or rank.get(token, 1) < rank.get(prev, 1):
                        worst[dim] = token
            for dim, token in worst.items():
                yield scorer_name, dim, token
        else:
            for dim, vd in (payload.get("dimensions") or {}).items():
                if isinstance(vd, dict) and "score" in vd:
                    yield scorer_name, dim, vd.get("score")


def _discover_dimensions(results: dict) -> tuple[set[str], set[str]]:
    """Discover (rules_dims, llm_dims) from result scenarios."""
    rules_dims: set[str] = set()
    llm_dims: set[str] = set()
    for s in results.get("scenarios", []):
        scores = s.get("scores", {})
        rules = scores.get("rules") or {}
        for check in rules.get("checks", []) or []:
            rules_dims.add(check.get("dimension", "unknown"))
        for _name, dim, _token in _iter_llm_dim_scores_from_scenario(s):
            llm_dims.add(dim)
    return rules_dims, llm_dims


def _get_llm_score(scenario: dict, dim: str) -> str | None:
    """Return the worst verdict for *dim* across every LLM-shaped payload."""
    rank = {"inconclusive": -1, "fail": 0, "low": 0, "medium": 1, "high": 2, "pass": 2}
    worst: str | None = None
    for _name, d, token in _iter_llm_dim_scores_from_scenario(scenario):
        if d != dim or token is None:
            continue
        if worst is None or rank.get(token, 1) < rank.get(worst, 1):
            worst = token
    return worst


def _get_rules_pass(scenario: dict, dim: str) -> bool | None:
    rules = scenario.get("scores", {}).get("rules") or {}
    for check in rules.get("checks", []) or []:
        if check.get("dimension") == dim:
            return check.get("passed", False)
    return None


def _leaf_name(key: str) -> str:
    """Extract the scenario name (last path component) from a scenario key."""
    return key.rsplit("/", 1)[-1]


def compare(results_a: dict, results_b: dict, label_a: str, label_b: str) -> dict:
    """Compare two result sets. Returns structured comparison data."""
    scenarios_a = {_scenario_key(s): s for s in results_a.get("scenarios", [])}
    scenarios_b = {_scenario_key(s): s for s in results_b.get("scenarios", [])}

    normalized = False
    shared = set(scenarios_a) & set(scenarios_b)
    if not shared and scenarios_a and scenarios_b:
        leaf_a = {_leaf_name(k): k for k in scenarios_a}
        leaf_b = {_leaf_name(k): k for k in scenarios_b}
        if set(leaf_a) & set(leaf_b):
            remapped_a = {_leaf_name(k): v for k, v in scenarios_a.items()}
            remapped_b = {_leaf_name(k): v for k, v in scenarios_b.items()}
            scenarios_a = remapped_a
            scenarios_b = remapped_b
            normalized = True

    all_keys = sorted(set(scenarios_a) | set(scenarios_b))

    _, llm_dims_a = _discover_dimensions(results_a)
    _, llm_dims_b = _discover_dimensions(results_b)
    shared_llm_dims = sorted(llm_dims_a & llm_dims_b)

    scenario_comparisons: list[dict] = []
    for key in all_keys:
        sa = scenarios_a.get(key)
        sb = scenarios_b.get(key)

        dim_deltas: list[dict] = []
        for dim in shared_llm_dims:
            score_a = _get_llm_score(sa, dim) if sa else None
            score_b = _get_llm_score(sb, dim) if sb else None
            rank_a = SCORE_RANK.get(score_a, -1) if score_a else -1
            rank_b = SCORE_RANK.get(score_b, -1) if score_b else -1
            delta = rank_b - rank_a if (rank_a >= 0 and rank_b >= 0) else None
            dim_deltas.append({"dimension": dim, "a": score_a, "b": score_b, "delta": delta})

        scenario_comparisons.append(
            {
                "key": key,
                "a_pass": sa.get("overall_pass") if sa else None,
                "b_pass": sb.get("overall_pass") if sb else None,
                "in_a": sa is not None,
                "in_b": sb is not None,
                "dimensions": dim_deltas,
            }
        )

    stats_a = results_a.get("stats", {})
    stats_b = results_b.get("stats", {})
    ct_a = results_a.get("cost_timing", {})
    ct_b = results_b.get("cost_timing", {})

    return {
        "label_a": label_a,
        "label_b": label_b,
        "total_a": results_a.get("total", 0),
        "total_b": results_b.get("total", 0),
        "pass_rate_a": stats_a.get("pass_rate", 0),
        "pass_rate_b": stats_b.get("pass_rate", 0),
        "cost_a": ct_a.get("total_cost_usd"),
        "cost_b": ct_b.get("total_cost_usd"),
        "mean_cost_a": ct_a.get("mean_cost_usd"),
        "mean_cost_b": ct_b.get("mean_cost_usd"),
        "time_a": ct_a.get("mean_seconds"),
        "time_b": ct_b.get("mean_seconds"),
        "shared_llm_dims": shared_llm_dims,
        "scenarios": scenario_comparisons,
        "keys_normalized": normalized,
    }


def print_terminal(comp: dict) -> None:
    """Print comparison as a terminal table."""
    la, lb = comp["label_a"], comp["label_b"]
    dims = comp["shared_llm_dims"]

    eprint("\n╭─ Comparison ─────────────────────────────────────────╮")
    eprint(f"│  A: {la:<50}│")
    eprint(f"│  B: {lb:<50}│")
    eprint(f"│  Shared LLM dimensions: {', '.join(dims) or '(none)':<30}│")
    if comp.get("keys_normalized"):
        eprint("│  ⚠️  Scenario keys normalized (different roots)      │")
    eprint("╰──────────────────────────────────────────────────────╯")

    pr_a = comp["pass_rate_a"]
    pr_b = comp["pass_rate_b"]
    pr_delta = pr_b - pr_a
    arrow = "↑" if pr_delta > 0 else "↓" if pr_delta < 0 else "="
    eprint(f"\n  Pass rate: A={pr_a:.0%}  B={pr_b:.0%}  ({arrow} {abs(pr_delta):.0%})")

    cost_a, cost_b = comp.get("cost_a"), comp.get("cost_b")
    if cost_a is not None or cost_b is not None:
        ca = f"${cost_a:.4f}" if cost_a is not None else "-"
        cb = f"${cost_b:.4f}" if cost_b is not None else "-"
        eprint(f"  Cost:      A={ca}  B={cb}")

    time_a, time_b = comp.get("time_a"), comp.get("time_b")
    if time_a is not None or time_b is not None:
        ta = f"{time_a:.1f}s" if time_a is not None else "-"
        tb = f"{time_b:.1f}s" if time_b is not None else "-"
        eprint(f"  Avg time:  A={ta}  B={tb}")

    if cost_a and cost_b and pr_a and pr_b:
        eff_a = pr_a / cost_a
        eff_b = pr_b / cost_b
        better = "B" if eff_b > eff_a else "A" if eff_a > eff_b else "="
        eprint(f"  Score/$:   A={eff_a:.0f}  B={eff_b:.0f}  (better: {better})")

    regressions = []
    improvements = []
    for sc in comp["scenarios"]:
        for dd in sc["dimensions"]:
            if dd["delta"] is not None and dd["delta"] < 0:
                regressions.append((sc["key"], dd["dimension"], dd["a"], dd["b"]))
            elif dd["delta"] is not None and dd["delta"] > 0:
                improvements.append((sc["key"], dd["dimension"], dd["a"], dd["b"]))

    if regressions:
        eprint(f"\n  ❌ Regressions ({len(regressions)}):")
        for key, dim, a, b in regressions:
            eprint(f"    {key} - {dim}: {a} → {b}")

    if improvements:
        eprint(f"\n  ✅ Improvements ({len(improvements)}):")
        for key, dim, a, b in improvements:
            eprint(f"    {key} - {dim}: {a} → {b}")

    only_a = [sc["key"] for sc in comp["scenarios"] if sc["in_a"] and not sc["in_b"]]
    only_b = [sc["key"] for sc in comp["scenarios"] if sc["in_b"] and not sc["in_a"]]
    if only_a:
        eprint(f"\n  Only in A: {', '.join(only_a)}")
    if only_b:
        eprint(f"\n  Only in B: {', '.join(only_b)}")

    shared_count = sum(1 for sc in comp["scenarios"] if sc["in_a"] and sc["in_b"])
    if shared_count == 0 and (only_a or only_b):
        eprint("\n  ⚠️  Zero shared scenarios. This usually means the two runs used")
        eprint("     different scenarios roots (e.g., examples/scenarios vs")
        eprint("     examples/scenarios/experience/tasktracker-claude). Re-run with the")
        eprint("     same root for a meaningful comparison.")

    if not regressions and not improvements:
        eprint("\n  No LLM score differences on shared dimensions.")


def build_markdown(comp: dict) -> str:
    """Build markdown report."""
    la, lb = comp["label_a"], comp["label_b"]
    dims = comp["shared_llm_dims"]
    lines: list[str] = []

    pr_a = comp["pass_rate_a"]
    pr_b = comp["pass_rate_b"]
    pr_delta = pr_b - pr_a
    icon = "✅" if pr_delta >= 0 else "❌"
    lines.append(f"## {icon} Cross-Agent Comparison")
    lines.append("")
    lines.append("| | **A** | **B** |")
    lines.append("|---|---|---|")
    lines.append(f"| Run | {la} | {lb} |")
    lines.append(f"| Scenarios | {comp['total_a']} | {comp['total_b']} |")
    lines.append(f"| Pass rate | {pr_a:.0%} | {pr_b:.0%} |")
    cost_a, cost_b = comp.get("cost_a"), comp.get("cost_b")
    if cost_a is not None or cost_b is not None:
        ca = f"${cost_a:.4f}" if cost_a is not None else "-"
        cb = f"${cost_b:.4f}" if cost_b is not None else "-"
        lines.append(f"| Total cost | {ca} | {cb} |")
    time_a, time_b = comp.get("time_a"), comp.get("time_b")
    if time_a is not None or time_b is not None:
        ta = f"{time_a:.1f}s" if time_a is not None else "-"
        tb = f"{time_b:.1f}s" if time_b is not None else "-"
        lines.append(f"| Avg time/scenario | {ta} | {tb} |")
    lines.append("")

    if dims:
        lines.append("### Per-Scenario LLM Dimensions")
        lines.append("")
        header = "| Scenario | " + " | ".join(f"{d} (A→B)" for d in dims) + " |"
        sep = "|---|" + "|".join("---" for _ in dims) + "|"
        lines.append(header)
        lines.append(sep)

        for sc in comp["scenarios"]:
            if not sc["in_a"] or not sc["in_b"]:
                continue
            cells = []
            for dd in sc["dimensions"]:
                a = dd["a"] or "-"
                b = dd["b"] or "-"
                if dd["delta"] is not None and dd["delta"] < 0:
                    cells.append(f"❌ {a}→{b}")
                elif dd["delta"] is not None and dd["delta"] > 0:
                    cells.append(f"✅ {a}→{b}")
                else:
                    cells.append(f"{a}→{b}")
            lines.append(f"| {sc['key']} | " + " | ".join(cells) + " |")
        lines.append("")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="belt compare", description="Compare two evaluation results")
    parser.add_argument("results_a", type=Path, help="First results.json")
    parser.add_argument("results_b", type=Path, help="Second results.json")
    parser.add_argument("--label-a", help="Label for first result set")
    parser.add_argument("--label-b", help="Label for second result set")
    parser.add_argument(
        "--output",
        choices=["terminal", "markdown", "json"],
        default="terminal",
        help="Output format (default: terminal)",
    )
    args = parser.parse_args(argv)

    try:
        ra = load_results(args.results_a)
        rb = load_results(args.results_b)
    except Exception as e:
        eprint(f"  ❌ {e}")
        return 1

    label_a = args.label_a or _extract_label(args.results_a)
    label_b = args.label_b or _extract_label(args.results_b)

    comp = compare(ra, rb, label_a, label_b)

    shared_count = sum(1 for sc in comp["scenarios"] if sc["in_a"] and sc["in_b"])
    if shared_count == 0 and comp["scenarios"]:
        eprint("\n  ❌ Zero shared scenarios between the two runs.")
        eprint("     This usually means different scenario roots or groups.")
        eprint(f"     A has {comp['total_a']} scenario(s), B has {comp['total_b']}.")
        eprint("     Re-run with the same scenario root for a meaningful comparison.")
        return 1

    if args.output == "terminal":
        print_terminal(comp)
    elif args.output == "markdown":
        # ``--output markdown`` / ``--output json`` are pipeable data
        # outputs: stdout is the contract (``belt compare ... --output
        # markdown > diff.md``). All other ``compare`` UI lives on
        # stderr via ``eprint``.
        print(build_markdown(comp))
    elif args.output == "json":
        print(json.dumps(comp, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
