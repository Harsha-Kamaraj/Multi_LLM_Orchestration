"""`report.html` — the human-facing half of R4's output.

`results.json` is the machine artifact and the thing the golden gate diffs.
This is what a person actually reads, and it is generated from exactly the same
payload so the two can never disagree.

Three design decisions, all driven by the project's own discipline:

**Every interval is drawn, not just printed.** The rule in CONTRIBUTING.md is
that no number ships bare. A table of digits satisfies that literally while
still letting a reader skim past the intervals — so each one is rendered as a
small bar against a zero line. Whether it crosses zero *is* the verdict, and
that should be visible at a glance rather than reconstructed from four decimal
places.

**The reference row is marked, not sorted to the top.** Every comparison is
against `verifier_gated_cascade`; burying that in a caption is how a reader
comes away thinking the policy was compared against the easy baselines.

**Warnings render above the numbers.** A blocked leakage audit or a 3% reward-hack
rate changes what every figure below it means, so it cannot sit in a footer.

Self-contained by construction: no CDN, no webfont, no script. The file opens
from a filesystem, survives being emailed, and renders the same in five years.
"""

from __future__ import annotations

import html
from typing import Any, Mapping, Sequence

from .report import Report

# --- design tokens ---------------------------------------------------------
# Cool-slate neutrals with a muted lab-teal accent. Semantic colours are a
# separate scale from the accent, so "significant" never competes with "brand".
_CSS = """
:root {
  --paper: #fbfbfa;
  --panel: #ffffff;
  --ink: #16191d;
  --ink-soft: #5b6167;
  --ink-faint: #878d93;
  --rule: #dcded9;
  --rule-soft: #ecedea;
  --accent: #2f6f6b;
  --accent-soft: #e4efed;
  --good: #2f6f4f;
  --warn: #8a6413;
  --warn-soft: #fbf3e0;
  --crit: #99342c;
  --crit-soft: #fbeae8;
  --zero: #b4b9bd;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper: #14171a;
    --panel: #1b1f23;
    --ink: #e7e9ea;
    --ink-soft: #a3aab0;
    --ink-faint: #7c848b;
    --rule: #2c3238;
    --rule-soft: #23282d;
    --accent: #6fbfb6;
    --accent-soft: #1e2f2e;
    --good: #6cbf8f;
    --warn: #d8ac52;
    --warn-soft: #2c2517;
    --crit: #e08078;
    --crit-soft: #2e1c1a;
    --zero: #4c5359;
  }
}
:root[data-theme="dark"] {
  --paper: #14171a;
  --panel: #1b1f23;
  --ink: #e7e9ea;
  --ink-soft: #a3aab0;
  --ink-faint: #7c848b;
  --rule: #2c3238;
  --rule-soft: #23282d;
  --accent: #6fbfb6;
  --accent-soft: #1e2f2e;
  --good: #6cbf8f;
  --warn: #d8ac52;
  --warn-soft: #2c2517;
  --crit: #e08078;
  --crit-soft: #2e1c1a;
  --zero: #4c5359;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto,
               "Helvetica Neue", Arial, sans-serif;
  font-size: 15px;
  line-height: 1.55;
}

.wrap {
  max-width: 1140px;
  margin: 0 auto;
  padding: 40px 24px 96px;
  display: flex;
  flex-direction: column;
  gap: 40px;
}

h1, h2 { text-wrap: balance; margin: 0; letter-spacing: -0.012em; }
h1 { font-size: 27px; font-weight: 620; }
h2 { font-size: 17px; font-weight: 600; }

.label {
  font-size: 11px;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  color: var(--ink-faint);
  font-weight: 600;
}

.meta {
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  font-size: 12.5px;
  color: var(--ink-soft);
  word-break: break-all;
}

section { display: flex; flex-direction: column; gap: 14px; }

.head {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding-bottom: 22px;
  border-bottom: 2px solid var(--ink);
}

.tiles {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(168px, 1fr));
  gap: 12px;
}
.tile {
  background: var(--panel);
  border: 1px solid var(--rule);
  padding: 14px 16px;
  display: flex;
  flex-direction: column;
  gap: 3px;
}
.tile .v {
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  font-size: 21px;
  font-variant-numeric: tabular-nums;
  font-weight: 600;
}
.tile .n { font-size: 12px; color: var(--ink-soft); }

.scroll { overflow-x: auto; border: 1px solid var(--rule); background: var(--panel); }

table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
th, td {
  padding: 9px 13px;
  text-align: right;
  white-space: nowrap;
  border-bottom: 1px solid var(--rule-soft);
}
th {
  font-size: 11px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--ink-faint);
  font-weight: 600;
  border-bottom: 1px solid var(--rule);
  position: sticky;
  top: 0;
  background: var(--panel);
}
th:first-child, td:first-child { text-align: left; }
tbody tr:last-child td { border-bottom: none; }
td.num, td.name {
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  font-variant-numeric: tabular-nums;
}
tr.ref { background: var(--accent-soft); }
tr.ref td:first-child::after {
  content: " reference";
  color: var(--accent);
  font-size: 10.5px;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  font-family: ui-sans-serif, system-ui, sans-serif;
}

.sig { color: var(--good); font-weight: 600; }
.nul { color: var(--ink-faint); }

.notes { display: flex; flex-direction: column; gap: 8px; }
.note {
  display: flex;
  gap: 10px;
  padding: 10px 14px;
  border-left: 3px solid var(--warn);
  background: var(--warn-soft);
  font-size: 13.5px;
}
.note.block { border-left-color: var(--crit); background: var(--crit-soft); }

.bars { display: flex; flex-direction: column; gap: 2px; }
.barrow { display: grid; grid-template-columns: 1fr auto; gap: 12px; align-items: center; }

p { margin: 0; max-width: 68ch; color: var(--ink-soft); font-size: 13.5px; }
code {
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  font-size: 0.92em;
}
footer { color: var(--ink-faint); font-size: 12px; border-top: 1px solid var(--rule);
         padding-top: 16px; }
"""


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def _num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _interval_bar(low: float, high: float, point: float, scale: float) -> str:
    """One interval drawn against a zero line.

    Whether the bar clears zero is the verdict, so it is a mark rather than
    four decimal places a reader can skim past. `scale` is shared across every
    row so the bars are comparable — rescaling each row independently would
    make a tiny effect look like a large one.
    """
    width, height = 132, 18
    mid = width / 2

    def x(value: float) -> float:
        return max(1.0, min(width - 1.0, mid + (value / scale) * (mid - 3)))

    x_low, x_high, x_point = x(low), x(high), x(point)
    crosses = low <= 0.0 <= high
    colour = "var(--zero)" if crosses else "var(--good)"
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="interval {low:+.4f} to {high:+.4f}">'
        f'<line x1="{mid}" y1="2" x2="{mid}" y2="{height - 2}" '
        f'stroke="var(--zero)" stroke-width="1"/>'
        f'<line x1="{x_low:.1f}" y1="{height / 2}" x2="{x_high:.1f}" y2="{height / 2}" '
        f'stroke="{colour}" stroke-width="2.5" stroke-linecap="round"/>'
        f'<circle cx="{x_point:.1f}" cy="{height / 2}" r="3" fill="{colour}"/>'
        f"</svg>"
    )


def _tiles(payload: Mapping[str, Any]) -> str:
    headroom = payload.get("headroom", {})
    taxonomy = payload.get("failure_taxonomy", {})
    solved = (taxonomy.get("rates") or {}).get("solved")
    tiles = [
        ("Tasks", str(payload.get("n_tasks", "—")), f"{payload.get('n_seeds', '—')} seeds per arm"),
        ("Solved", _pct(solved) if solved is not None else "—", "all hidden tests pass"),
        ("Routing headroom", _pct(headroom.get("escalation_helps")),
         "tasks only the large arm solves"),
        ("Unsolvable", _pct(headroom.get("neither")), "no arm solves — not a routing failure"),
    ]
    cells = "".join(
        f'<div class="tile"><span class="label">{_esc(label)}</span>'
        f'<span class="v">{_esc(value)}</span>'
        f'<span class="n">{_esc(note)}</span></div>'
        for label, value, note in tiles
    )
    return f'<div class="tiles">{cells}</div>'


def _comparison_table(payload: Mapping[str, Any]) -> str:
    reference = payload.get("reference_policy", "")
    policies: Mapping[str, Any] = payload.get("policies", {})
    comparisons: Mapping[str, Any] = payload.get("comparisons", {})

    bounds = [
        abs(v)
        for cmp in comparisons.values()
        for v in (cmp["accuracy_difference"]["low"], cmp["accuracy_difference"]["high"])
    ]
    scale = max(bounds) if bounds else 1.0

    rows = []
    for name in sorted(policies):
        summary = policies[name]
        is_ref = name == reference
        if is_ref:
            delta_cell = '<td class="num nul">—</td><td></td><td class="num nul">—</td>'
        else:
            ci = comparisons[name]["accuracy_difference"]
            mc = comparisons[name]["mcnemar"]
            cls = "sig" if ci["excludes_zero"] else "nul"
            p_adj = mc.get("p_adjusted", mc.get("p_value"))
            delta_cell = (
                f'<td class="num {cls}">{ci["point"]:+.4f}<br>'
                f'<span class="nul" style="font-size:11.5px">'
                f'[{ci["low"]:+.4f}, {ci["high"]:+.4f}]</span></td>'
                f'<td>{_interval_bar(ci["low"], ci["high"], ci["point"], scale)}</td>'
                f'<td class="num {cls}">{p_adj:.4g}</td>'
            )
        rows.append(
            f'<tr class="{"ref" if is_ref else ""}">'
            f'<td class="name">{_esc(name)}</td>'
            f'<td class="num">{_pct(summary.get("accuracy"), 2)}</td>'
            f'<td class="num">{_num(summary.get("cost"))}</td>'
            f'<td class="num">{_num(summary.get("latency_p95"))}</td>'
            f"{delta_cell}</tr>"
        )

    return (
        '<div class="scroll"><table><thead><tr>'
        "<th>Policy</th><th>Accuracy</th><th>Cost</th><th>p95 latency</th>"
        "<th>&Delta; accuracy vs reference</th><th></th><th>p (BH)</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _taxonomy_table(payload: Mapping[str, Any]) -> str:
    taxonomy = payload.get("failure_taxonomy") or {}
    counts: Mapping[str, int] = taxonomy.get("counts", {})
    by_arm: Mapping[str, Mapping[str, int]] = taxonomy.get("by_arm", {})
    meaning: Mapping[str, str] = taxonomy.get("meaning", {})
    if not counts:
        return ""

    arms = sorted(by_arm)
    header = "".join(f"<th>{_esc(a)}</th>" for a in arms)
    rows = []
    for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        if not count:
            continue
        cells = ""
        for arm in arms:
            total = sum(by_arm[arm].values()) or 1
            cells += f'<td class="num">{_pct(by_arm[arm].get(name, 0) / total)}</td>'
        rows.append(
            f'<tr><td class="name">{_esc(name)}</td>'
            f'<td class="num">{count}</td>'
            f'<td class="num">{_pct(taxonomy["rates"].get(name))}</td>'
            f"{cells}"
            f'<td style="text-align:left;white-space:normal;color:var(--ink-soft)">'
            f"{_esc(meaning.get(name, ''))}</td></tr>"
        )
    return (
        '<div class="scroll"><table><thead><tr>'
        f"<th>Category</th><th>n</th><th>Rate</th>{header}<th>Means</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _notes(payload: Mapping[str, Any], warnings: Sequence[str]) -> str:
    audit = payload.get("leakage_audit") or {}
    items = []
    if audit.get("blocked"):
        items.append(
            '<div class="note block"><strong>Leakage audit blocked.</strong>'
            " This report is not publishable until the findings below are resolved.</div>"
        )
    for warning in warnings:
        blocked = "BLOCKED" in warning
        items.append(
            f'<div class="note{" block" if blocked else ""}">{_esc(warning)}</div>'
        )
    return f'<div class="notes">{"".join(items)}</div>' if items else ""


def render(report: Report, *, title: str | None = None) -> str:
    """Render a `Report` to a self-contained HTML document."""
    payload = report.payload
    run_id = payload.get("run_id", "unknown")
    name = title or f"Evaluation — {run_id}"
    boot = payload.get("bootstrap", {})
    attribution = payload.get("gap_attribution") or []

    attribution_html = ""
    if attribution:
        items = "".join(f"<li>{_esc(line)}</li>" for line in attribution)
        attribution_html = (
            "<section><h2>What the gap is made of</h2>"
            "<p>Accuracy says how often the system is wrong. This says how — which "
            "is what changes the next decision. A gap made of timeouts is a serving "
            "problem; the same gap made of wrong answers is a capability problem.</p>"
            f'<ul style="margin:0;padding-left:20px;color:var(--ink-soft);'
            f'font-size:13.5px">{items}</ul></section>'
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(name)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">

<header class="head">
  <span class="label">Cost-aware LLM orchestration — evaluation</span>
  <h1>{_esc(name)}</h1>
  <span class="meta">run {_esc(run_id)} &middot;
    splits {_esc(", ".join(payload.get("splits", [])))} &middot;
    arms {_esc(", ".join(payload.get("arms", [])))}</span>
</header>

{_notes(payload, report.warnings)}

<section>
  <h2>At a glance</h2>
  {_tiles(payload)}
</section>

<section>
  <h2>Every policy, against the reference</h2>
  <p>The reference is <code>{_esc(payload.get("reference_policy", ""))}</code> — run
  the small model, execute the visible tests, escalate on failure. It is the
  baseline the claim has to survive, because observing failure beats predicting
  it. Each interval is a paired cluster bootstrap over tasks and seeds; a bar
  that clears the zero line supports a directional claim.</p>
  {_comparison_table(payload)}
</section>

{attribution_html}

<section>
  <h2>Why generations failed</h2>
  {_taxonomy_table(payload)}
</section>

<footer>
  {boot.get("n_resamples", "—")} bootstrap resamples, seed {boot.get("seed", "—")},
  {int(float(boot.get("level", 0.95)) * 100)}% intervals &middot;
  {_esc((payload.get("multiplicity") or {}).get("method", "—"))} at q={
    (payload.get("multiplicity") or {}).get("q", "—")} &middot;
  generated from results.json — the same payload the golden-run gate diffs
</footer>

</div>
</body>
</html>
"""
