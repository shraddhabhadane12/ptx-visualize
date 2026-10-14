"""
visualizer.py — ASCII Chart Renderer

Renders three visualizations:
  1. LIVE RANGE CHART    — timeline bar per register showing birth/death
  2. PRESSURE GRAPH      — bar chart of simultaneous live registers over time
  3. SPILL REPORT        — table of spill store/reload events

All output is plain ASCII — works in any terminal, no dependencies.
"""

from typing import List, Optional, Dict
from .liveness  import LiveRangeMap, LiveRange, sort_by_first_def
from .pressure  import PressureProfile, pressure_graph
from .spill_detector import SpillReport
from .ir        import Instruction, PTXNode

WIDTH = 70   # total report box width


# ── Box drawing helpers ───────────────────────────────────────────────────────

def _top()     -> str: return "╔" + "═" * (WIDTH - 2) + "╗"
def _bottom()  -> str: return "╚" + "═" * (WIDTH - 2) + "╝"
def _divider() -> str: return "╠" + "═" * (WIDTH - 2) + "╣"
def _line(text: str = "") -> str:
    return "║  " + text[:WIDTH - 4].ljust(WIDTH - 4) + "║"
def _header(title: str) -> str:
    pad = (WIDTH - 2 - len(title)) // 2
    return "║" + " " * pad + title + " " * (WIDTH - 2 - pad - len(title)) + "║"


# ── Live Range Chart ──────────────────────────────────────────────────────────

def render_live_range_chart(
    live_ranges:  LiveRangeMap,
    instructions: List[Instruction],
    spill_report: SpillReport,
    max_cols:     int = 40,
) -> str:
    """
    Render an ASCII timeline showing when each register is alive.

    ██ = register is live at this instruction
    ░░ = register is not live
    ↓S = spill store at this line
    ↑R = spill reload at this line

    Args:
        live_ranges:  from liveness.compute_live_ranges()
        instructions: list of Instruction objects (in order)
        spill_report: from spill_detector.detect_spills()
        max_cols:     max number of instruction columns to display

    Returns:
        multi-line string
    """
    if not live_ranges or not instructions:
        return "  (no registers or instructions found)\n"

    sorted_ranges = sort_by_first_def(live_ranges)
    linenos = [i.lineno for i in instructions]

    # Sample columns if too many instructions
    if len(linenos) > max_cols:
        step    = len(linenos) / max_cols
        col_nos = [linenos[int(i * step)] for i in range(max_cols)]
    else:
        col_nos = linenos

    # Build spill marker sets
    spill_store_lines  = {e.lineno for e in spill_report.stores}
    spill_reload_lines = {e.lineno for e in spill_report.reloads}

    REG_COL = 10   # width of register name column
    lines   = []

    # Header row: column indices (every 5th)
    hdr = " " * (REG_COL + 3)
    for i, ln in enumerate(col_nos):
        if i % 5 == 0:
            hdr += str(ln % 100).ljust(2)
        else:
            hdr += "  "
    lines.append(hdr)
    lines.append(" " * (REG_COL + 3) + "─" * (len(col_nos) * 2))

    # One row per register
    for lr in sorted_ranges:
        name_col = lr.name[:REG_COL].ljust(REG_COL)
        bar = ""
        for ln in col_nos:
            if ln in spill_store_lines and lr.is_live_at(ln):
                bar += "↓S"
            elif ln in spill_reload_lines and lr.is_live_at(ln):
                bar += "↑R"
            elif lr.is_live_at(ln):
                bar += "██"
            else:
                bar += "░░"
        lines.append(f"  {name_col} │{bar}")

    # Pressure row
    lines.append(" " * (REG_COL + 3) + "─" * (len(col_nos) * 2))
    pres_row = "  " + "PRESSURE".ljust(REG_COL) + " │"
    for ln in col_nos:
        live_count = sum(1 for lr in sorted_ranges if lr.is_live_at(ln))
        pres_row += str(live_count % 10) + " "
    lines.append(pres_row)

    return "\n".join(lines)


# ── Full Report ───────────────────────────────────────────────────────────────

def render_full_report(
    filename:     str,
    live_ranges:  LiveRangeMap,
    profile:      PressureProfile,
    spill_report: SpillReport,
    instructions: List[Instruction],
    verbose:      bool = False,
) -> str:
    """
    Render the complete register visualizer report.

    Sections:
      1. Summary statistics
      2. Live range chart
      3. Pressure graph
      4. Spill report
    """
    out = []

    # ── Header ────────────────────────────────────────────────────────────
    out.append(_top())
    out.append(_header("PTX REGISTER VISUALIZER"))
    out.append(_header(f"File: {filename}"))
    out.append(_divider())

    # ── Section 1: Summary ────────────────────────────────────────────────
    out.append(_line("SUMMARY"))
    out.append(_line("─" * (WIDTH - 6)))
    out.append(_line(f"  Total instructions     : {len(instructions)}"))
    out.append(_line(f"  Unique registers       : {len(live_ranges)}"))
    out.append(_line(f"  Peak register pressure : {profile.peak_pressure}"
                     f"  ({profile.pressure_level})"))
    out.append(_line(f"  Avg register pressure  : {profile.avg_pressure:.1f}"))
    out.append(_line(f"  Peak at lines          : "
                     f"{profile.peak_lines[:5]}"))
    out.append(_line(f"  Estimated occupancy    : {profile.estimated_occupancy}"))
    out.append(_line(f"  Spill operations       : {spill_report.total_spill_ops}"
                     f"  ({spill_report.summary[:40]})"))

    # ── Section 2: Live Range Chart ───────────────────────────────────────
    out.append(_divider())
    out.append(_header("LIVE RANGE CHART"))
    out.append(_line("  ██ = live   ░░ = dead   ↓S = spill store   ↑R = spill reload"))
    out.append(_line(""))

    chart = render_live_range_chart(
        live_ranges, instructions, spill_report, max_cols=35
    )
    for row in chart.splitlines():
        out.append("  " + row)

    # ── Section 3: Pressure Graph ─────────────────────────────────────────
    out.append("")
    out.append(_divider())
    out.append(_header("REGISTER PRESSURE OVER TIME"))
    out.append(_line("  Each column = one instruction. Height = #live registers."))
    out.append(_line(""))

    graph = pressure_graph(profile, width=50, height=8)
    for row in graph.splitlines():
        out.append("  " + row)

    # ── Section 4: Spill Report ───────────────────────────────────────────
    out.append("")
    out.append(_divider())
    out.append(_header("SPILL DETECTION REPORT"))
    out.append(_line(""))

    if not spill_report.has_spills:
        out.append(_line("  ✓  No register spills detected."))
        out.append(_line("     All values kept in registers throughout execution."))
    else:
        out.append(_line(f"  Spill stores  : {len(spill_report.stores)}"))
        out.append(_line(f"  Spill reloads : {len(spill_report.reloads)}"))
        out.append(_line(f"  Matched pairs : {len(spill_report.pairs)}"))
        out.append(_line(""))

        if spill_report.stores:
            out.append(_line("  SPILL STORES (register → memory):"))
            for ev in spill_report.stores[:10]:
                out.append(_line(f"    Line {ev.lineno:4d}: {ev.raw[:50]}"))

        if spill_report.reloads:
            out.append(_line(""))
            out.append(_line("  SPILL RELOADS (memory → register):"))
            for ev in spill_report.reloads[:10]:
                out.append(_line(f"    Line {ev.lineno:4d}: {ev.raw[:50]}"))

        if spill_report.pairs:
            out.append(_line(""))
            out.append(_line("  MATCHED PAIRS (spill distance = lines between store & reload):"))
            for pair in spill_report.pairs[:8]:
                out.append(_line(
                    f"    {pair.store.reg:<8} "
                    f"store@{pair.store.lineno} → reload@{pair.reload.lineno}"
                    f"  (dist={pair.spill_distance})"
                ))

    # ── Section 5: Top registers by live range length (verbose) ──────────
    if verbose and live_ranges:
        out.append("")
        out.append(_divider())
        out.append(_header("TOP 10 LONGEST LIVE RANGES"))
        out.append(_line(""))
        sorted_lr = sorted(live_ranges.values(),
                           key=lambda r: r.length, reverse=True)
        for lr in sorted_lr[:10]:
            bar = "█" * min(lr.length, 30)
            out.append(_line(
                f"  {lr.name:<8} lines {lr.first_def:3d}–{lr.last_use:3d}"
                f"  len={lr.length:3d}  {bar}"
            ))

    out.append(_bottom())
    return "\n".join(out)
