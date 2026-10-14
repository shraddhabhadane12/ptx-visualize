"""
main.py — CLI Entry Point for PTX Register Visualizer

Usage:
    python -m src.main samples/ptx_files/vector_add.ptx
    python -m src.main samples/ptx_files/vector_add.ptx --verbose
    python -m src.main samples/ptx_files/vector_add.ptx --only live
    python -m src.main samples/ptx_files/vector_add.ptx --only pressure
    python -m src.main samples/ptx_files/vector_add.ptx --only spills
    python -m src.main samples/ptx_files/vector_add.ptx --regs %f0,%f1
    python -m src.main samples/ptx_files/             (all PTX in directory)
"""

import argparse
import io
import json
import sys
from pathlib import Path
from typing import List, Optional

# Fix Windows terminal encoding for box-drawing characters
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                  errors='replace')

from .parser        import parse_source, instructions_only
from .liveness      import compute_live_ranges, sort_by_first_def
from .pressure      import compute_pressure, pressure_graph
from .spill_detector import detect_spills
from .visualizer    import render_full_report, render_live_range_chart


def run_analysis(source: str, filename: str,
                 verbose: bool = False,
                 only: Optional[str] = None,
                 filter_regs: Optional[List[str]] = None,
                 as_json: bool = False) -> dict:
    """
    Run the full register visualization pipeline.

    Args:
        source:      PTX file contents
        filename:    display name
        verbose:     show extra detail
        only:        'live', 'pressure', or 'spills'
        filter_regs: only show these registers in chart
        as_json:     output JSON instead of ASCII

    Returns:
        dict with all analysis results
    """
    nodes        = parse_source(source)
    instrs       = instructions_only(nodes)
    linenos      = [i.lineno for i in instrs]

    live_ranges  = compute_live_ranges(nodes)
    profile      = compute_pressure(live_ranges, linenos)
    spill_report = detect_spills(nodes)

    # Filter live ranges if --regs specified
    display_ranges = live_ranges
    if filter_regs:
        display_ranges = {k: v for k, v in live_ranges.items()
                         if k in filter_regs}

    if as_json:
        data = {
            "file": filename,
            "total_instructions": len(instrs),
            "unique_registers":   len(live_ranges),
            "peak_pressure":      profile.peak_pressure,
            "avg_pressure":       round(profile.avg_pressure, 2),
            "peak_at_lines":      profile.peak_lines,
            "pressure_level":     profile.pressure_level,
            "estimated_occupancy":profile.estimated_occupancy,
            "spill_stores":       len(spill_report.stores),
            "spill_reloads":      len(spill_report.reloads),
            "live_ranges": {
                reg: {"first_def": lr.first_def,
                      "last_use":  lr.last_use,
                      "length":    lr.length}
                for reg, lr in live_ranges.items()
            },
        }
        print(json.dumps(data, indent=2))
        return data

    # Selective output
    if only == 'live':
        print(f"\nLIVE RANGE CHART — {filename}\n")
        chart = render_live_range_chart(
            display_ranges, instrs, spill_report, max_cols=40)
        print(chart)

    elif only == 'pressure':
        print(f"\nREGISTER PRESSURE GRAPH — {filename}")
        print(f"Peak: {profile.peak_pressure}  "
              f"Avg: {profile.avg_pressure:.1f}  "
              f"Level: {profile.pressure_level}\n")
        print(pressure_graph(profile, width=55, height=10))

    elif only == 'spills':
        print(f"\nSPILL REPORT — {filename}")
        print(spill_report.summary)
        for ev in spill_report.stores + spill_report.reloads:
            print(f"  Line {ev.lineno:4d} [{ev.label}]  {ev.raw}")

    else:
        # Full report
        report = render_full_report(
            filename, display_ranges, profile, spill_report,
            instrs, verbose=verbose
        )
        print(report)

    return {
        "live_ranges":  live_ranges,
        "profile":      profile,
        "spill_report": spill_report,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='ptx-visualizer',
        description='PTX Register Visualizer — live ranges, pressure, spills',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main samples/ptx_files/vector_add.ptx
  python -m src.main samples/ptx_files/vector_add.ptx --verbose
  python -m src.main samples/ptx_files/vector_add.ptx --only pressure
  python -m src.main samples/ptx_files/vector_add.ptx --only live
  python -m src.main samples/ptx_files/vector_add.ptx --only spills
  python -m src.main samples/ptx_files/vector_add.ptx --regs %f0,%f1,%rd0
  python -m src.main samples/ptx_files/vector_add.ptx --json
  python -m src.main samples/ptx_files/        (all PTX files in dir)
        """
    )
    p.add_argument('inputs', nargs='+',
                   help='PTX file(s) or directory')
    p.add_argument('-v', '--verbose', action='store_true',
                   help='Show top-10 longest live ranges')
    p.add_argument('--only', choices=['live', 'pressure', 'spills'],
                   help='Show only one section')
    p.add_argument('--regs',
                   help='Comma-separated list of registers to show e.g. %%f0,%%rd1')
    p.add_argument('--json', action='store_true',
                   help='Output as JSON')
    return p


def main(argv=None) -> int:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    # Expand directories
    paths = []
    for inp in args.inputs:
        p = Path(inp)
        if p.is_dir():
            paths.extend(sorted(p.glob('*.ptx')))
        elif p.exists():
            paths.append(p)
        else:
            print(f"File not found: {p}", file=sys.stderr)
            return 1

    if not paths:
        print("No PTX files found.", file=sys.stderr)
        return 1

    filter_regs = None
    if args.regs:
        filter_regs = [r.strip() for r in args.regs.split(',')]

    for path in paths:
        source = path.read_text(encoding='utf-8')
        run_analysis(
            source       = source,
            filename     = path.name,
            verbose      = args.verbose,
            only         = args.only,
            filter_regs  = filter_regs,
            as_json      = args.json,
        )
        if len(paths) > 1:
            print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
