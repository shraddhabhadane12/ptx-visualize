"""
pressure.py — Register Pressure Analysis

THEORY:
    Register PRESSURE at a program point = number of registers
    simultaneously live at that point.

    High pressure → many registers needed at once → risk of spilling.

    GPU CONTEXT:
        Each SM has 65,536 registers (Ampere architecture).
        These are shared across all active threads.

        If pressure = 32 per thread and block = 256 threads:
            registers needed = 32 × 256 = 8,192 per block
            blocks per SM    = 65,536 / 8,192 = 8 blocks
            occupancy        = 8 × 256 / 2048 = 100%  ← good

        If pressure spikes to 64:
            registers needed = 64 × 256 = 16,384 per block
            blocks per SM    = 65,536 / 16,384 = 4 blocks
            occupancy        = 4 × 256 / 2048 = 50%   ← half speed!

PRESSURE LEVELS (per thread):
    0–23   LOW      — excellent occupancy
    24–31  MEDIUM   — good occupancy
    32–47  HIGH     — reduced occupancy
    48+    CRITICAL — severe occupancy impact, likely spilling
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from .liveness import LiveRangeMap, get_live_registers_at, get_line_range


@dataclass
class PressurePoint:
    """Pressure at a single instruction line."""
    lineno:   int
    pressure: int           # number of live registers
    live_regs: List[str]    # which registers are live


@dataclass
class PressureProfile:
    """Complete register pressure profile across all instructions."""
    points:       List[PressurePoint] = field(default_factory=list)
    peak_pressure: int = 0
    peak_lines:   List[int] = field(default_factory=list)
    avg_pressure: float = 0.0

    @property
    def pressure_level(self) -> str:
        p = self.peak_pressure
        if p == 0:  return "NONE"
        if p < 24:  return "LOW"
        if p < 32:  return "MEDIUM"
        if p < 48:  return "HIGH"
        return "CRITICAL"

    @property
    def estimated_occupancy(self) -> str:
        """
        Rough occupancy estimate based on peak register pressure.
        Assumes 256-thread blocks on Ampere (65536 regs/SM).
        """
        p = self.peak_pressure
        if p == 0:
            return "N/A"
        regs_per_block = p * 256
        # Round up to nearest 256
        regs_per_block = ((regs_per_block + 255) // 256) * 256
        blocks_per_sm  = min(65536 // max(regs_per_block, 1), 8)
        threads_per_sm = blocks_per_sm * 256
        pct            = threads_per_sm / 2048 * 100
        return f"{pct:.0f}%  ({threads_per_sm} threads/SM)"

    def pressure_at(self, lineno: int) -> int:
        for p in self.points:
            if p.lineno == lineno:
                return p.pressure
        return 0

    def pressure_bar(self, lineno: int, width: int = 20) -> str:
        """ASCII bar proportional to pressure at this line."""
        p   = self.pressure_at(lineno)
        pct = p / max(self.peak_pressure, 1)
        filled = int(pct * width)
        return '█' * filled + '░' * (width - filled)


def compute_pressure(live_ranges: LiveRangeMap,
                     instruction_linenos: List[int]) -> PressureProfile:
    """
    Compute register pressure at every instruction line.

    Args:
        live_ranges:         output of liveness.compute_live_ranges()
        instruction_linenos: sorted list of line numbers that have instructions

    Returns:
        PressureProfile with per-line pressure data
    """
    if not instruction_linenos or not live_ranges:
        return PressureProfile()

    profile = PressureProfile()

    for lineno in instruction_linenos:
        live = get_live_registers_at(live_ranges, lineno)
        p    = len(live)
        profile.points.append(PressurePoint(
            lineno=lineno, pressure=p, live_regs=live
        ))
        if p > profile.peak_pressure:
            profile.peak_pressure = p
            profile.peak_lines    = [lineno]
        elif p == profile.peak_pressure and p > 0:
            profile.peak_lines.append(lineno)

    if profile.points:
        profile.avg_pressure = (
            sum(pt.pressure for pt in profile.points) / len(profile.points)
        )

    return profile


def pressure_graph(profile: PressureProfile,
                   width: int = 50,
                   height: int = 8) -> str:
    """
    Render a vertical ASCII bar chart of register pressure over time.

    Args:
        profile: PressureProfile from compute_pressure()
        width:   max horizontal width in characters
        height:  number of rows in the chart

    Returns:
        multi-line string containing the chart
    """
    if not profile.points:
        return "  (no instructions found)"

    pressures = [pt.pressure for pt in profile.points]
    max_p     = max(pressures) if pressures else 1
    n         = len(pressures)

    # Sample down if too many instructions for display width
    if n > width:
        step   = n / width
        sampled = [pressures[int(i * step)] for i in range(width)]
    else:
        sampled = pressures

    lines = []
    # Y-axis labels + bars
    for row in range(height, 0, -1):
        threshold = row / height * max_p
        bar_line  = ""
        for p in sampled:
            bar_line += "█" if p >= threshold else " "
        y_label = f"{int(threshold):2d} │"
        lines.append(f"  {y_label}{bar_line}")

    # X-axis
    lines.append("   └" + "─" * len(sampled))
    lines.append(f"     instruction flow →  "
                 f"(peak={max_p}, avg={profile.avg_pressure:.1f})")
    return "\n".join(lines)
