"""
liveness.py — Live Range Computation (Core Algorithm)

THEORY — Two-Pass Dataflow Analysis:

    A register is LIVE at a program point if its value will be READ
    in the future. We compute, for each register:

        first_def  = line number where it is FIRST WRITTEN
        last_use   = line number where it is LAST READ

    This gives us the LIVE RANGE: [first_def, last_use]
    The register occupies a hardware register slot for every
    instruction in that range.

TWO PASSES:
    Pass 1 (Forward  — top to bottom):
        Record the FIRST time each register is written → first_def

    Pass 2 (Backward — bottom to top):
        Record the LAST  time each register is read   → last_use

WHY BACKWARD FOR LAST USE?
    Scanning forward would give us the first use, not the last.
    Scanning backward means the FIRST time we see a read is the
    LAST occurrence in program order.

EXAMPLE:
    Line 1:  mov.f32  %f0, 0f40000000    → %f0 born  (first_def=1)
    Line 2:  ld.global.f32 %f1, [%rd0]  → %f1 born  (first_def=2)
    Line 3:  add.f32  %f2, %f0, %f1     → %f0 read, %f1 read, %f2 born
    Line 4:  mul.f32  %f3, %f2, %f0     → %f2 read, %f0 read
    Line 5:  st.global.f32 [%rd0], %f3  → %f3 read  (last_use=5)

    Live ranges:
        %f0: [1, 4]   (alive lines 1-4)
        %f1: [2, 3]   (alive lines 2-3)
        %f2: [3, 4]   (alive lines 3-4)
        %f3: [4, 5]   (alive lines 4-5)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from .ir import Instruction, PTXNode


@dataclass
class LiveRange:
    """Live range of a single register."""
    name:       str        # register name e.g. "%f0"
    first_def:  int        # line number of first definition (write)
    last_use:   int        # line number of last use (read)

    @property
    def length(self) -> int:
        """Number of instructions this register is live for."""
        return self.last_use - self.first_def + 1

    def is_live_at(self, lineno: int) -> bool:
        """True if register is live at the given line number."""
        return self.first_def <= lineno <= self.last_use

    def __repr__(self) -> str:
        return (f"LiveRange({self.name}: "
                f"lines {self.first_def}–{self.last_use}, "
                f"length={self.length})")


# Maps register name → LiveRange
LiveRangeMap = Dict[str, LiveRange]


def compute_live_ranges(nodes: List[PTXNode]) -> LiveRangeMap:
    """
    Compute live ranges for all registers in the PTX node list.

    Args:
        nodes: parsed PTX node list (from parser.parse_source)

    Returns:
        dict mapping register name → LiveRange
    """
    instructions = [n for n in nodes if isinstance(n, Instruction)]

    # ── Pass 1: Forward scan — find first_def for each register ──────────
    first_def: Dict[str, int] = {}

    for instr in instructions:
        written = instr.written_reg()
        if written and written not in first_def:
            first_def[written] = instr.lineno

    # ── Pass 2: Backward scan — find last_use for each register ──────────
    last_use: Dict[str, int] = {}

    for instr in reversed(instructions):
        for reg in instr.all_read_regs():
            if reg not in last_use:
                last_use[reg] = instr.lineno

        # A register with no reads after its definition:
        # last_use = first_def (lives for exactly 1 line)
        written = instr.written_reg()
        if written and written not in last_use:
            last_use[written] = instr.lineno

    # ── Build LiveRange objects ───────────────────────────────────────────
    live_ranges: LiveRangeMap = {}

    all_regs = set(first_def.keys()) | set(last_use.keys())
    for reg in all_regs:
        fd = first_def.get(reg)
        lu = last_use.get(reg)
        if fd is None or lu is None:
            continue
        # Ensure first_def <= last_use (a register used before defined
        # can happen with predicate regs in loops — clamp conservatively)
        if lu < fd:
            lu = fd
        live_ranges[reg] = LiveRange(name=reg, first_def=fd, last_use=lu)

    return live_ranges


def get_live_registers_at(live_ranges: LiveRangeMap,
                           lineno: int) -> List[str]:
    """
    Return list of register names that are live at a given line number.

    Args:
        live_ranges: output of compute_live_ranges()
        lineno:      1-based line number

    Returns:
        sorted list of register names live at that line
    """
    return sorted(
        reg for reg, lr in live_ranges.items()
        if lr.is_live_at(lineno)
    )


def get_line_range(live_ranges: LiveRangeMap) -> Tuple[int, int]:
    """Return (min_line, max_line) across all live ranges."""
    if not live_ranges:
        return (1, 1)
    min_line = min(lr.first_def for lr in live_ranges.values())
    max_line = max(lr.last_use  for lr in live_ranges.values())
    return (min_line, max_line)


def sort_by_first_def(live_ranges: LiveRangeMap) -> List[LiveRange]:
    """Return live ranges sorted by their first definition line."""
    return sorted(live_ranges.values(), key=lambda lr: (lr.first_def, lr.name))
