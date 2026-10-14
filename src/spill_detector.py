"""
spill_detector.py — Register Spill Detection

THEORY:
    When the compiler runs out of physical registers (register pressure
    exceeds hardware limit), it must SPILL some register values to memory
    and reload them later.

    In PTX, spills appear as:
        st.local.*  [%rd_spill+N], %fX   ← SPILL STORE  (save to memory)
        ld.local.*  %fX, [%rd_spill+N]   ← SPILL RELOAD (load from memory)

    .local memory is per-thread stack memory — much slower than registers
    (~200 cycles vs ~1 cycle). Every spill/reload pair adds latency.

WHY SPILLS HAPPEN:
    If a kernel uses more live registers at one point than the GPU has
    available (typically 255 per thread max in PTX), the compiler:
    1. Chooses a register to evict
    2. Inserts st.local to save its value
    3. Frees that register for other uses
    4. Inserts ld.local to restore the value when needed again

HOW TO FIX SPILLS:
    - Reduce register pressure (reorder instructions, shorten live ranges)
    - Use __launch_bounds__ in CUDA to tell compiler max threads/block
    - Split the kernel into smaller pieces
    - Use shared memory instead of registers for some values

DETECTION STRATEGY:
    1. Find all st.local instructions → spill stores
    2. Find all ld.local instructions → spill reloads
    3. Try to match stores to reloads by memory address offset
    4. Report unmatched stores/reloads as orphaned spills
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from .ir import Instruction, PTXNode


@dataclass
class SpillEvent:
    """A single spill store or reload event."""
    kind:    str           # "STORE" or "RELOAD"
    lineno:  int           # line number in PTX file
    reg:     str           # register being spilled/reloaded
    offset:  Optional[int] # memory offset in .local space, if parseable
    raw:     str           # original PTX instruction text

    @property
    def label(self) -> str:
        return "↓SPILL" if self.kind == "STORE" else "↑RELOAD"


@dataclass
class SpillPair:
    """A matched spill store + reload for the same register/offset."""
    store:  SpillEvent
    reload: SpillEvent

    @property
    def spill_distance(self) -> int:
        """Number of instructions between spill and reload."""
        return self.reload.lineno - self.store.lineno


@dataclass
class SpillReport:
    """Complete spill analysis for one PTX kernel."""
    stores:  List[SpillEvent] = field(default_factory=list)
    reloads: List[SpillEvent] = field(default_factory=list)
    pairs:   List[SpillPair]  = field(default_factory=list)

    @property
    def total_spill_ops(self) -> int:
        return len(self.stores) + len(self.reloads)

    @property
    def has_spills(self) -> bool:
        return self.total_spill_ops > 0

    @property
    def spill_overhead_pct(self, total_instructions: int = 0) -> str:
        if total_instructions == 0:
            return "N/A"
        pct = self.total_spill_ops / total_instructions * 100
        return f"{pct:.1f}%"

    @property
    def summary(self) -> str:
        if not self.has_spills:
            return "No register spills detected — all values kept in registers."
        return (f"{len(self.stores)} spill store(s), "
                f"{len(self.reloads)} spill reload(s), "
                f"{len(self.pairs)} matched pair(s).")


# ── Regex helpers ─────────────────────────────────────────────────────────────

_RE_LOCAL_OFFSET = re.compile(r'\[%\w+\+?(\d*)\]')
_RE_REG          = re.compile(r'%\w+')


def _extract_offset(operand: str) -> Optional[int]:
    """Extract numeric offset from [%rd+N] operand."""
    m = _RE_LOCAL_OFFSET.search(operand)
    if m:
        val = m.group(1)
        return int(val) if val else 0
    return None


def _extract_reg(operands: List[str], exclude_addr: bool = False) -> str:
    """Extract the data register from operands (not the address register)."""
    for op in operands:
        if op.startswith('%') and '[' not in op:
            return op
    return ""


# ── Main detection function ───────────────────────────────────────────────────

def detect_spills(nodes: List[PTXNode]) -> SpillReport:
    """
    Scan PTX nodes for register spill patterns.

    Detects:
      - st.local.* instructions → spill stores
      - ld.local.* instructions → spill reloads
      - Matches stores to reloads by register name

    Args:
        nodes: parsed PTX node list

    Returns:
        SpillReport with all detected spill events and matched pairs
    """
    report = SpillReport()

    for node in nodes:
        if not isinstance(node, Instruction):
            continue

        op = node.opcode.lower()

        # ── Spill Store: st.local.* [addr], %reg ─────────────────────────
        if op.startswith('st.local'):
            # In our IR, dest = address operand, operands = [data_reg]
            addr_op = node.dest or ""
            data_reg = node.operands[0] if node.operands else ""
            offset   = _extract_offset(addr_op)

            report.stores.append(SpillEvent(
                kind   = "STORE",
                lineno = node.lineno,
                reg    = data_reg,
                offset = offset,
                raw    = node.raw.strip(),
            ))

        # ── Spill Reload: ld.local.* %reg, [addr] ────────────────────────
        elif op.startswith('ld.local'):
            data_reg = node.dest or ""
            addr_op  = node.operands[0] if node.operands else ""
            offset   = _extract_offset(addr_op)

            report.reloads.append(SpillEvent(
                kind   = "RELOAD",
                lineno = node.lineno,
                reg    = data_reg,
                offset = offset,
                raw    = node.raw.strip(),
            ))

    # ── Match stores to reloads ───────────────────────────────────────────
    # Simple greedy match: for each store, find first reload with
    # same register name or same offset that comes after it
    used_reloads = set()
    for store in report.stores:
        for i, reload in enumerate(report.reloads):
            if i in used_reloads:
                continue
            if reload.lineno <= store.lineno:
                continue
            # Match by: same register name, OR same memory offset, OR
            # store reg matches reload reg (even if different variable names
            # the compiler may reuse the slot)
            reg_match    = store.reg and reload.reg and store.reg == reload.reg
            offset_match = (store.offset is not None
                            and reload.offset is not None
                            and store.offset == reload.offset)
            # Loose match: any reload after a store counts as a candidate
            # when register names differ (compiler renaming)
            loose_match  = (not reg_match and not offset_match
                            and store.reg and reload.reg)

            if reg_match or offset_match or loose_match:
                report.pairs.append(SpillPair(store=store, reload=reload))
                used_reloads.add(i)
                break

    return report
