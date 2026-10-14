"""
ir.py — Intermediate Representation for PTX Register Visualizer.

Each PTX line is parsed into one of:
  Instruction — executable line with opcode, dest, operands
  RawLine     — directive, label, blank, comment (passed through)
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Instruction:
    """A single executable PTX instruction."""
    opcode:   str                  # e.g. "add.f32", "ld.global.f32"
    dest:     Optional[str]        # destination register, None if no dest
    operands: List[str]            # source operands
    raw:      str                  # original text line
    lineno:   int = 0              # 1-based line number in file

    def base_op(self) -> str:
        """'add.f32' → 'add'"""
        return self.opcode.split('.')[0]

    def is_store(self)        -> bool: return self.opcode.startswith('st.')
    def is_load(self)         -> bool: return self.opcode.startswith('ld.')
    def is_spill_store(self)  -> bool: return self.opcode.startswith('st.local')
    def is_spill_reload(self) -> bool: return self.opcode.startswith('ld.local')
    def is_branch(self)       -> bool:
        return self.base_op() in ('bra','ret','exit','call','brx')
    def is_barrier(self)      -> bool:
        return self.base_op() in ('bar','membar')

    def all_read_regs(self) -> List[str]:
        """All registers READ by this instruction (sources + store dest-addr)."""
        import re
        regs = []
        for op in self.operands:
            regs += re.findall(r'%\w+', op)
        # Store: the memory address operand is a read, not a write
        if self.is_store() and self.dest:
            regs += re.findall(r'%\w+', self.dest)
        return regs

    def written_reg(self) -> Optional[str]:
        """Register written by this instruction, or None."""
        if self.dest and not self.is_store() and not self.is_branch():
            return self.dest
        return None


@dataclass
class RawLine:
    """A non-instruction PTX line."""
    text:   str
    lineno: int = 0


PTXNode = Instruction | RawLine
