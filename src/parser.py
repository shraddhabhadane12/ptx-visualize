"""
parser.py — PTX Parser for the Register Visualizer.
Parses PTX source text into a flat list of PTXNodes (Instruction | RawLine).
Only executable instructions are parsed deeply; everything else is a RawLine.
"""

import re
from typing import List
from .ir import Instruction, RawLine, PTXNode


_RE_INSTR = re.compile(
    r'^\s*(?:@(!?)(%\w+)\s+)?'   # optional predicate @%p0
    r'([\w][\w\.]*)'              # opcode
    r'(?:\s+(.+?))?'              # operands (lazy)
    r'\s*;?\s*(?://.*)?$'
)

_RE_REGISTER = re.compile(r'^%\w+$')

NO_DEST = ('st.','ret','bra','exit','bar','call','trap','membar',
           'prefetch','prefetchu')


def _has_dest(opcode: str) -> bool:
    for p in NO_DEST:
        if opcode == p.rstrip('.') or opcode.startswith(p):
            return False
    return True


def _split_ops(s: str) -> List[str]:
    """Split operand string on commas, respecting brackets."""
    s = re.sub(r'//.*$', '', s).rstrip(';').strip()
    if not s:
        return []
    parts, depth, cur = [], 0, []
    for ch in s:
        if   ch == '[': depth += 1; cur.append(ch)
        elif ch == ']': depth -= 1; cur.append(ch)
        elif ch == ',' and depth == 0:
            p = ''.join(cur).strip()
            if p: parts.append(p)
            cur = []
        else:
            cur.append(ch)
    p = ''.join(cur).strip()
    if p: parts.append(p)
    return parts


def parse_source(source: str) -> List[PTXNode]:
    """
    Parse full PTX source into PTXNodes.
    Instructions get lineno set to their 1-based position in the file.
    """
    nodes: List[PTXNode] = []
    for lineno, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()

        # Non-instruction lines → RawLine
        if (not stripped
                or stripped.startswith('//')
                or stripped.startswith('.')
                or (stripped and stripped[0] not in ('@','%')
                    and not (line and line[0] in (' ','\t')))):
            nodes.append(RawLine(text=line, lineno=lineno))
            continue

        m = _RE_INSTR.match(line)
        if not m:
            nodes.append(RawLine(text=line, lineno=lineno))
            continue

        opcode  = m.group(3)
        op_str  = m.group(4) or ''
        operands = _split_ops(op_str)

        dest = None
        src_ops = operands
        if _has_dest(opcode) and operands and _RE_REGISTER.match(operands[0]):
            dest    = operands[0]
            src_ops = operands[1:]

        nodes.append(Instruction(
            opcode=opcode, dest=dest,
            operands=src_ops, raw=line, lineno=lineno,
        ))
    return nodes


def instructions_only(nodes: List[PTXNode]) -> List[Instruction]:
    """Filter to only Instruction nodes."""
    return [n for n in nodes if isinstance(n, Instruction)]
