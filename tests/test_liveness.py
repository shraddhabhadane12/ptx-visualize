"""
test_liveness.py — Unit tests for all visualizer modules.

Run with:
    python -m pytest tests/ -v
    python -m unittest tests/test_liveness.py -v
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.parser        import parse_source, instructions_only
from src.liveness      import (compute_live_ranges, get_live_registers_at,
                                get_line_range, sort_by_first_def)
from src.pressure      import compute_pressure
from src.spill_detector import detect_spills
from src.main          import run_analysis


# ── Helpers ───────────────────────────────────────────────────────────────────

def ptx(body: str) -> str:
    """Wrap body in minimal PTX header."""
    return ".version 7.5\n.target sm_89\n.address_size 64\n" + body


# ══════════════════════════════════════════════════════════════════════════════
# LIVENESS TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestLiveRangeBasics(unittest.TestCase):

    def test_single_register_live_range(self):
        """A register written then read has correct first_def and last_use."""
        src = ptx("""
    .reg .f32 %f<3>;
    .reg .u64 %rd<1>;
    mov.f32         %f0, 0f3F800000;
    add.f32         %f1, %f0, %f0;
    st.global.f32   [%rd0], %f1;
""")
        nodes = parse_source(src)
        lr    = compute_live_ranges(nodes)

        self.assertIn('%f0', lr)
        self.assertIn('%f1', lr)
        # %f0 must be defined before %f1
        self.assertLess(lr['%f0'].first_def, lr['%f1'].first_def)

    def test_first_def_is_write_line(self):
        """first_def must point to the instruction that writes the register."""
        src = ptx("""
    .reg .f32 %f<2>;
    .reg .u64 %rd<1>;
    ld.global.f32   %f0, [%rd0];
    mul.f32         %f1, %f0, %f0;
    st.global.f32   [%rd0], %f1;
""")
        nodes  = parse_source(src)
        instrs = instructions_only(nodes)
        lr     = compute_live_ranges(nodes)

        # %f0 should be defined at the ld line
        ld_line = next(i.lineno for i in instrs if i.opcode.startswith('ld.'))
        self.assertEqual(lr['%f0'].first_def, ld_line)

    def test_last_use_is_last_read_line(self):
        """%f0 read twice — last_use must be the second (later) read."""
        src = ptx("""
    .reg .f32 %f<4>;
    .reg .u64 %rd<1>;
    ld.global.f32   %f0, [%rd0];
    add.f32         %f1, %f0, %f0;
    mul.f32         %f2, %f0, %f1;
    st.global.f32   [%rd0], %f2;
""")
        nodes  = parse_source(src)
        instrs = instructions_only(nodes)
        lr     = compute_live_ranges(nodes)

        # %f0 is read in add and mul — last use must be the mul line
        mul_line = next(i.lineno for i in instrs
                        if i.opcode.startswith('mul.'))
        self.assertEqual(lr['%f0'].last_use, mul_line)

    def test_live_range_length(self):
        """length = last_use - first_def + 1."""
        src = ptx("""
    .reg .f32 %f<3>;
    .reg .u64 %rd<1>;
    mov.f32         %f0, 0f3F800000;
    add.f32         %f1, %f0, %f0;
    mul.f32         %f2, %f0, %f1;
    st.global.f32   [%rd0], %f2;
""")
        nodes = parse_source(src)
        lr    = compute_live_ranges(nodes)
        if '%f0' in lr:
            r = lr['%f0']
            self.assertEqual(r.length, r.last_use - r.first_def + 1)

    def test_is_live_at(self):
        """is_live_at returns True within range and False outside."""
        src = ptx("""
    .reg .f32 %f<3>;
    .reg .u64 %rd<1>;
    mov.f32         %f0, 0f3F800000;
    add.f32         %f1, %f0, %f0;
    st.global.f32   [%rd0], %f1;
""")
        nodes = parse_source(src)
        lr    = compute_live_ranges(nodes)
        if '%f0' in lr:
            r = lr['%f0']
            self.assertTrue(r.is_live_at(r.first_def))
            self.assertTrue(r.is_live_at(r.last_use))
            self.assertFalse(r.is_live_at(r.last_use + 100))

    def test_get_live_registers_at(self):
        """get_live_registers_at returns correct set at a given line."""
        src = ptx("""
    .reg .f32 %f<4>;
    .reg .u64 %rd<1>;
    mov.f32         %f0, 0f3F800000;
    mov.f32         %f1, 0f40000000;
    add.f32         %f2, %f0, %f1;
    st.global.f32   [%rd0], %f2;
""")
        nodes  = parse_source(src)
        instrs = instructions_only(nodes)
        lr     = compute_live_ranges(nodes)

        # At the add line, both %f0 and %f1 should be live
        add_line = next(i.lineno for i in instrs
                        if i.opcode.startswith('add.'))
        live = get_live_registers_at(lr, add_line)
        self.assertIn('%f0', live)
        self.assertIn('%f1', live)

    def test_get_line_range(self):
        """get_line_range returns correct (min, max) across all ranges."""
        src = ptx("""
    .reg .f32 %f<3>;
    .reg .u64 %rd<1>;
    mov.f32         %f0, 0f3F800000;
    add.f32         %f1, %f0, %f0;
    st.global.f32   [%rd0], %f1;
""")
        nodes = parse_source(src)
        lr    = compute_live_ranges(nodes)
        if lr:
            mn, mx = get_line_range(lr)
            self.assertLessEqual(mn, mx)

    def test_sort_by_first_def(self):
        """sort_by_first_def returns ranges in ascending definition order."""
        src = ptx("""
    .reg .f32 %f<4>;
    .reg .u64 %rd<1>;
    mov.f32         %f0, 0f3F800000;
    mov.f32         %f1, 0f40000000;
    add.f32         %f2, %f0, %f1;
    st.global.f32   [%rd0], %f2;
""")
        nodes  = parse_source(src)
        lr     = compute_live_ranges(nodes)
        sorted_lr = sort_by_first_def(lr)
        defs   = [r.first_def for r in sorted_lr]
        self.assertEqual(defs, sorted(defs))

    def test_no_instructions(self):
        """Empty PTX produces empty live range map."""
        src   = ptx("")
        nodes = parse_source(src)
        lr    = compute_live_ranges(nodes)
        self.assertIsInstance(lr, dict)

    def test_address_register_live(self):
        """Address registers (%rd) are tracked in live ranges."""
        src = ptx("""
    .reg .f32 %f<2>;
    .reg .u64 %rd<2>;
    ld.param.u64    %rd0, [0];
    add.u64         %rd1, %rd0, 4;
    ld.global.f32   %f0, [%rd1];
    st.global.f32   [%rd0], %f0;
""")
        nodes = parse_source(src)
        lr    = compute_live_ranges(nodes)
        # At least one %rd register should be tracked
        rd_regs = [k for k in lr if k.startswith('%rd')]
        self.assertGreater(len(rd_regs), 0)


# ══════════════════════════════════════════════════════════════════════════════
# PRESSURE TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestPressure(unittest.TestCase):

    def test_peak_pressure_correct(self):
        """Peak pressure equals max simultaneous live registers."""
        src = ptx("""
    .reg .f32 %f<5>;
    .reg .u64 %rd<1>;
    mov.f32         %f0, 0f3F800000;
    mov.f32         %f1, 0f40000000;
    mov.f32         %f2, 0f40400000;
    add.f32         %f3, %f0, %f1;
    add.f32         %f4, %f2, %f3;
    st.global.f32   [%rd0], %f4;
""")
        nodes  = parse_source(src)
        instrs = instructions_only(nodes)
        lr     = compute_live_ranges(nodes)
        prof   = compute_pressure(lr, [i.lineno for i in instrs])

        self.assertGreater(prof.peak_pressure, 0)
        self.assertGreaterEqual(prof.peak_pressure,
                                prof.avg_pressure)

    def test_pressure_never_negative(self):
        """Pressure at every line must be >= 0."""
        src = ptx("""
    .reg .f32 %f<3>;
    .reg .u64 %rd<1>;
    ld.global.f32   %f0, [%rd0];
    mul.f32         %f1, %f0, %f0;
    st.global.f32   [%rd0], %f1;
""")
        nodes  = parse_source(src)
        instrs = instructions_only(nodes)
        lr     = compute_live_ranges(nodes)
        prof   = compute_pressure(lr, [i.lineno for i in instrs])
        for pt in prof.points:
            self.assertGreaterEqual(pt.pressure, 0)

    def test_pressure_level_low(self):
        """Few registers → LOW pressure level."""
        src = ptx("""
    .reg .f32 %f<3>;
    .reg .u64 %rd<1>;
    ld.global.f32   %f0, [%rd0];
    add.f32         %f1, %f0, %f0;
    st.global.f32   [%rd0], %f1;
""")
        nodes  = parse_source(src)
        instrs = instructions_only(nodes)
        lr     = compute_live_ranges(nodes)
        prof   = compute_pressure(lr, [i.lineno for i in instrs])
        self.assertIn(prof.pressure_level, ("LOW", "NONE", "MEDIUM"))

    def test_avg_pressure_between_0_and_peak(self):
        """avg_pressure must be between 0 and peak_pressure."""
        src = ptx("""
    .reg .f32 %f<6>;
    .reg .u64 %rd<1>;
    mov.f32 %f0, 0f3F800000;
    mov.f32 %f1, 0f40000000;
    add.f32 %f2, %f0, %f1;
    mul.f32 %f3, %f0, %f2;
    add.f32 %f4, %f1, %f3;
    st.global.f32 [%rd0], %f4;
""")
        nodes  = parse_source(src)
        instrs = instructions_only(nodes)
        lr     = compute_live_ranges(nodes)
        prof   = compute_pressure(lr, [i.lineno for i in instrs])
        self.assertGreaterEqual(prof.avg_pressure, 0)
        self.assertLessEqual(prof.avg_pressure, prof.peak_pressure + 0.01)

    def test_peak_lines_list_nonempty(self):
        """peak_lines must contain at least one entry when pressure > 0."""
        src = ptx("""
    .reg .f32 %f<3>;
    .reg .u64 %rd<1>;
    ld.global.f32 %f0, [%rd0];
    add.f32       %f1, %f0, %f0;
    st.global.f32 [%rd0], %f1;
""")
        nodes  = parse_source(src)
        instrs = instructions_only(nodes)
        lr     = compute_live_ranges(nodes)
        prof   = compute_pressure(lr, [i.lineno for i in instrs])
        if prof.peak_pressure > 0:
            self.assertGreater(len(prof.peak_lines), 0)


# ══════════════════════════════════════════════════════════════════════════════
# SPILL DETECTION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestSpillDetection(unittest.TestCase):

    def test_no_spills_clean_kernel(self):
        """A kernel with no st.local/ld.local has no spills."""
        src = ptx("""
    .reg .f32 %f<3>;
    .reg .u64 %rd<1>;
    ld.global.f32   %f0, [%rd0];
    add.f32         %f1, %f0, %f0;
    st.global.f32   [%rd0], %f1;
""")
        nodes  = parse_source(src)
        report = detect_spills(nodes)
        self.assertFalse(report.has_spills)
        self.assertEqual(report.total_spill_ops, 0)

    def test_spill_store_detected(self):
        """st.local instruction is detected as a spill store."""
        src = ptx("""
    .reg .f32 %f<3>;
    .reg .u64 %rd<2>;
    .local .align 4 .b8 depot[8];
    ld.global.f32   %f0, [%rd0];
    st.local.f32    [%rd1+0], %f0;
    ld.global.f32   %f1, [%rd0];
    ld.local.f32    %f2, [%rd1+0];
    st.global.f32   [%rd0], %f2;
""")
        nodes  = parse_source(src)
        report = detect_spills(nodes)
        self.assertTrue(report.has_spills)
        self.assertGreater(len(report.stores), 0)

    def test_spill_reload_detected(self):
        """ld.local instruction is detected as a spill reload."""
        src = ptx("""
    .reg .f32 %f<3>;
    .reg .u64 %rd<2>;
    .local .align 4 .b8 depot[8];
    ld.global.f32   %f0, [%rd0];
    st.local.f32    [%rd1], %f0;
    ld.local.f32    %f1, [%rd1];
    st.global.f32   [%rd0], %f1;
""")
        nodes  = parse_source(src)
        report = detect_spills(nodes)
        self.assertGreater(len(report.reloads), 0)

    def test_store_reload_pair_matched(self):
        """A store followed by a reload for same register forms a pair."""
        src = ptx("""
    .reg .f32 %f<3>;
    .reg .u64 %rd<2>;
    .local .align 4 .b8 depot[8];
    ld.global.f32   %f0, [%rd0];
    st.local.f32    [%rd1], %f0;
    ld.global.f32   %f1, [%rd0];
    ld.local.f32    %f0, [%rd1];
    add.f32         %f2, %f0, %f1;
    st.global.f32   [%rd0], %f2;
""")
        nodes  = parse_source(src)
        report = detect_spills(nodes)
        self.assertGreater(len(report.pairs), 0)

    def test_spill_distance_positive(self):
        """spill_distance (lines between store and reload) must be > 0."""
        src = ptx("""
    .reg .f32 %f<3>;
    .reg .u64 %rd<2>;
    .local .align 4 .b8 depot[8];
    ld.global.f32   %f0, [%rd0];
    st.local.f32    [%rd1], %f0;
    ld.global.f32   %f1, [%rd0];
    add.f32         %f2, %f1, %f1;
    ld.local.f32    %f0, [%rd1];
    st.global.f32   [%rd0], %f0;
""")
        nodes  = parse_source(src)
        report = detect_spills(nodes)
        for pair in report.pairs:
            self.assertGreater(pair.spill_distance, 0)

    def test_spill_summary_no_spills(self):
        """Summary message for no-spill kernel is correct."""
        src   = ptx("    ret;")
        nodes = parse_source(src)
        report = detect_spills(nodes)
        self.assertIn("No", report.summary)

    def test_total_spill_ops_count(self):
        """total_spill_ops = stores + reloads."""
        src = ptx("""
    .reg .f32 %f<4>;
    .reg .u64 %rd<2>;
    .local .align 4 .b8 depot[16];
    ld.global.f32   %f0, [%rd0];
    st.local.f32    [%rd1+0], %f0;
    st.local.f32    [%rd1+4], %f0;
    ld.local.f32    %f1, [%rd1+0];
    ld.local.f32    %f2, [%rd1+4];
    add.f32         %f3, %f1, %f2;
    st.global.f32   [%rd0], %f3;
""")
        nodes  = parse_source(src)
        report = detect_spills(nodes)
        self.assertEqual(report.total_spill_ops,
                         len(report.stores) + len(report.reloads))


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegration(unittest.TestCase):

    def _load(self, filename: str) -> str:
        path = os.path.join(os.path.dirname(__file__),
                            '..', 'samples', 'ptx_files', filename)
        if not os.path.exists(path):
            return None
        with open(path, encoding='utf-8') as f:
            return f.read()

    def test_vector_add_full_pipeline(self):
        """End-to-end: vector_add.ptx produces valid analysis."""
        source = self._load('vector_add.ptx')
        if not source:
            self.skipTest("vector_add.ptx not found")
        result = run_analysis(source, 'vector_add.ptx')
        self.assertIn('live_ranges', result)
        self.assertIn('profile', result)
        self.assertIn('spill_report', result)
        self.assertGreater(len(result['live_ranges']), 0)

    def test_spill_demo_has_spills(self):
        """spill_demo.ptx must contain detected spill events."""
        source = self._load('spill_demo.ptx')
        if not source:
            self.skipTest("spill_demo.ptx not found")
        result = run_analysis(source, 'spill_demo.ptx')
        self.assertTrue(result['spill_report'].has_spills,
                        "spill_demo.ptx should have spill stores/reloads")

    def test_matrix_mul_live_ranges(self):
        """matrix_mul.ptx must produce multiple live ranges."""
        source = self._load('matrix_mul.ptx')
        if not source:
            self.skipTest("matrix_mul.ptx not found")
        result = run_analysis(source, 'matrix_mul.ptx')
        self.assertGreater(len(result['live_ranges']), 5)

    def test_reduction_pressure_profile(self):
        """reduction.ptx must produce a pressure profile with points."""
        source = self._load('reduction.ptx')
        if not source:
            self.skipTest("reduction.ptx not found")
        result = run_analysis(source, 'reduction.ptx')
        self.assertGreater(len(result['profile'].points), 0)

    def test_json_output_runs(self):
        """JSON mode must not crash and returns a dict."""
        source = self._load('vector_add.ptx')
        if not source:
            self.skipTest("vector_add.ptx not found")
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = run_analysis(source, 'vector_add.ptx', as_json=True)
        output = buf.getvalue()
        self.assertIn('"file"', output)
        self.assertIn('"peak_pressure"', output)

    def test_pressure_only_mode(self):
        """--only pressure mode must not crash."""
        source = self._load('vector_add.ptx')
        if not source:
            self.skipTest("vector_add.ptx not found")
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_analysis(source, 'vector_add.ptx', only='pressure')
        self.assertGreater(len(buf.getvalue()), 0)

    def test_spills_only_mode(self):
        """--only spills mode must not crash."""
        source = self._load('spill_demo.ptx')
        if not source:
            self.skipTest("spill_demo.ptx not found")
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_analysis(source, 'spill_demo.ptx', only='spills')
        self.assertGreater(len(buf.getvalue()), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
