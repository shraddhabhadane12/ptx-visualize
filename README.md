# PTX Register Visualizer

A static analysis tool for NVIDIA PTX assembly that visualizes **register live ranges**, **register pressure**, and **spill locations** — making it easy to see why a GPU kernel runs slow due to register allocation problems.

Built as a compiler engineering portfolio project targeting the **NVIDIA PTX Compiler Intern** role — third tool in the PTX toolkit alongside [ptx-optimizer](https://github.com/shraddhabhadane12/ptx-optimizer) and [ptx-analyzer](https://github.com/shraddhabhadane12/ptx-analyzer).

```
CUDA C++ → PTX (nvcc) → [THIS TOOL] → Live Range Chart + Pressure Graph + Spill Report
```

---

## What It Shows

### 1. Live Range Chart
A timeline showing when each register is alive (being held in a hardware register slot):

```
LIVE RANGE CHART — vector_add.ptx

             L4  L6  L8  L10 L12 L14 L16
             ──────────────────────────────
  %f0        ██  ██  ██  ██  ░░  ░░  ░░
  %f1        ░░  ██  ██  ░░  ░░  ░░  ░░
  %f2        ░░  ░░  ██  ██  ██  ░░  ░░
  %f3        ░░  ░░  ░░  ██  ██  ██  ░░
  %rd0       ██  ██  ██  ██  ██  ██  ██
  %p0        ░░  ░░  ░░  ██  ██  ░░  ░░
  ──────────────────────────────────────
  PRESSURE   2   3   4   4   3   2   1
```

`██` = live | `░░` = dead | `↓S` = spill store | `↑R` = spill reload

### 2. Pressure Graph
A vertical bar chart showing simultaneous live registers over time:

```
 8 │
 6 │       ████
 4 │    ████  ████
 2 │ ████        ████
   └────────────────────
     instruction flow →   (peak=6, avg=3.2)
```

### 3. Spill Report
Detects `st.local` (spill store) and `ld.local` (spill reload) patterns — signs that the compiler ran out of registers:

```
  Spill stores  : 2
  Spill reloads : 2
  Matched pairs : 2

  MATCHED PAIRS:
    %f4      store@28 → reload@35  (dist=7)
    %f6      store@29 → reload@36  (dist=7)
```

---

## Project Structure

```
ptx-visualizer/
├── src/
│   ├── ir.py              # Instruction / RawLine data structures
│   ├── parser.py          # PTX parser with lineno tracking
│   ├── liveness.py        # Two-pass dataflow: first_def + last_use
│   ├── pressure.py        # Per-line pressure + occupancy estimate
│   ├── spill_detector.py  # st.local / ld.local detection + pair matching
│   ├── visualizer.py      # ASCII chart renderer (live range + pressure + spills)
│   └── main.py            # CLI entry point
├── samples/
│   └── ptx_files/
│       ├── vector_add.ptx    # Simple parallel addition
│       ├── matrix_mul.ptx    # Tiled matrix multiplication
│       ├── reduction.ptx     # Parallel tree reduction
│       └── spill_demo.ptx    # Kernel with intentional spills
├── tests/
│   └── test_liveness.py   # 24 unit tests
└── README.md
```

---

## Quick Start

**Requirements:** Python 3.10+, no external dependencies.

```bash
# Clone
git clone https://github.com/shraddhabhadane12/ptx-visualizer.git
cd ptx-visualizer

# Full report (live ranges + pressure + spills)
python -m src.main samples/ptx_files/vector_add.ptx

# Verbose (also shows top-10 longest live ranges)
python -m src.main samples/ptx_files/vector_add.ptx --verbose

# Show only the pressure graph
python -m src.main samples/ptx_files/vector_add.ptx --only pressure

# Show only the live range chart
python -m src.main samples/ptx_files/matrix_mul.ptx --only live

# Show only the spill report
python -m src.main samples/ptx_files/spill_demo.ptx --only spills

# Filter to specific registers
python -m src.main samples/ptx_files/vector_add.ptx --regs %f0,%f1,%rd0

# JSON output
python -m src.main samples/ptx_files/vector_add.ptx --json

# Analyze all PTX files in a directory
python -m src.main samples/ptx_files/

# Run all unit tests
python -m unittest tests/test_liveness.py -v
```

---

## Sample Output

```
╔════════════════════════════════════════════════════════════════════╗
║                    PTX REGISTER VISUALIZER                        ║
║                    File: spill_demo.ptx                           ║
╠════════════════════════════════════════════════════════════════════╣
║  SUMMARY                                                          ║
║  ──────────────────────────────────────────────────────────────   ║
║    Total instructions     : 38                                    ║
║    Unique registers       : 24                                    ║
║    Peak register pressure : 14  (HIGH)                            ║
║    Avg register pressure  : 8.3                                   ║
║    Estimated occupancy    : 50%  (1024 threads/SM)                ║
║    Spill operations       : 4  (2 stores, 2 reloads)              ║
╠════════════════════════════════════════════════════════════════════╣
║                     LIVE RANGE CHART                              ║
║    ██ = live   ░░ = dead  ↓S = spill store  ↑R = spill reload     ║
║                                                                   ║
║  ... (register timeline chart) ...                                ║
╠════════════════════════════════════════════════════════════════════╣
║                  REGISTER PRESSURE OVER TIME                      ║
║  ... (ASCII bar chart) ...                                        ║
╠════════════════════════════════════════════════════════════════════╣
║                    SPILL DETECTION REPORT                         ║
║    Spill stores  : 2                                              ║
║    Spill reloads : 2                                              ║
║    Matched pairs : 2                                              ║
║    %f4   store@28 → reload@35  (dist=7)                           ║
║    %f6   store@29 → reload@36  (dist=7)                           ║
╚════════════════════════════════════════════════════════════════════╝
```

---

## How It Works

### Two-Pass Dataflow Analysis (liveness.py)

```
Pass 1 — Forward (top → bottom):
    Record first_def[reg] = line where reg is first WRITTEN

Pass 2 — Backward (bottom → top):
    Record last_use[reg]  = line where reg is last READ

Live Range = [first_def, last_use]
```

### Register Pressure (pressure.py)

```
For each instruction line L:
    pressure[L] = count of registers where first_def ≤ L ≤ last_use
```

GPU occupancy formula:
```
max_threads_per_SM = 65536 / registers_per_thread
occupancy          = max_threads_per_SM / 2048 × 100%
```

### Spill Detection (spill_detector.py)

```
st.local.* [addr], %reg  →  SPILL STORE  (value pushed to memory)
ld.local.* %reg,   [addr] →  SPILL RELOAD (value restored from memory)
```

Matched pairs: each store is paired with its closest subsequent reload of the same register.

---

## Skills Demonstrated

| Concept | Implementation |
|---|---|
| **Liveness Analysis** | Two-pass dataflow (forward + backward scan) |
| **Live Ranges** | first_def + last_use computation per register |
| **Register Pressure** | Per-instruction count of simultaneously live regs |
| **GPU Occupancy** | 65536/regs formula for max threads per SM |
| **Spill Detection** | st.local/ld.local pattern matching + pair matching |
| **ASCII Visualization** | Timeline chart + pressure bar graph |
| **PTX ISA** | Parsing real PTX register declarations and instructions |
| **Dataflow Analysis** | Core compiler theory applied to GPU assembly |

---

## The PTX Toolkit

All three tools work on the same PTX files and build on each other:

```
ptx-optimizer   →  removes dead instructions, folds constants  (transform)
ptx-analyzer    →  reports register count, divergence, mix     (measure)
ptx-visualizer  →  shows live ranges, pressure, spill points   (visualize)
```

Together: analyze → understand → optimize.

---

## References

- [NVIDIA PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Register Allocation — Dragon Book (Aho, Lam, Sethi, Ullman)](https://en.wikipedia.org/wiki/Compilers:_Principles,_Techniques,_and_Tools)
- [CUDA Occupancy Calculator](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/)

---

## License

MIT License
