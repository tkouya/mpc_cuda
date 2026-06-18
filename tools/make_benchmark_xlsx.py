#!/usr/bin/env python3
"""make_benchmark_xlsx.py -- write doc/mpc_cuda_benchmarks.xlsx with the measured
mpc_cuda benchmark results and embedded charts (xlsxwriter).

Covers two GPUs: NVIDIA GB10 (compute capability 12.1 / sm_121) and NVIDIA
H100 NVL (9.0 / sm_90).  The precision sweep (bench3) is GB10-only (it needs
system MPFR/MPC, absent on the H100 host); the six runtime-precision demos run
identically on both and are compared head-to-head.

Run:  PYTHONPATH=/tmp/pylibs python3 tools/make_benchmark_xlsx.py
"""
import os
import xlsxwriter

OUT = os.path.join(os.path.dirname(__file__), "..", "doc", "mpc_cuda_benchmarks.xlsx")
bits   = [128, 256, 512, 1024, 2048, 4096, 8192]   # GB10 (not re-measurable here)
bits_h = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]  # H100 extended

# bench3 AXPY (ms), GB10, N=4096, all bit-exact
real_freal = [0.005, 0.005, 0.006, 0.008, 0.056, 0.184, 0.823]
real_mpfr  = [0.050, 0.088, 0.174, 0.388, 0.506, 0.996, 1.989]
real_cpu   = [0.190, 0.222, 0.320, 0.677, 0.322, 0.473, 0.814]
cplx_fcplx = [0.009, 0.010, 0.012, 0.023, 0.259, 0.992, 3.644]
cplx_mpc   = [0.125, 0.225, 0.465, 1.202, 3.717, 7.206, 14.314]
cplx_cpu   = [0.610, 0.730, 1.123, 2.472, 2.634, 3.830, 6.377]
# CPU-only ratios (fixed / runtime), >1 = fixed faster
cpu_real_ratio = [1.28, 1.23, 1.03, 0.71, 0.09, 0.03, 0.01]
cpu_cplx_ratio = [1.79, 1.48, 0.99, 0.67, 0.17, 0.06, 0.03]

# bench3 AXPY (ms), H100 NVL, N=4096, all bit-exact (same demo, different GPU
# and host CPU -- the GPU columns are directly comparable to GB10 above).
# Extended to 65536 bits so the fixed path crosses the runtime path and the CPU.
h_real_freal = [0.004, 0.004, 0.005, 0.008, 0.022, 0.083, 0.430, 2.477, 64.635, 482.420]
h_real_mpfr  = [0.028, 0.036, 0.059, 0.120, 0.129, 0.228, 0.436, 0.884, 1.769, 3.601]
h_real_cpu   = [0.681, 0.663, 0.776, 0.786, 0.888, 1.235, 1.787, 2.847, 3.443, 6.375]
h_cplx_fcplx = [0.009, 0.010, 0.015, 0.044, 0.125, 0.509, 2.025, 11.693, 263.499, 1929.433]
h_cplx_mpc   = [0.074, 0.100, 0.163, 0.351, 0.932, 1.785, 3.626, 7.529, 15.429, 31.360]
h_cplx_cpu   = [1.955, 2.029, 2.595, 4.670, 6.309, 8.403, 13.105, 23.115, 30.636, 60.912]

wb = xlsxwriter.Workbook(OUT)
hdr  = wb.add_format({"bold": True, "bg_color": "#DDEBF7", "border": 1})
cell = wb.add_format({"border": 1})
num  = wb.add_format({"border": 1, "num_format": "0.000"})
rat  = wb.add_format({"border": 1, "num_format": "0.0\"x\""})
tit  = wb.add_format({"bold": True, "font_size": 13})

def ratio(a, b):  # b/a
    return [round(y/x, 2) for x, y in zip(a, b)]

# ---------- helper to write a precision-sweep sheet with charts ----------
def axpy_sheet(name, fixed, runtime, cpu, fixed_lbl, runtime_lbl, cpu_lbl, sem, platform="GB10", bitlist=None):
    bl = bitlist if bitlist is not None else bits
    ws = wb.add_worksheet(name)
    ws.write(0, 0, f"Multiple-precision AXPY  r = a*x + y   ({sem}, {platform}, N=4096, 8-iter avg, bit-exact)", tit)
    cols = ["precision (bits)", f"{fixed_lbl} [ms]", f"{runtime_lbl} [ms]", f"{cpu_lbl} [ms]",
            f"{fixed_lbl} vs {runtime_lbl}", f"{fixed_lbl} vs CPU"]
    for c, h in enumerate(cols):
        ws.write(2, c, h, hdr)
    r2r = ratio(fixed, runtime)
    r2c = ratio(fixed, cpu)
    for i, b in enumerate(bl):
        row = 3 + i
        ws.write(row, 0, b, cell)
        ws.write(row, 1, fixed[i], num)
        ws.write(row, 2, runtime[i], num)
        ws.write(row, 3, cpu[i], num)
        ws.write(row, 4, r2r[i], rat)
        ws.write(row, 5, r2c[i], rat)
    ws.set_column(0, 5, 20)
    n = len(bl); first, last = 3, 3 + n - 1

    # time chart (log y)
    ch = wb.add_chart({"type": "line"})
    for col, lbl in [(1, fixed_lbl), (2, runtime_lbl), (3, cpu_lbl)]:
        ch.add_series({
            "name": lbl,
            "categories": [name, first, 0, last, 0],
            "values":     [name, first, col, last, col],
            "marker": {"type": "circle", "size": 5},
        })
    ch.set_title({"name": f"AXPY time vs precision ({sem})"})
    ch.set_x_axis({"name": "precision (bits)"})
    ch.set_y_axis({"name": "time [ms]", "log_base": 10})
    ch.set_size({"width": 560, "height": 360})
    ws.insert_chart("H3", ch)

    # speedup chart
    ch2 = wb.add_chart({"type": "line"})
    for col, lbl in [(4, f"vs {runtime_lbl}"), (5, "vs CPU")]:
        ch2.add_series({
            "name": lbl,
            "categories": [name, first, 0, last, 0],
            "values":     [name, first, col, last, col],
            "marker": {"type": "circle", "size": 5},
        })
    ch2.set_title({"name": f"fixed-precision speedup ({sem})"})
    ch2.set_x_axis({"name": "precision (bits)"})
    ch2.set_y_axis({"name": "speedup (x)"})
    ch2.set_size({"width": 560, "height": 360})
    ws.insert_chart("H22", ch2)

axpy_sheet("AXPY real (GB10)", real_freal, real_mpfr, real_cpu,
           "cu_freal", "cu_mpfr", "system MPFR", "MPFR semantics", "GB10")
axpy_sheet("AXPY complex (GB10)", cplx_fcplx, cplx_mpc, cplx_cpu,
           "cu_fcomplex", "cu_mpc", "system MPC", "MPC semantics", "GB10")
axpy_sheet("AXPY real (H100)", h_real_freal, h_real_mpfr, h_real_cpu,
           "cu_freal", "cu_mpfr", "system MPFR", "MPFR semantics", "H100", bits_h)
axpy_sheet("AXPY complex (H100)", h_cplx_fcplx, h_cplx_mpc, h_cplx_cpu,
           "cu_fcomplex", "cu_mpc", "system MPC", "MPC semantics", "H100", bits_h)

# ---------- CPU-only crossover sheet ----------
ws = wb.add_worksheet("CPU fixed vs MPFR")
ws.write(0, 0, "CPU only (single thread): fixed precision vs system MPFR/MPC  (ratio > 1 => fixed precision faster)", tit)
for c, h in enumerate(["precision (bits)", "cu_freal / MPFR", "cu_fcomplex / MPC"]):
    ws.write(2, c, h, hdr)
for i, b in enumerate(bits):
    ws.write(3 + i, 0, b, cell)
    ws.write(3 + i, 1, cpu_real_ratio[i], rat)
    ws.write(3 + i, 2, cpu_cplx_ratio[i], rat)
ws.set_column(0, 2, 20)
ch = wb.add_chart({"type": "line"})
for col, lbl in [(1, "cu_freal / MPFR"), (2, "cu_fcomplex / MPC")]:
    ch.add_series({"name": lbl, "categories": ["CPU fixed vs MPFR", 3, 0, 9, 0],
                   "values": ["CPU fixed vs MPFR", 3, col, 9, col],
                   "marker": {"type": "circle", "size": 5}})
ch.set_title({"name": "CPU: fixed precision vs MPFR/MPC (parity at y=1)"})
ch.set_x_axis({"name": "precision (bits)"})
ch.set_y_axis({"name": "ratio (fixed / runtime)", "log_base": 10})
ch.set_size({"width": 600, "height": 360})
ws.insert_chart("E3", ch)

# ---------- runtime-precision AXPY + linear algebra + elementary ----------
ws = wb.add_worksheet("Linear algebra (GB10)")
ws.write(0, 0, "Runtime-precision MPFR/MPC on GPU vs CPU -- GB10 (1024-bit, bit-exact)", tit)
# AXPY summary
ws.write(2, 0, "AXPY  y=a*x+y  (N=16384)", wb.add_format({"bold": True}))
for c, h in enumerate(["kernel", "GPU [ms]", "CPU [ms]", "speedup"]):
    ws.write(3, c, h, hdr)
axpy = [("MPFR real", 1.17, 48.2, "41.3x"), ("MPC complex", 4.81, 191.7, "39.9x")]
for i, (k, g, c, s) in enumerate(axpy):
    ws.write(4 + i, 0, k, cell); ws.write(4 + i, 1, g, num)
    ws.write(4 + i, 2, c, num); ws.write(4 + i, 3, s, cell)
# matvec/matmul
ws.write(8, 0, "Matrix-vector / matrix multiply", wb.add_format({"bold": True}))
for c, h in enumerate(["kernel", "N", "GPU [ms]", "CPU [ms]", "speedup"]):
    ws.write(9, c, h, hdr)
la = [("MPFR matrix-vector", 256, 11.3, 177.0, 15.6),
      ("MPFR matrix multiply", 96, 23.0, 2410.0, 105.0),
      ("MPC matrix-vector", 128, 24.4, 179.0, 7.3),
      ("MPC matrix multiply", 64, 27.9, 2880.0, 103.0)]
for i, (k, n, g, c, s) in enumerate(la):
    row = 10 + i
    ws.write(row, 0, k, cell); ws.write(row, 1, n, cell)
    ws.write(row, 2, g, num); ws.write(row, 3, c, num)
    ws.write(row, 4, s, wb.add_format({"border": 1, "num_format": "0.0\"x\""}))
ws.set_column(0, 0, 22); ws.set_column(1, 4, 12)
ch = wb.add_chart({"type": "column"})
ch.add_series({"name": "GPU speedup vs CPU",
               "categories": ["Linear algebra (GB10)", 10, 0, 13, 0],
               "values": ["Linear algebra (GB10)", 10, 4, 13, 4],
               "data_labels": {"value": True}})
ch.set_title({"name": "Linear-algebra speedup (GB10, GPU vs CPU, 1024-bit)"})
ch.set_y_axis({"name": "speedup (x)"}); ch.set_legend({"none": True})
ch.set_size({"width": 560, "height": 340})
ws.insert_chart("G9", ch)

# elementary functions ranges
ws.write(16, 0, "Elementary/transcendental functions -- GB10 (N=4096, bit-exact)", wb.add_format({"bold": True}))
for c, h in enumerate(["suite", "functions", "GPU speedup vs CPU"]):
    ws.write(17, c, h, hdr)
ws.write(18, 0, "MPFR", cell)
ws.write(18, 1, "sqrt cbrt exp expm1 log log1p sin cos tan atan sinh cosh", cell)
ws.write(18, 2, "42-77x", cell)
ws.write(19, 0, "MPC", cell)
ws.write(19, 1, "sqr sqrt exp log sin cos tan sinh cosh asin acos atan", cell)
ws.write(19, 2, "22-63x", cell)
ws.set_column(1, 1, 50)

# ====================================================================
# Platform comparison sheet: GB10 vs H100 (the six runtime-precision demos)
# ====================================================================
# GPU time [ms] is the hardware-to-hardware metric (same source, same N, same
# 1024-bit precision, all bit-exact).  The GPU-vs-CPU speedups use each host's
# own CPU (GB10 = Grace ARM, H100 host = x86), so they are listed but not the
# basis of the GPU/GPU ratio.
cmp_rows = [
    # demo,                 N,     GB10 GPU, H100 GPU, GB10 spd, H100 spd
    ("AXPY MPFR",         16384,   1.17,    0.295,   "41.3x",  "143x"),
    ("AXPY MPC",          16384,   4.81,    1.037,   "39.9x",  "160x"),
    ("matvec MPFR",         256,  11.30,   15.800,   "15.6x",  "9.2x"),
    ("matvec MPC",          128,  24.40,   29.680,    "7.3x",  "5.2x"),
    ("matmul MPFR",          96,  23.00,   12.000,    "105x",  "175x"),
    ("matmul MPC",           64,  27.90,   15.625,    "103x",  "155x"),
]
ws = wb.add_worksheet("GB10 vs H100")
ws.write(0, 0, "Cross-platform GPU comparison: GB10 (sm_121) vs H100 NVL (sm_90), 1024-bit, bit-exact", tit)
cols = ["demo", "N", "GB10 GPU [ms]", "H100 GPU [ms]",
        "H100 speedup over GB10", "GB10 vs its CPU", "H100 vs its CPU"]
for c, h in enumerate(cols):
    ws.write(2, c, h, hdr)
for i, (demo, n, g_gpu, h_gpu, g_spd, h_spd) in enumerate(cmp_rows):
    row = 3 + i
    ws.write(row, 0, demo, cell)
    ws.write(row, 1, n, cell)
    ws.write(row, 2, g_gpu, num)
    ws.write(row, 3, h_gpu, num)
    ws.write(row, 4, round(g_gpu / h_gpu, 2), rat)   # >1 => H100 faster
    ws.write(row, 5, g_spd, cell)
    ws.write(row, 6, h_spd, cell)
ws.set_column(0, 0, 16); ws.set_column(1, 6, 18)
n = len(cmp_rows); first, last = 3, 3 + n - 1

# grouped GPU-time chart (log y): GB10 vs H100
chc = wb.add_chart({"type": "column"})
for col, lbl in [(2, "GB10 (sm_121)"), (3, "H100 (sm_90)")]:
    chc.add_series({"name": lbl,
                    "categories": ["GB10 vs H100", first, 0, last, 0],
                    "values":     ["GB10 vs H100", first, col, last, col]})
chc.set_title({"name": "GPU time per pass: GB10 vs H100 (1024-bit)"})
chc.set_x_axis({"name": "demo"})
chc.set_y_axis({"name": "GPU time [ms]", "log_base": 10})
chc.set_size({"width": 600, "height": 360})
ws.insert_chart("I3", chc)

# H100-over-GB10 GPU speedup chart (parity at y=1)
chr_ = wb.add_chart({"type": "column"})
chr_.add_series({"name": "H100 / GB10 GPU-time ratio",
                 "categories": ["GB10 vs H100", first, 0, last, 0],
                 "values":     ["GB10 vs H100", first, 4, last, 4],
                 "data_labels": {"value": True}})
chr_.set_title({"name": "H100 speedup over GB10 (>1 = H100 faster)"})
chr_.set_y_axis({"name": "ratio (x)"}); chr_.set_legend({"none": True})
chr_.set_size({"width": 600, "height": 360})
ws.insert_chart("I22", chr_)

# ====================================================================
# H100 elementary/transcendental per-function detail (new measurement)
# ====================================================================
elem_mpfr = [("sqrt",0.856,185.901,217.1),("cbrt",3.514,597.675,170.1),
             ("exp",10.600,1271.345,119.9),("expm1",9.693,1305.699,134.7),
             ("log",39.712,4189.484,105.5),("log1p",25.988,3510.634,135.1),
             ("sin",3.144,362.101,115.2),("cos",2.049,219.372,107.1),
             ("tan",3.366,380.269,113.0),("atan",21.747,1834.064,84.3),
             ("sinh",20.754,957.128,46.1),("cosh",14.316,936.438,65.4)]
elem_mpc  = [("sqr",0.238,42.640,179.5),("sqrt",2.929,362.583,123.8),
             ("exp",23.708,3255.791,137.3),("log",77.456,6378.885,82.4),
             ("sin",24.785,1398.009,56.4),("cos",24.749,1396.677,56.4),
             ("tan",36.796,1583.731,43.0),("sinh",24.655,1422.346,57.7),
             ("cosh",24.611,1420.464,57.7),("asin",87.691,6886.191,78.5),
             ("acos",94.979,7511.901,79.1),("atan",121.121,11179.519,92.3)]
ws = wb.add_worksheet("Elementary (H100)")
ws.write(0, 0, "H100 elementary/transcendental functions (N=4096, 1024-bit, bit-exact)", tit)

def elem_block(ws, top, title, data, sheet):
    ws.write(top, 0, title, wb.add_format({"bold": True}))
    for c, h in enumerate(["func", "GPU [ms]", "CPU [ms]", "GPU speedup vs CPU"]):
        ws.write(top + 1, c, h, hdr)
    base = top + 2
    for i, (fn, g, c, s) in enumerate(data):
        r = base + i
        ws.write(r, 0, fn, cell); ws.write(r, 1, g, num)
        ws.write(r, 2, c, num); ws.write(r, 3, s, num)
    ch = wb.add_chart({"type": "column"})
    ch.add_series({"name": "GPU speedup vs CPU",
                   "categories": [sheet, base, 0, base + len(data) - 1, 0],
                   "values":     [sheet, base, 3, base + len(data) - 1, 3],
                   "data_labels": {"value": True}})
    ch.set_title({"name": f"{title}: GPU speedup vs CPU"})
    ch.set_y_axis({"name": "speedup (x)"}); ch.set_legend({"none": True})
    ch.set_size({"width": 560, "height": 320})
    ws.insert_chart(top, 5, ch)
    return base + len(data)

next_top = elem_block(ws, 2, "MPFR (real)", elem_mpfr, "Elementary (H100)")
elem_block(ws, next_top + 2, "MPC (complex)", elem_mpc, "Elementary (H100)")
ws.set_column(0, 3, 14)

# ====================================================================
# Precision-sweep GPU comparison: GB10 vs H100 (bench3, GPU times only)
# ====================================================================
# The GPU columns (cu_freal/cu_fcomplex fixed, cu_mpfr/cu_mpc runtime) are
# directly comparable across the two GPUs; the host-CPU column is not (different
# CPUs), so it is omitted here -- see the per-platform AXPY sheets for it.
SWEET = "Sweep GB10 vs H100"
ws = wb.add_worksheet(SWEET)
ws.write(0, 0, "Precision sweep, GPU time [ms]: GB10 vs H100 (bench3 AXPY, N=4096, bit-exact)", tit)
series = [  # column label, data
    ("cu_freal GB10",  real_freal),  ("cu_freal H100",  h_real_freal),
    ("cu_mpfr GB10",   real_mpfr),   ("cu_mpfr H100",   h_real_mpfr),
    ("cu_fcplx GB10",  cplx_fcplx),  ("cu_fcplx H100",  h_cplx_fcplx),
    ("cu_mpc GB10",    cplx_mpc),    ("cu_mpc H100",    h_cplx_mpc),
]
ws.write(2, 0, "precision (bits)", hdr)
for c, (lbl, _) in enumerate(series):
    ws.write(2, 1 + c, lbl, hdr)
for i, b in enumerate(bits_h):
    row = 3 + i
    ws.write(row, 0, b, cell)
    for c, (_, data) in enumerate(series):
        if i < len(data):                 # GB10 series stop at 8192
            ws.write(row, 1 + c, data[i], num)
        else:
            ws.write_blank(row, 1 + c, None, cell)
ws.set_column(0, len(series), 15)
n = len(bits_h); first, last = 3, 3 + n - 1

def sweep_chart(title, col_pairs, anchor):
    ch = wb.add_chart({"type": "line"})
    for col, lbl in col_pairs:
        dash = "dash" if lbl.endswith("GB10") else "solid"
        ch.add_series({
            "name": lbl,
            "categories": [SWEET, first, 0, last, 0],
            "values":     [SWEET, first, col, last, col],
            "line": {"dash_type": dash},
            "marker": {"type": "circle", "size": 4},
        })
    ch.set_title({"name": title})
    ch.set_x_axis({"name": "precision (bits)"})
    ch.set_y_axis({"name": "GPU time [ms]", "log_base": 10})
    ch.set_size({"width": 560, "height": 360})
    ws.insert_chart(anchor, ch)

# columns 1..8 map to the 8 series above
sweep_chart("Real AXPY GPU time: solid=H100, dashed=GB10",
            [(1, "cu_freal GB10"), (2, "cu_freal H100"),
             (3, "cu_mpfr GB10"), (4, "cu_mpfr H100")], "K3")
sweep_chart("Complex AXPY GPU time: solid=H100, dashed=GB10",
            [(5, "cu_fcplx GB10"), (6, "cu_fcplx H100"),
             (7, "cu_mpc GB10"), (8, "cu_mpc H100")], "K22")

# ---------- README sheet ----------
ws = wb.add_worksheet("README")
notes = [
 "mpc_cuda benchmark results", "",
 "Platforms: NVIDIA GB10 (compute capability 12.1 / sm_121) and",
 "           NVIDIA H100 NVL (9.0 / sm_90).  CUDA 13, nvcc -fmad=false.",
 "All GPU results are bit-identical to host MPFR/MPC (elementary functions: <= 1 ULP).",
 "",
 "Sheets:",
 " - AXPY real/complex (GB10): precision sweep 128..8192-bit (make bench3);",
 "   AXPY real/complex (H100): same sweep EXTENDED to 65536-bit so the fixed path",
 "     crosses the runtime path (~8192-bit) and the CPU (~16384-bit).",
 "     GPU fixed-precision (cu_freal/cu_fcomplex) vs GPU runtime (cu_mpfr/cu_mpc) vs host CPU.",
 " - Sweep GB10 vs H100       : the bench3 GPU times (fixed + runtime) on BOTH GPUs, overlaid.",
 " - CPU fixed vs MPFR        : header-only cu_freal/cu_fcomplex run on the CPU vs system MPFR/MPC.",
 " - Linear algebra (GB10)    : GB10 runtime-precision AXPY, matvec/matmul, elementary ranges.",
 " - GB10 vs H100             : the six runtime-precision demos run on BOTH GPUs, head-to-head.",
 "     GPU time [ms] is the hardware metric; H100/GB10 ratio > 1 means H100 is faster.",
 " - Elementary (H100)        : per-function GPU/CPU times + speedup on the H100 (new detail).",
 "",
 "Key finding (fixed precision): the register-resident path wins hugely up to ~1024 bits,",
 "hits a register->local-memory spill knee at 2048 bits, and reaches CPU parity at 8192 bits.",
 "On a single CPU thread it beats MPFR/MPC only at 128-256 bits",
 "(schoolbook O(n^2) vs GMP/MPFR sub-quadratic multiply).",
 "",
 "Key finding (GB10 vs H100, demos): the H100 dominates the high-parallelism kernels",
 "(AXPY ~4x, matmul ~1.8-1.9x faster GPU time) but LOSES on the small-N matvec",
 "(0.72-0.82x): at N=128-256 only ~128-256 threads are active, so the kernel is",
 "occupancy-starved and the newer GB10 core wins that latency-bound regime.",
 "",
 "Key finding (GB10 vs H100, precision sweep): at N=4096 the H100 hides latency better,",
 "so the GPU runtime path (cu_mpfr/cu_mpc) is ~2-3x faster than GB10 at every precision,",
 "and the H100's fixed-precision spill knee at 2048 bits is far softer (cu_freal jumps",
 "~2.75x vs GB10's ~7x) thanks to its larger register file / local-memory bandwidth.",
]
for i, t in enumerate(notes):
    ws.write(i, 0, t)
ws.set_column(0, 0, 95)

wb.close()
print("wrote", os.path.normpath(OUT))
