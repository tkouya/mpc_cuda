#!/usr/bin/env python3
"""make_figures.py -- generate the benchmark PDF figures for the TeX reports.

Data are the measured mpc_cuda benchmark results on an NVIDIA GB10 (compute
capability 12.1 / sm_121) and an NVIDIA H100 NVL (9.0 / sm_90), 1024-bit unless a
precision sweep.  The precision sweep is reported on BOTH GPUs side by side; the
H100 sweep is extended to 65536 bits so that the GPU fixed path (cu_freal /
cu_fcomplex), the GPU runtime path (cu_mpfr / cu_mpc) and the CPU all cross.
The GB10 sweep was measured to 8192 bits only (GB10 not re-measurable here), so
its curves stop there.  Output: doc/figures/*.pdf (vector, dvipdfmx-friendly).

Run with a Python that has matplotlib, e.g. a venv:
    PYTHONPATH=/tmp/pylibs python3 tools/make_figures.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "doc", "figures")
os.makedirs(OUT, exist_ok=True)

# precision axes: GB10 to 8192, H100 extended to 65536
bits   = [128, 256, 512, 1024, 2048, 4096, 8192]
bits_h = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

# --- bench3 AXPY times (ms), N=4096 ---
# GB10 (cu_freal / cu_mpfr / CPU; cu_fcomplex / cu_mpc / CPU)
g_real_freal = [0.005, 0.005, 0.006, 0.008, 0.056, 0.184, 0.823]
g_real_mpfr  = [0.050, 0.088, 0.174, 0.388, 0.506, 0.996, 1.989]
g_real_cpu   = [0.190, 0.222, 0.320, 0.677, 0.322, 0.473, 0.814]
g_cplx_fcplx = [0.009, 0.010, 0.012, 0.023, 0.259, 0.992, 3.644]
g_cplx_mpc   = [0.125, 0.225, 0.465, 1.202, 3.717, 7.206, 14.314]
g_cplx_cpu   = [0.610, 0.730, 1.123, 2.472, 2.634, 3.830, 6.377]
# H100 (extended to 65536 bits; all bit-exact)
h_real_freal = [0.004, 0.004, 0.005, 0.008, 0.022, 0.083, 0.430, 2.477, 64.635, 482.420]
h_real_mpfr  = [0.028, 0.036, 0.059, 0.120, 0.129, 0.228, 0.436, 0.884, 1.769, 3.601]
h_real_cpu   = [0.681, 0.663, 0.776, 0.786, 0.888, 1.235, 1.787, 2.847, 3.443, 6.375]
h_cplx_fcplx = [0.009, 0.010, 0.015, 0.044, 0.125, 0.509, 2.025, 11.693, 263.499, 1929.433]
h_cplx_mpc   = [0.074, 0.100, 0.163, 0.351, 0.932, 1.785, 3.626, 7.529, 15.429, 31.360]
h_cplx_cpu   = [1.955, 2.029, 2.595, 4.670, 6.309, 8.403, 13.105, 23.115, 30.636, 60.912]

# --- CPU-only: fixed precision / runtime MPFR(MPC) ratio (>1 = fixed faster) ---
cpu_real_ratio = [1.28, 1.23, 1.03, 0.71, 0.09, 0.03, 0.01]
cpu_cplx_ratio = [1.79, 1.48, 0.99, 0.67, 0.17, 0.06, 0.03]

MARK = dict(marker="o", markersize=5, linewidth=1.8)
# colour by series role, line style by GPU (solid = H100, dashed = GB10)
C_FIX = "#4C72B0"   # fixed precision  (cu_freal / cu_fcomplex)
C_RUN = "#DD8452"   # GPU runtime      (cu_mpfr  / cu_mpc)
C_CPU = "#55A868"   # CPU              (system MPFR / MPC)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ===================================================================
# Figs 4, 5: AXPY time vs precision, GB10 vs H100 (real, complex)
# solid = H100 (to 65536), dashed = GB10 (to 8192); blue/orange/green by path
# ===================================================================
def sweep_time(fixed_lbl, runtime_lbl,
               gf, gr, gc, hf, hr, hc, title, fname):
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    # H100 (solid)
    ax.loglog(bits_h, hf, "-",  base=2, color=C_FIX, **MARK, label=f"{fixed_lbl} H100")
    ax.loglog(bits_h, hr, "-",  base=2, color=C_RUN, **MARK, label=f"{runtime_lbl} H100")
    ax.loglog(bits_h, hc, "-",  base=2, color=C_CPU, **MARK, label="CPU H100")
    # GB10 (dashed, to 8192)
    ax.loglog(bits,   gf, "--", base=2, color=C_FIX, marker="s", markersize=4,
              linewidth=1.4, label=f"{fixed_lbl} GB10")
    ax.loglog(bits,   gr, "--", base=2, color=C_RUN, marker="s", markersize=4,
              linewidth=1.4, label=f"{runtime_lbl} GB10")
    ax.loglog(bits,   gc, "--", base=2, color=C_CPU, marker="s", markersize=4,
              linewidth=1.4, label="CPU GB10")
    ax.set_xlabel("mantissa precision (bits)")
    ax.set_ylabel("time per AXPY pass (ms)")
    ax.set_title(title)
    ax.set_xticks(bits_h); ax.set_xticklabels([str(b) for b in bits_h], fontsize=7, rotation=45)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=7, ncol=2)
    save(fig, fname)

sweep_time("cu_freal", "cu_mpfr", g_real_freal, g_real_mpfr, g_real_cpu,
           h_real_freal, h_real_mpfr, h_real_cpu,
           "AXPY real (MPFR): solid=H100, dashed=GB10", "fig_bench3_real.pdf")
sweep_time("cu_fcomplex", "cu_mpc", g_cplx_fcplx, g_cplx_mpc, g_cplx_cpu,
           h_cplx_fcplx, h_cplx_mpc, h_cplx_cpu,
           "AXPY complex (MPC): solid=H100, dashed=GB10", "fig_bench3_complex.pdf")


# ===================================================================
# Fig 6: GPU fixed-precision speedup vs precision, GB10 vs H100 (extended)
# speedup = baseline / fixed; y=1 is parity (fixed stops winning)
# ===================================================================
def spd(num, den):
    return [n / d for n, d in zip(num, den)]

fig, (axR, axC) = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
# real panel
axR.loglog(bits_h, spd(h_real_cpu,  h_real_freal), "-",  base=2, color=C_CPU, **MARK, label="vs CPU H100")
axR.loglog(bits_h, spd(h_real_mpfr, h_real_freal), "-",  base=2, color=C_RUN, **MARK, label="vs cu_mpfr H100")
axR.loglog(bits,   spd(g_real_cpu,  g_real_freal), "--", base=2, color=C_CPU, marker="s", markersize=4, linewidth=1.4, label="vs CPU GB10")
axR.loglog(bits,   spd(g_real_mpfr, g_real_freal), "--", base=2, color=C_RUN, marker="s", markersize=4, linewidth=1.4, label="vs cu_mpfr GB10")
axR.axhline(1.0, color="red", ls="--", lw=1)
axR.set_title("(a) cu_freal speedup (real)")
axR.set_ylabel(r"speedup ($\times$, log)")
# complex panel
axC.loglog(bits_h, spd(h_cplx_cpu, h_cplx_fcplx), "-",  base=2, color=C_CPU, **MARK, label="vs CPU H100")
axC.loglog(bits_h, spd(h_cplx_mpc, h_cplx_fcplx), "-",  base=2, color=C_RUN, **MARK, label="vs cu_mpc H100")
axC.loglog(bits,   spd(g_cplx_cpu, g_cplx_fcplx), "--", base=2, color=C_CPU, marker="s", markersize=4, linewidth=1.4, label="vs CPU GB10")
axC.loglog(bits,   spd(g_cplx_mpc, g_cplx_fcplx), "--", base=2, color=C_RUN, marker="s", markersize=4, linewidth=1.4, label="vs cu_mpc GB10")
axC.axhline(1.0, color="red", ls="--", lw=1)
axC.set_title("(b) cu_fcomplex speedup (complex)")
for ax in (axR, axC):
    ax.set_xlabel("mantissa precision (bits)")
    ax.set_xticks(bits_h); ax.set_xticklabels([str(b) for b in bits_h], fontsize=7, rotation=45)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=7)
fig.suptitle("GPU fixed-precision speedup vs precision (y=1: fixed stops winning)", fontsize=10)
save(fig, "fig_speedup_gpu.pdf")


# ===================================================================
# CPU-only crossover (fixed / MPFR ratio); >1 means fixed precision wins
# ===================================================================
fig, ax = plt.subplots(figsize=(5.4, 3.6))
ax.semilogx(bits, cpu_real_ratio, **MARK, base=2, label="cu_freal / system MPFR")
ax.semilogx(bits, cpu_cplx_ratio, **MARK, base=2, label="cu_fcomplex / system MPC")
ax.axhline(1.0, color="red", ls="--", lw=1, label="parity")
ax.set_yscale("log", base=10)
ax.set_xlabel("mantissa precision (bits)")
ax.set_ylabel("ratio (fixed / runtime)")
ax.set_title("CPU only: fixed precision vs MPFR/MPC")
ax.set_xticks(bits); ax.set_xticklabels([str(b) for b in bits])
ax.grid(True, which="both", ls=":", alpha=0.5)
ax.legend(fontsize=8)
save(fig, "fig_cpu_crossover.pdf")

# ===================================================================
# Cross-platform comparison: GB10 vs H100 (runtime-precision, 1024-bit)
# ===================================================================
demo_lbl   = ["AXPY\nMPFR", "AXPY\nMPC", "matvec\nMPFR", "matvec\nMPC",
              "matmul\nMPFR", "matmul\nMPC"]
gb10_gpu   = [1.17, 4.81, 11.3, 24.4, 23.0, 27.9]    # GB10 GPU time [ms]
h100_gpu   = [0.295, 1.037, 15.80, 29.68, 12.0, 15.625]  # H100 GPU time [ms]
gb10_spd   = [41.3, 39.9, 15.6, 7.3, 105.0, 103.0]   # GB10 GPU-vs-CPU speedup
h100_spd   = [143.0, 160.0, 9.2, 5.2, 175.0, 155.0]  # H100 GPU-vs-CPU speedup
xpos = list(range(len(demo_lbl)))
W = 0.38

# Fig: two panels -- (a) GPU time GB10 vs H100, (b) H100-over-GB10 GPU speedup
fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 3.8))
axL.bar([x - W/2 for x in xpos], gb10_gpu, W, label="GB10 (sm_121)", color="#4C72B0")
axL.bar([x + W/2 for x in xpos], h100_gpu, W, label="H100 (sm_90)",  color="#DD8452")
axL.set_yscale("log", base=10)
axL.set_ylabel("GPU time per pass (ms, log)")
axL.set_title("(a) GPU time: GB10 vs H100 (1024-bit)")
axL.set_xticks(xpos); axL.set_xticklabels(demo_lbl, fontsize=8)
axL.grid(True, axis="y", which="both", ls=":", alpha=0.5)
axL.legend(fontsize=8)

ratio = [g / h for g, h in zip(gb10_gpu, h100_gpu)]   # >1 => H100 faster
colors = ["#55A868" if r >= 1 else "#C44E52" for r in ratio]
axR.bar(xpos, ratio, 0.6, color=colors)
axR.axhline(1.0, color="grey", ls="--", lw=1)
for x, r in zip(xpos, ratio):
    axR.text(x, r + 0.07, f"{r:.2f}x", ha="center", va="bottom", fontsize=8)
axR.set_ylabel(r"H100 speedup over GB10 ($\times$)")
axR.set_title("(b) H100 / GB10 GPU-time ratio")
axR.set_xticks(xpos); axR.set_xticklabels(demo_lbl, fontsize=8)
axR.set_ylim(0, max(ratio) * 1.25)
axR.grid(True, axis="y", ls=":", alpha=0.5)
save(fig, "fig_gb10_vs_h100.pdf")

# Fig: GPU-vs-CPU speedup, GB10 vs H100, same six demos (each its own host CPU)
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.bar([x - W/2 for x in xpos], gb10_spd, W, label="GB10 (vs Grace CPU)", color="#4C72B0")
ax.bar([x + W/2 for x in xpos], h100_spd, W, label="H100 (vs x86 CPU)",  color="#DD8452")
ax.set_ylabel(r"GPU speedup vs host CPU ($\times$)")
ax.set_title("GPU-vs-CPU speedup by platform (1024-bit)")
ax.set_xticks(xpos); ax.set_xticklabels(demo_lbl, fontsize=8)
ax.grid(True, axis="y", ls=":", alpha=0.5)
ax.legend(fontsize=8)
save(fig, "fig_platform_speedup.pdf")

# ===================================================================
# Elementary/transcendental GPU speedup vs host CPU.
# H100 per-function detail (measured) + GB10 measured min..max range band.
# ===================================================================
elem_mpfr = [("sqrt",217.1),("cbrt",170.1),("exp",119.9),("expm1",134.7),
             ("log",105.5),("log1p",135.1),("sin",115.2),("cos",107.1),
             ("tan",113.0),("atan",84.3),("sinh",46.1),("cosh",65.4)]
elem_mpc  = [("sqr",179.5),("sqrt",123.8),("exp",137.3),("log",82.4),
             ("sin",56.4),("cos",56.4),("tan",43.0),("sinh",57.7),
             ("cosh",57.7),("asin",78.5),("acos",79.1),("atan",92.3)]
gb10_range = {"MPFR (real)": (42.0, 77.0), "MPC (complex)": (22.0, 63.0)}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.8))
for ax, data, ttl, col in ((a1, elem_mpfr, "MPFR (real)", "#4C72B0"),
                           (a2, elem_mpc, "MPC (complex)", "#DD8452")):
    names = [n for n, _ in data]; vals = [v for _, v in data]
    xs = list(range(len(names)))
    ax.bar(xs, vals, 0.7, color=col, label="H100 (per function)")
    lo, hi = gb10_range[ttl]
    ax.axhspan(lo, hi, color="grey", alpha=0.25,
               label=f"GB10 range {int(lo)}-{int(hi)}x")
    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=60, fontsize=7, ha="right")
    ax.set_title(f"{ttl}")
    ax.set_ylabel(r"GPU speedup vs host CPU ($\times$)")
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    ax.legend(fontsize=7)
fig.suptitle("Elementary/transcendental speedup: H100 per-function bars, GB10 range band (1024-bit)",
             fontsize=9)
save(fig, "fig_h100_elem.pdf")

print("done -> doc/figures/")
