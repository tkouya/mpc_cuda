# mpc_cuda

**Version 0.0.1** &nbsp;·&nbsp; LGPL-3.0-or-later

A CUDA port of the GNU multiple-precision stack — **mini-gmp → MPFR → MPC** —
so that arbitrary-precision integer, correctly-rounded floating-point, and
complex arithmetic can run **inside CUDA kernels**.

📖 **User manual:** [`doc/manual.md`](doc/manual.md) — requirements, build/install,
the `cu_` namespace, writing kernels, linking, API, limitations.

## Quick start

```sh
./configure                # detects nvcc, python3, CUDA arch (default sm_121)
make                       # build libmpc_cuda.a + libmpc_cuda.so
make check                 # build + run the test suite
make demos                 # AXPY / matrix / benchmark demos
sudo make install          # headers, libmpc_cuda.a/.so, mpc_cuda-link, mpc_cuda.pc
```

Then link your own kernel: `mpc_cuda-link my_kernel.cu my_program` (after install)
or `tools/build_cuda_test.sh my_kernel.cu my_program` (in the build tree).

## Easiest use — one header, coexists with the system GMP/MPFR/MPC

Just include **one** header and you get the whole stack inside your kernels:

```cuda
#include "mpc_cuda.cuh"     // or "mpc_cuda.h"  — the only include you need
```

Everything is in the `cu_` / `CU_` namespace, so this header can sit in the
**same `.cu` file** as the system CPU `<gmp.h>`/`<mpfr.h>`/`<mpc.h>` with no
clash, in either include order:

```cuda
#include <mpfr.h>          // system CPU MPFR  (plain names: mpfr_t, MPFR_RNDN …)
#include <mpc.h>           // system CPU MPC
#include "mpc_cuda.cuh"    // this GPU library (cu_ names: cu_mpfr_t, CU_MPFR_RNDN …)

__global__ void k(double *out) {
    cu_mpfr_t a;
    cu_mpfr_init2(a, 256);
    cu_mpfr_const_pi(a, CU_MPFR_RNDN);     // GPU
    *out = cu_mpfr_get_d(a, CU_MPFR_RNDN);
    cu_mpfr_clear(a);
}

int main() {
    mpfr_t c; mpfr_init2(c, 256);
    mpfr_const_pi(c, MPFR_RNDN);           // CPU, system libmpfr — no conflict
    mpfr_clear(c);
}
```

The naming map is mechanical:

| system (CPU) | mpc_cuda (GPU) |
|---|---|
| `mpfr_t`, `mpc_t`, `mpz_t` | `cu_mpfr_t`, `cu_mpc_t`, `cu_mpz_t` |
| `mpfr_init2`, `mpc_fma`, … | `cu_mpfr_init2`, `cu_mpc_fma`, … |
| `MPFR_RNDN`, `MPC_RNDNN` | `CU_MPFR_RNDN`, `CU_MPC_RNDNN` |
| `MPFR_PREC_MAX`, `MPFR_DECL_INIT` | `CU_MPFR_PREC_MAX`, `CU_MPFR_DECL_INIT` |

No `-D` flags and no extra `-I` are required beyond the install include root —
the umbrella header carries the build constants the library was compiled with.

```sh
make sample          # build + run demos/sample_easy.cu (GPU vs CPU, bit-exact)

# build your own (installed tree):
mpc_cuda-link my_kernel.cu my_program -lmpc -lmpfr -lgmp   # if you also use system MPC
# or by hand (minimal flags):
nvcc -arch=sm_121 -rdc=true -fmad=false -I<prefix>/include my_kernel.cu \
     <prefix>/lib/libmpc_cuda.a -lmpc -lmpfr -lgmp -o my_program
```

The single-include headers (`include/mpc_cuda/cu_{gmp,mpfr,mpc}.h`) are generated
from the cudafied public headers by `tools/make_umbrella.py` (`make umbrella`);
they are committed, so a normal build does not need to regenerate them.

> Tip: do **not** add `-I…/include/mpc_cuda/mpfr` or `…/mpc` to your compile when
> using the umbrella — those dirs contain the library's own `mpfr.h`/`mpc.h` and
> would shadow the system `<mpfr.h>`/`<mpc.h>`.  Just `-I<prefix>/include`.

### Fixed-precision fast path — `cu_freal<PB>` (compile-time precision)

The same umbrella header also gives a **register-resident fixed-precision** real
type for when the precision is known at compile time. It is **bit-exact with
MPFR (RNDN)** but ~**140× faster than the runtime `cu_mpfr` GPU path** at
1024-bit (AXPY, GB10), because a compile-time precision lets every significand
limb live in registers — no arena, no runtime dispatch.

```cuda
#include "mpc_cuda.cuh"
using cu_fp::cu_freal;

__global__ void axpy(const double *x, const double *y, double *out, int n) {
    cu_freal<256> a = 1.5;                 // 256-bit mantissa (any multiple of 32)
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += gridDim.x*blockDim.x)
        out[i] = (double)( a * cu_freal<256>(x[i]) + cu_freal<256>(y[i]) );
}
```

`PB` is the **mantissa width in bits, any multiple of 32** (32, 64, 96, 128,
256, 512, 1024, …). Operators `+ - *`, `from_double`/`(double)` conversions;
header-only, works on GPU **and** host. Validated bit-exact vs system MPFR for
`PB` = 32…2048. For runtime precision, division, or transcendentals, use the
`cu_mpfr` API.

A **complex** counterpart `cu_fp::cu_fcomplex<PB>` is included — its multiply is
correctly rounded per component (bit-exact with MPC `MPC_RNDNN`) via exact
2·N-limb partial products, and complex AXPY runs ~**67×** faster than the runtime
`cu_mpc` GPU path at 1024-bit:

```cuda
using cu_fp::cu_fcomplex;
cu_fcomplex<256> z = a*x + y;            // a,x,y complex; + - * operators
```

Try it: `make sample-fixed` (real + complex); validate: `make check-fixed`.

**Fixed-precision elementary functions** (`cu_fmath.cuh` / `cu_fcmath.cuh`, also
in the umbrella): `cu_fp::cu_`{`sqrt`,`cbrt`,`exp`,`expm1`,`log`,`log1p`,`sin`,
`cos`,`tan`,`atan`,`sinh`,`cosh`,`asin`,`acos`,`atanh`,`pow`} and complex
`cu_`{`cexp`,`clog`,`csqrt`,`csin`,`ccos`,`ctan`,`csinh`,`ccosh`}. High-accuracy
(working precision `PB+128`, Newton + argument-reduced series), agreeing with
MPFR/MPC to **≤1 ULP** (empirically 0 over `make check-fmath`, PB = 64…1024) —
not correctly-rounded by design. Special functions (gamma, zeta, erf, Bessel)
are future work.

The original goal was to run an `axpy` (`y = a*x + y`) demo with MPFR (real) and
MPC (complex) operands on the GPU, mirroring
[`cump_sm121/demos/axpy.cu`](https://github.com/tkouya/cump_sm121/tree/master/demos)
(which ports GMP's `mpf`).  Unlike CUMP's `mpf`, this library targets the
correctly-rounded **MPFR** semantics and the complex **MPC** layer.

Target hardware: NVIDIA **GB10** (compute capability 12.1), CUDA 13.
Set another GPU with `./configure --with-cuda-arch=sm_90`.

## Layered plan

| Phase | Layer    | Status | Notes |
|-------|----------|--------|-------|
| 1 | mini-gmp (`mpz`/`mpn`) | ✅ working | integer + low-level limb layer on device; tested vs host |
| 2 | MPFR (full, faithful) | ✅ core working | real `mpfr-4.2.2` `init2/set/mul/add/get_d` run in kernels, match host MPFR (`make test-mpfr`); transcendental/IO not yet device-ported |
| 3 | MPC (complex) | ✅ core working | real `mpc-1.4.1` `init2/set_d_d/mul/add` run in kernels, match host MPC (`make test-mpc`); see `docs/phase3-mpc.md` |
| 2b | MPFR transcendentals | ✅ working | `pi/exp/log/sin/cos/atan/sqrt` run on device, 128/128 threads match host (`make test-mpfr-trans`). Needed an mpz-based device `mpfr_div` (the generic `mpfr_div` miscompiles on device) — see `docs/phase2-mpfr.md` |
| 4 | `axpy` demos + benchmarks | ✅ working | MPFR (real) & MPC (complex) GPU-vs-CPU time + accuracy (`make demos`) |

### Phase 4 — AXPY benchmarks (`make axpy-mpfr`, `make axpy-mpc`)

`y = a*x + y` on N-vectors, MPFR (real) and MPC (complex), run on the GPU and
on the CPU (this library's MPFR/MPC host path), reporting time and accuracy.
The GPU launches a fixed pool of threads, grid-strides over the vector, and
backs limb allocation with a **per-thread bump arena** (see below). On the GB10,
N=16384 @ 1024-bit:

| demo | GPU | CPU | speedup | accuracy (max rel \|GPU−CPU\|) |
|------|-----|-----|---------|-------------------------------|
| MPFR real    | **1.17 ms** | 48.2 ms  | **41.3×** | **0.0 (bit-exact)** |
| MPC  complex | **4.81 ms** | 191.7 ms | **39.9×** | **0.0 (bit-exact)** |

GPU and CPU agree **bit-for-bit**, and the GPU is **~40× faster** for both real
and complex AXPY. Two things were needed:

1. **Per-thread bump arena** — without it each element does many device
   `malloc`/`free`s for its limbs and that allocator churn dominates (the real
   AXPY was 0.9× before the arena).
2. **Enough resident threads** — the deep `mpc_mul → mpfr_fmms → mpfr_sub`
   chain keeps its limbs in (global-memory) arena slabs, so many warps must be
   in flight to hide the memory latency. A small launch starves it (MPC was
   0.7× at N=1024); a fixed pool of ~8–16 K resident threads grid-striding over
   the vector hides the latency and gives the ~40× above.

For an optimized *system*-MPFR/MPC CPU baseline, link the system libraries in a
separate binary (their symbols would clash with the device build, which is
`__host__ __device__`).

### Linear algebra — matrix-vector and matrix-matrix products

`make matrix` (or `matvec-mpfr`, `matmul-mpfr`, `matvec-mpc`, `matmul-mpc`):
multiple-precision `y = A·x` and `C = A·B` for real (MPFR) and complex (MPC)
matrices, GPU vs CPU, every result **bit-exact** vs the host. The dot-product
accumulator/scratch are **stack-backed** (`MPFR_DECL_INIT` / a complex
`MPC_DECL_INIT` built from `mpfr_custom_init_set`), so the bump arena only has to
hold one multiply-add and is reset every inner iteration — the arena stays a few
KB regardless of the dimension. The GPU grid-strides over the output elements, so
the O(N²) matrix multiply (N² independent dot products) saturates the device:

| kernel | N | GPU | CPU | speedup |
|--------|---|-----|-----|---------|
| MPFR  matrix-vector | 256 | 11.3 ms | 177 ms  | 15.6× |
| MPFR  matrix multiply | 96 | 23.0 ms | 2.41 s  | **105×** |
| MPC   matrix-vector | 128 | 24.4 ms | 179 ms  | 7.3× |
| MPC   matrix multiply | 64 | 27.9 ms | 2.88 s  | **103×** |

(1024-bit; tune `N`/`PREC`/`LBLOCKS`/`LTHREADS` with `-D`.)

### Elementary / transcendental function benchmarks

`make bench` (or `bench-mpfr`, `bench-mpc`): evaluate a suite of functions over
N=4096 inputs, GPU vs CPU, reporting per-function time, speedup, and max relative
difference. **Every function is bit-exact** vs the host (max rel diff = 0).

- **MPFR** (`bench-mpfr`): `sqrt cbrt exp expm1 log log1p sin cos tan atan sinh
  cosh` — **42–77×** faster on the GPU.
- **MPC** (`bench-mpc`): `sqr sqrt exp log sin cos tan sinh cosh asin acos atan`
  — **22–63×** faster on the GPU.

### `cu_` symbol namespace — coexists with system GMP/MPFR/MPC

Every exported function/data symbol is renamed to a `cu_` namespace
(`mpfr_mul → cu_mpfr_mul`, `mpc_mul → cu_mpc_mul`, `mpz_add → cu_mpz_add`,
`mpn_* → cu_mpn_*`, …), so this CUDA library and the **real system
libgmp/libmpfr/libmpc** can be linked into the **same binary** with no
"multiple definition" collision. The rename is a whole-word textual pass
(`tools/cu_prefix.py`, driven by `tools/cu_rename_syms.txt`) applied to the
generated sources; **types** (`mpfr_t`, `mp_limb_t`), **rounding/enum macros**
(`MPFR_RNDN`), and the arena helpers (`mpc_cuda_*`) are left unchanged.

- Call the GPU functions as `cu_mpfr_*` / `cu_mpc_*` / `cu_mpz_*` in your `.cu`.
- `#include "mpc_cuda/cu_compat.h"` (optional) aliases the familiar names back
  (`mpfr_mul → cu_mpfr_mul`) for code that does **not** also include the system
  headers in the same translation unit — the demos/tests use it.
- Use the unprefixed `mpfr_*` (system) in your CPU `.cpp` as usual.

`make coexist` proves it: one executable computes π and log 2 on the **GPU with
`cu_mpfr`** and on the **CPU with the system `libmpfr`**, and the two agree:

```
=== Coexistence in ONE binary ===
GPU side : this library's cu_mpfr (CUDA kernel)
CPU side : system libmpfr 4.2.1 (unprefixed mpfr_*)
             GPU cu_mpfr            CPU system mpfr
  pi         3.141592653589793      3.141592653589793       MATCH
  log(2)     0.693147180559945      0.693147180559945       MATCH
```

#### Test & benchmark against the CPU `libmpfr` / `libmpc`

The same one-binary coexistence powers a correctness test and a benchmark that
link the **GPU port** and the **ordinary CPU `libmpfr` + `libmpc`** together and
check the former against the latter (needs the system libraries **and** their
`<mpfr.h>`/`<mpc.h>` development headers — `./configure` reports `system libmpc
.... yes`). The CPU half is `demos/cpu_ref.cpp` (system headers); the GPU half
is a `.cu` (the `cu_` API); they share `demos/cpu_ref.h`, which carries no
GMP/MPFR/MPC type so neither side's names leak into the other.

```
make cputest      # GPU cu_mpfr/cu_mpc  vs  system libmpfr/libmpc — bit-exact check
make cpubench     # same suite, timed: GPU [ms] vs CPU [ms], speedup, max rel diff
```

Both MPFR (real) and MPC (complex) are covered, over `sqrt/cbrt/exp/log/sin/cos/
tan/…` and `sqr/sqrt/exp/log/sin/cos/asin/acos/atan/…`. Because both libraries
are correctly rounded, every result agrees to the last bit (`max rel diff
0.000e+00`):

```
=== GPU cu_mpfr/cu_mpc  vs  system CPU libmpfr/libmpc ===
    system libmpfr 4.2.1, libmpc 1.3.1   (N=256 inputs, precision=200 bits)
MPFR (real):
  sqrt    max rel diff  0.000e+00   bit-exact 256/256   OK
  ...
MPC (complex):
  exp     max rel diff  0.000e+00   bit-exact 256/256   OK
  ...
ALL MATCH -- GPU port agrees with the system CPU libraries
```

### Per-thread bump arena (removes device-`malloc` churn)

`tools/cudafy_minigmp.py` provides an optional arena that backs **both**
mini-gmp and MPFR/MPC device allocation. Install a *bounded* arena sized for the
resident thread pool (not for N) and grid-stride over the data:

```c
#include "mpc_cuda/cuda_minigmp.h"
size_t ntot = LBLOCKS * LTHREADS;            // resident threads = arena slots
char *arena; size_t *top;
cudaMalloc(&arena, ntot*SLAB); cudaMalloc(&top, ntot*sizeof(size_t));
cudaMemset(top, 0, ntot*sizeof(size_t));
mpc_cuda_arena_base = arena;  mpc_cuda_arena_slab = SLAB;  mpc_cuda_arena_top = top;
// kernel: for (i = tid; i < N; i += stride) { mpc_cuda_arena_reset(); work(i); }
```

Each device allocation bumps the calling thread's slab; `free` is a no-op;
`mpc_cuda_arena_reset()` (once per work item) reclaims the slab. When
`mpc_cuda_arena_base == NULL` (default) allocation falls back to the device
heap, and the host always uses libc — so the same code runs both ways and the
existing correctness tests are unaffected. `SLAB`/`LBLOCKS`/`LTHREADS` are tunable
in the demos (`-DSLAB=...`, etc.).

**Device stack note:** kernels that call deep MPC/MPFR operations must raise the
CUDA per-thread stack (`cudaDeviceSetLimit(cudaLimitStackSize, 128*1024)`) — the
default ~1 KB overflows in e.g. `mpc_mul → mpfr_fmms → mpfr_sub`.

### Phase 2 — MPFR (faithful full port, validated feasible)

Per the chosen direction, **the real `mpfr-4.2.2` is ported faithfully** (not a
reimplementation). Two facts make this tractable and are already verified:

* MPFR builds against our mini-gmp on the host
  (`--with-mini-gmp … --disable-thread-safe --disable-decimal-float
  --disable-float128`) → no full-GMP internals needed.
* A sweep of all 263 `src/*.c` under `nvcc -x cu -dc` gives **258 clean
  compiles**; the 5 remaining are `#include`-only fragments, not standalone
  units. The only define change vs. host is dropping `-DMPFR_HAVE_NORETURN`
  (C11 `_Noreturn` is rejected by nvcc's C++ frontend).

**Core now runs on the GPU.** `tools/cudafy_mpfr.py` injects
`__host__ __device__`, routes MPFR's `mini-gmp.h` include to the device port,
makes the global state `__device__ __managed__`, tags lookup tables
`MPFR_RODATA`, and fixes the allocation routing (MPFR's `mpfr_allocate_func`
otherwise calls mini-gmp's *host* allocation hooks from the device). The test
`tests/test_mpfr.cu` runs `a*x + y` at 200 bits across 256 threads and matches
host MPFR bit-for-bit (`make test-mpfr`, 0 sanitizer errors). The full list of
device-porting fixes and the remaining (transcendental/IO) files are in
`docs/phase2-mpfr.md`; the validated define set is `tools/mpfr_cuda_defs.txt`.

## Approach

The GMP/MPFR/MPC sources are large and written for the host.  Rather than
fork-and-hand-edit them (which would make upstream updates painful), the device
adaptation is performed by **re-runnable transformation scripts** under
`tools/`.  Each script reads the pristine upstream source and emits a
CUDA-callable version where every device-reachable function carries
`__host__ __device__`, host-only facilities (stdio, `realloc`, `abort`,
`ctype`) are isolated, and dynamic allocation is rerouted to a device-aware
allocator.

### Phase 1 — mini-gmp (done)

`tools/cudafy_minigmp.py` adapts `gmp-6.3.0/mini-gmp/mini-gmp.{c,h}` into:

* `include/mpc_cuda/cuda_minigmp.h` — header with device-qualified prototypes
* `src/cuda_minigmp.cu` — implementation, device-qualified, with a CUDA
  allocation preamble

Key adaptations:

* **Allocation** — CUDA device code has `malloc`/`free` (per-thread device
  heap) but no `realloc`, so `gmp_realloc` is emulated with
  `malloc`+`memcpy`+`free` (see `mg_cuda_realloc`). Host builds use libc.
* **Error path** — `gmp_die` traps (`__trap`) on device, `abort`s on host.
* **Host-only functions** — those using stdio/`realloc`/`ctype`
  (`gmp_default_*`, `mp_*_memory_functions`, `mpz_out_str`, `mpz_set_str`, …)
  stay host-only and are detected automatically by a word-boundary token scan.

`mpz` arithmetic (add/sub/mul/`tdiv_qr`/`pow_ui`) and `mpz_get_str` all run on
the device and match host mini-gmp bit-for-bit
(`tests/test_minigmp.cu`, 256 threads).

> **Note on the device heap.** Each multi-precision variable allocates its
> limbs with device `malloc`. Tests raise `cudaLimitMallocHeapSize`. A future
> optimization is fixed-capacity, statically-sized limb storage to avoid the
> device heap entirely on the hot path.

## Build & test

mpc_cuda uses an **autoconf**-based build (hand-written `Makefile.in`, no automake,
since the build is driven by `nvcc` and the transform scripts). Full details in
[`doc/manual.md`](doc/manual.md).

```sh
./autogen.sh                       # only from a git checkout (regenerates ./configure)
./configure                        # detect nvcc/python3; --with-cuda-arch=sm_XX to retarget
make                               # build libmpc_cuda.a + libmpc_cuda.so (once)
make check                         # build + run test_minigmp / test_mpfr / test_mpc / test_mpfr_trans
make sample                        # one-file GPU+CPU coexistence sample (demos/sample_easy.cu)
make demos                         # AXPY / matrix / benchmark demos
make coexist                       # cu_mpfr (GPU) + system libmpfr (CPU) in one binary
make cputest                       # GPU port vs system libmpfr/libmpc — bit-exact check
make cpubench                      # GPU port vs system libmpfr/libmpc — timed benchmark
sudo make install                  # headers, libmpc_cuda.a/.so, mpc_cuda-link, mpc_cuda.pc
```

> Don't run multiple builds against the same tree concurrently — they share the
> `build/` directory and would race. `make` is serial by default.

## Layout

```
configure.ac Makefile.in mpc_cuda.pc.in   autoconf build system
tools/        re-runnable upstream → CUDA transform scripts; build_lib.sh,
              link_program.sh, mpc_cuda-link.in, cu_prefix.py, make_umbrella.py
include/      mpc_cuda.cuh / mpc_cuda.h   single-include umbrella headers
include/mpc_cuda/   device headers (cuda_minigmp.h, cu_compat.h,
              cu_gmp.h / cu_mpfr.h / cu_mpc.h — generated by make_umbrella.py)
src/          generated device sources
tests/        correctness tests (device vs host)
demos/        sample_easy (one-file GPU+CPU), axpy, matvec, matmul, benchmarks, coexistence
doc/manual.md user manual          docs/        design notes (phase 2/3)
COPYING / COPYING.LESSER           GPLv3 / LGPLv3 license texts
```
