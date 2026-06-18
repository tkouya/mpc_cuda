# mpc_cuda Manual

**Version 0.0.1**

mpc_cuda runs the GNU multiple-precision stack — **mini-gmp** (arbitrary-precision
integers), **MPFR** (correctly-rounded real floating point) and **MPC** (complex)
— *inside CUDA kernels*, so each GPU thread can do its own arbitrary-precision
computation. Every operation matches the host libraries bit-for-bit on the tested
inputs.

The device code is produced from the pristine upstream sources by re-runnable
transform scripts (`tools/cudafy_*.py`); mpc_cuda is not a hand-fork, so it can be
re-generated against future upstream releases.

---

## 1. Contents

1. [Requirements](#2-requirements)
2. [Building and installing](#3-building-and-installing)
3. [The `cu_` namespace and coexistence](#4-the-cu_-namespace-and-coexistence)
4. [Writing a kernel](#5-writing-a-kernel)
5. [The bump arena: concept and sizing](#6-the-bump-arena-concept-and-sizing)
6. [Linking your program](#7-linking-your-program)
7. [API surface](#8-api-surface)
8. [Demos and benchmarks](#9-demos-and-benchmarks)
9. [How it works](#10-how-it-works)
10. [Limitations](#11-limitations)
11. [License](#12-license)

---

## 2. Requirements

* An NVIDIA GPU and the **CUDA Toolkit** (`nvcc`). Default target architecture is
  `sm_121` (NVIDIA GB10); set another with `--with-cuda-arch=`.
* **Python 3** (runs the transform scripts).
* A C++ compiler (`g++`) and `nm` (binutils).
* The bundled upstream sources `gmp-6.3.0/mini-gmp`, `mpfr-4.2.2/src`,
  `mpc-1.4.1/src` (shipped in the distribution; override with `--with-*-src`).
* Optional: the **system** `libgmp`/`libmpfr`/`libmpc` and their development
  headers (`<mpfr.h>`, `<mpc.h>`) — only for `make coexist` / `make cputest` /
  `make cpubench`, which check the GPU port against the CPU libraries.

---

## 3. Building and installing

mpc_cuda uses an autoconf-based build (a hand-written `Makefile.in`, no automake,
because the build is driven by `nvcc` and the transform scripts).

```sh
./autogen.sh          # only if building from a git checkout (regenerates ./configure)
./configure           # see options below
make                  # build libmpc_cuda.a + libmpc_cuda.so (regenerate + compile)
make check            # build and run the test suite
make demos            # build the AXPY / matrix / benchmark demos
sudo make install     # install headers, libmpc_cuda.a/.so, mpc_cuda-link, mpc_cuda.pc
```

### configure options

| option | meaning | default |
|--------|---------|---------|
| `--with-cuda-arch=ARCH` | target GPU compute arch (`sm_121`, `sm_90`, …) | `sm_121` |
| `--with-gmp-src=DIR`  | mini-gmp source directory | `gmp-6.3.0/mini-gmp` |
| `--with-mpfr-src=DIR` | MPFR source directory     | `mpfr-4.2.2/src` |
| `--with-mpc-src=DIR`  | MPC source directory      | `mpc-1.4.1/src` |
| `--prefix=DIR`        | install prefix            | `/usr/local` |
| `NVCC=…`, `PYTHON=…`  | override the detected tools | autodetected |

> **Do not run several builds concurrently** in the same tree: they all
> regenerate and compile into the shared `build/` directory and would race.
> `make` is serial by default, so a normal `make -j1` (the default) is fine.

### What `make` produces and `make install` installs

`make` compiles every CUDA-adapted source into a **device-relocatable (`-rdc`,
`-fPIC`) object**, then bundles them into a single library in two forms:

* **`libmpc_cuda.a`** — static archive, the form a CUDA program links against.
  `nvcc -rdc=true … libmpc_cuda.a` device-links it: `nvlink` pulls exactly the
  device objects your kernels reference out of the archive.
* **`libmpc_cuda.so`** — shared library. A few upstream sources cannot be ported
  to device code (FILE\* I/O, missing mini-gmp helpers: `mpc_pow`, the `mpcb_*`
  balls, `out_str`, …); the objects that *transitively depend* on those missing
  symbols are pruned from the bundle so the archive is **self-consistent**
  (closed under reference) and neither `nvlink` nor `ld` ever hits an
  unresolved internal symbol. The `.so` is device-linked internally, so
  `nvlink` **cannot** pull device code out of it for an external kernel — a
  device-linked executable must use `libmpc_cuda.a`. The `.so` exposes the
  host-callable (`__host__ __device__`) `cu_*` API for **host-only** consumers.

`make install` lays down:

```
$(includedir)/mpc_cuda/            cuda_minigmp.h, cu_compat.h
$(includedir)/mpc_cuda/mpfr/       MPFR headers (mpfr.h, mpfr-impl.h, …)
$(includedir)/mpc_cuda/mpc/        MPC headers  (mpc.h, mpc-impl.h, …)
$(libdir)/libmpc_cuda.a            static library (device-link this)
$(libdir)/libmpc_cuda.so           shared library (host-callable cu_* API)
$(datadir)/mpc_cuda/               mpfr_cuda_defs.txt
$(bindir)/mpc_cuda-link            link helper (paths baked in)
$(libdir)/pkgconfig/mpc_cuda.pc    pkg-config metadata
```

---

## 4. The `cu_` namespace and coexistence

Every exported symbol is renamed into a **`cu_` namespace**:

| upstream | mpc_cuda |
|----------|----------|
| `mpz_*`, `mpn_*`, `gmp_*` | `cu_mpz_*`, `cu_mpn_*`, `cu_gmp_*` |
| `mpfr_*` | `cu_mpfr_*` |
| `mpc_*`  | `cu_mpc_*`  |

This lets a single program link **both** mpc_cuda (for the GPU) and the **real
system `libgmp`/`libmpfr`/`libmpc`** (for the CPU) without "multiple definition"
collisions. **Types** (`mpfr_t`, `mp_limb_t`), **rounding/enum macros**
(`MPFR_RNDN`, `MPC_RNDNN`), and the arena helpers (`mpc_cuda_*`) are *not* renamed.

* In your **`.cu`** files, call `cu_mpfr_*` / `cu_mpc_*` / `cu_mpz_*` directly, or
  `#include "mpc_cuda/cu_compat.h"` to keep writing the familiar `mpfr_*` names
  (it just `#define`s them to the `cu_` versions).
* In your **`.cpp`** files that use the system libraries, include the system
  `<mpfr.h>` and call `mpfr_*` as usual.
* **Do not** include both the mpc_cuda headers and the system headers in the *same*
  translation unit (the type `mpfr_t` would be defined twice). Keep them in
  separate `.cu` / `.cpp` files and link the objects together.

`make coexist` demonstrates a single binary that computes π and log 2 on the GPU
with `cu_mpfr` and on the CPU with the system `libmpfr`, and shows they agree.

---

## 5. Writing a kernel

### Minimal example

```cuda
#include <cstdio>
typedef long int gmp_randstate_t[1];   // shim expected by the headers
#include "mpfr.h"
#include "mpc_cuda/cuda_minigmp.h"
#include "mpc_cuda/cu_compat.h"        // optional: lets us write mpfr_* below

__global__ void k(double *out)
{
    mpfr_t x, r;
    mpfr_init2(x, 256);                 // 256-bit precision
    mpfr_init2(r, 256);
    mpfr_set_d(x, 2.0, MPFR_RNDN);
    mpfr_log(r, x, MPFR_RNDN);          // r = ln 2, correctly rounded
    out[0] = mpfr_get_d(r, MPFR_RNDN);
    mpfr_clear(x); mpfr_clear(r);
}
```

### Device resource limits

Deep MPFR/MPC call chains and the limb heap need larger-than-default limits. Set
them on the host *before* launching:

```cuda
cudaDeviceSetLimit(cudaLimitStackSize,      192 * 1024);          // 128–256 KB
cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256ull * 1024 * 1024);
```

MPC's complex multiply recurses fairly deep — use **128 KB+** of stack for MPC.

### Per-thread bump arena (recommended for speed)

By default each `mpfr_init2`/operation calls the device `malloc`/`free`, and that
allocator churn dominates. mpc_cuda provides an optional **per-thread bump arena**:
allocation is a pointer bump, `free` is a no-op, and you reclaim everything for the
next work item with one reset. This is what gives the large GPU-vs-CPU speedups.

Install it from the host and reset it per work item in the kernel:

```cuda
// host: one slab per launched thread
size_t SLAB = 32 * 1024;                 // bytes per thread; size to your workload
size_t nthreads = blocks * threads;
char   *arena; size_t *top;
cudaMalloc(&arena, nthreads * SLAB);
cudaMalloc(&top,   nthreads * sizeof(size_t));
cudaMemset(top, 0, nthreads * sizeof(size_t));
mpc_cuda_arena_base = arena;             // managed globals
mpc_cuda_arena_slab = SLAB;
mpc_cuda_arena_top  = top;

// kernel: grid-stride, reset the arena each iteration
for (int i = tid; i < N; i += stride) {
    mpc_cuda_arena_reset();              // reclaim the previous item's scratch
    ... mpfr/mpc work for element i ...
}
```

Leaving `mpc_cuda_arena_base == NULL` falls back to device `malloc`/`free`
(correct, just slower). Size `SLAB` to hold one work item's peak live memory; if an
item overflows its slab the allocator falls back to `malloc`.

**Launch shape matters.** The deep call chains keep limbs in (global-memory) arena
slabs, so many warps must be resident to hide that latency. Prefer a *fixed* pool
of thousands of threads grid-striding over a large problem, rather than one thread
per element of a small problem.

### Reduction/accumulation loops: stack-backed values

When you accumulate across an inner loop (dot products, series), the accumulator
must survive across `mpc_cuda_arena_reset()` calls. Put the persistent
operands on the **stack** so the arena only holds one operation's scratch:

* **Real:** use MPFR's `MPFR_DECL_INIT(name, PREC)` — a stack-allocated `mpfr_t`
  whose mantissa is a local array (no heap).
* **Complex:** build a stack-backed `mpc_t` from two `mpfr_custom_init_set`
  components (see `demos/matvec_mpc.cu`'s `MPC_DECL_INIT` macro).

```cuda
MPFR_DECL_INIT(acc, PREC);
MPFR_DECL_INIT(t, PREC);
mpfr_set_zero(acc, 1);
for (int j = 0; j < n; ++j) {
    mpc_cuda_arena_reset();              // acc and t survive (they are on the stack)
    mpfr_mul(t, a[j], x[j], MPFR_RNDN);
    mpfr_add(acc, acc, t, MPFR_RNDN);
}
```

---

## 6. The bump arena: concept and sizing

Section 5 showed *how* to install the arena. This section explains *what* it is and,
most importantly, *how large* to make it — the single tuning knob that most affects
performance and the one most likely to be set wrong.

### 6.1 What problem the arena solves

Every multiple-precision operation allocates memory. A `mpfr_t` needs a buffer for
its mantissa (the limbs), and almost every routine allocates one or more temporary
values on top of that — a single `cu_mpfr_div` or `cu_mpc_mul` can allocate and free
*dozens* of small buffers as it descends its call chain. On the CPU that is cheap. On
the GPU it is not: the device `malloc`/`free` is a single global serializing allocator
shared by every thread on the device, and when thousands of threads hammer it
simultaneously it becomes the dominant cost — in this library it was the difference
between GPU parity and a ~40× speedup.

The **bump arena** removes that cost. Instead of going to the device heap, every
allocation is served from a pre-reserved block of GPU global memory:

* **Allocate** = add the request size to a per-thread offset (a "bump") and hand back
  the old position. No search, no lock, no contention — a handful of instructions.
* **Free** = *nothing*. Freed memory is not reclaimed individually.
* **Reset** = set the per-thread offset back to zero, reclaiming the thread's entire
  scratch region in one stroke.

Because free is a no-op, every buffer you allocate between two resets stays live until
the next reset. The arena is therefore **not a general-purpose heap** — it is a
*scratchpad for one unit of work*. You do all the allocation for one work item, then
`mpc_cuda_arena_reset()` wipes it clean for the next.

### 6.2 How it is laid out

The arena is one contiguous `cudaMalloc`'d block, partitioned into one fixed-size
**slab** per *resident* thread:

```
mpc_cuda_arena_base ─┐
                     ▼
   ┌──────────┬──────────┬──────────┬─────  ...  ─────┐
   │ thread 0 │ thread 1 │ thread 2 │                 │   each region = SLAB bytes
   │  slab    │  slab    │  slab    │                 │
   └──────────┴──────────┴──────────┴─────  ...  ─────┘
        ▲
        └─ a thread's allocations bump upward inside its own slab;
           mpc_cuda_arena_top[tid] is the current offset (its high-water mark)
```

Each thread is indexed by its global thread id and owns exactly one slab, so there is
**no cross-thread contention** — threads never touch each other's slabs. The same
hooks back *both* mini-gmp and (through `mpfr-gmp.c`) MPFR and MPC, so installing the
arena accelerates every layer at once.

Two safety fallbacks keep results correct no matter how you size it:

* **No arena installed** (`mpc_cuda_arena_base == NULL`) → allocations go to the device
  `malloc`/`free`. Correct, just slow. This is why the existing CPU host path and tests
  are unaffected.
* **Slab exhausted** (one work item needs more than `SLAB` bytes) → *that* allocation
  spills to the device `malloc` and is freed normally. The result is still correct; you
  simply lose the speedup for the part that spilled. Overflow is therefore silent — it
  shows up as *disappointing performance*, not as a wrong answer or a crash.

### 6.3 The two numbers you choose

Total arena memory is the product of two independent quantities:

```
arena bytes  =  LAUNCH  ×  SLAB
               (resident   (bytes per
                threads)    thread)
```

* **`LAUNCH` = `LBLOCKS × LTHREADS`** — the *fixed* number of threads you launch. Pick
  this for **occupancy**: the deep MPFR/MPC call chains keep their limbs in
  global-memory slabs, and hiding that memory latency requires many warps resident at
  once. Thousands of threads (the demos use 8K–16K) grid-striding over the whole problem
  is the right shape — not one thread per element of a small problem. `LAUNCH` is
  independent of the problem size `N`.
* **`SLAB`** — bytes of scratch per thread. This is the value to get right, and §6.4 is
  about choosing it.

### 6.4 Sizing `SLAB`

`SLAB` must hold **one work item's peak simultaneously-live scratch**. Because free is a
no-op, "peak live" equals "everything allocated since the last reset" — so `SLAB` is the
total bytes a single element's worth of work allocates before its `mpc_cuda_arena_reset()`.

Two things drive that number:

1. **Precision.** One `mpfr_t` mantissa at *p* bits is about `ceil(p/64) × 8` bytes plus
   a small header. Scratch usage scales roughly **linearly** with precision: doubling *p*
   roughly doubles the slab you need.
2. **Operation depth.** A bare add needs a couple of temporaries; a transcendental
   (`exp`, `log`, `sin`) or a correctly-rounded `div` allocates many. **Complex (MPC) is
   far hungrier than real (MPFR)** because each complex operation expands into several
   real operations, often at extended internal precision.

**Empirical anchors** (the values the demos ship with, at **1024-bit**):

| workload | `SLAB` | note |
|----------|--------|------|
| real MPFR `axpy` / dot products | **32 KB** | a few `mpfr_t` plus arithmetic scratch |
| complex MPC `axpy` / dot products | **256 KB** | ~8× the real case: deeper chains, more temporaries |

Use these as your starting point and **scale linearly with precision**: e.g. for a real
MPFR kernel at 4096-bit, start around `32 KB × (4096/1024) = 128 KB`. Round up generously
— over-provisioning a slab only costs memory, while under-provisioning silently drops you
onto the slow path.

### 6.5 Measuring the exact peak (recommended)

You do not have to guess. `mpc_cuda_arena_top[tid]` *is* the high-water mark for a thread:
run **one** work item **without** calling `reset`, then copy `mpc_cuda_arena_top` back to
the host — the offset is exactly how many bytes that item consumed.

```cuda
// install a deliberately oversized arena (e.g. 8 MB/thread) so nothing spills,
// then run a SINGLE element on ONE thread and DO NOT reset:
//     ... one element's mpfr/mpc work ...
// (no mpc_cuda_arena_reset())

size_t peak;
cudaMemcpy(&peak, top, sizeof(size_t), cudaMemcpyDeviceToHost);
printf("peak live scratch = %zu bytes\n", peak);   // set SLAB a bit above this
```

Set `SLAB` to that peak plus a safety margin (say 25–50%) to absorb input-dependent
variation, then round to a convenient size. This turns sizing from guesswork into one
measurement.

### 6.6 Fitting the total budget, and a sanity checklist

`LAUNCH × SLAB` must fit in GPU memory alongside your input/output data. The demos reserve
**512 MB** (real axpy: 16K threads × 32 KB) and **2 GB** (complex axpy: 8K threads × 256 KB).
The usual order is: choose `LAUNCH` for occupancy first, then make sure `SLAB` is at least
the measured peak; if `LAUNCH × SLAB` then exceeds your memory budget, *lower `LAUNCH`*
(occupancy) rather than starving the slab below its peak.

* **Keep accumulators off the arena.** In reduction loops (dot products, series) the
  accumulator must survive across resets — put it on the **stack** with `MPFR_DECL_INIT`
  (real) or the `MPC_DECL_INIT` pattern (complex, see `demos/matvec_mpc.cu`). Then the
  arena only ever holds *one* multiply-add, so `SLAB` stays tiny regardless of loop length
  (this is exactly what §5's accumulation example does).
* **Reset once per work item**, at the top of each grid-stride iteration — not per thread
  lifetime. A thread processes many elements; it needs `SLAB` for only one at a time.
* **Symptom of an undersized slab:** performance well below the demo speedups even though
  results are bit-exact. That is the silent `malloc` fallback. Re-measure the peak (§6.5)
  and raise `SLAB`.
* **Symptom of an oversized arena:** `cudaMalloc` fails / out-of-memory at startup. Lower
  `SLAB` toward the measured peak, or lower `LAUNCH`.

Tune all of these without recompiling the library via `-D` overrides
(`SLAB`, `LBLOCKS`, `LTHREADS`, `N`, `PREC`) when building the demos.

---

## 7. Linking your program

`libmpc_cuda` is device-relocatable code, so a program is **device-linked**
against the static archive: compile your kernel with `-dc`, then link with
`-rdc=true` and `libmpc_cuda.a`, and `nvlink` pulls in exactly the device objects
your kernels reference. Three ways to do it:

**(a) installed helper** — the simplest:

```sh
mpc_cuda-link my_kernel.cu my_program
./my_program
```

Override the arch with `CUDA_ARCH=sm_90 mpc_cuda-link …`, and pass extra link
inputs (e.g. a CPU reference object plus system libraries) as trailing arguments
or via `EXTRA_LINK`.

**(b) from the build tree** — `tools/link_program.sh my_kernel.cu my_program`
(after `make`), or the all-in-one `tools/build_cuda_test.sh my_kernel.cu
my_program` (regenerate + compile + link).

**(c) by hand / pkg-config** — compile with `-dc`, then device-link the archive:

```sh
nvcc -dc -rdc=true $(pkg-config --cflags mpc_cuda) my_kernel.cu -o my_kernel.o
nvcc -rdc=true my_kernel.o $(pkg-config --libs mpc_cuda) -o my_program
```

(`pkg-config --libs mpc_cuda` expands to `-L$(libdir) -lmpc_cuda`. Because both
`.a` and `.so` are installed, force the static archive for the device link by
naming it directly — `$(libdir)/libmpc_cuda.a` — or use the `mpc_cuda-link`
helper, which always device-links the `.a`. The `.so` is for host-only programs
that call the `cu_*` API on the CPU.)

### Coexistence with the system libraries

Build the GPU side and the CPU side as **separate** translation units and link them
together. The GPU `.cu` uses `cu_mpfr_*`; the CPU `.cpp` includes the system
`<mpfr.h>` and uses `mpfr_*`; add `-lmpfr -lgmp` to the link:

```sh
g++ -c cpu_ref.cpp -o cpu_ref.o                       # system libmpfr
EXTRA_LINK="cpu_ref.o -lmpfr -lgmp" mpc_cuda-link gpu_side.cu app
```

See `demos/coexist_demo.cu` + `demos/coexist_cpu.cpp` for a complete example.

---

## 8. API surface

The API is the upstream GMP/MPFR/MPC API with the `cu_` prefix. Notable points:

* **mini-gmp (integers):** `cu_mpz_*` (`init`, `set`, `add`, `sub`, `mul`,
  `tdiv_qr`, `pow_ui`, `get_str`, …) and the low-level `cu_mpn_*`.
* **MPFR (real):** initialisation/assignment/arithmetic
  (`cu_mpfr_init2`, `cu_mpfr_set_d`, `cu_mpfr_add/sub/mul/div/sqrt`, …), the
  constant `cu_mpfr_const_pi` / `cu_mpfr_const_log2`, and the
  **transcendentals** `cu_mpfr_exp`, `expm1`, `log`, `log1p`, `sin`, `cos`,
  `tan`, `atan`, `sinh`, `cosh`, `cbrt`, … — all correctly rounded and
  bit-identical to host MPFR.
* **MPC (complex):** `cu_mpc_init2`, `cu_mpc_set_d_d`, `cu_mpc_add/sub/mul/sqr`,
  and complex elementary functions `cu_mpc_sqrt`, `exp`, `log`, `sin`, `cos`,
  `tan`, `sinh`, `cosh`, `asin`, `acos`, `atan`, …

**Rounding modes.** The runtime `cu_mpfr` / `cu_mpc` API uses **exactly the same
rounding-mode semantics as upstream MPFR / MPC**: every operation takes an
explicit rounding-mode argument and returns the same ternary value. The
`cu_mpfr_rnd_t` enum mirrors `mpfr_rnd_t` value-for-value — `CU_MPFR_RNDN` (=0,
nearest, ties to even), `CU_MPFR_RNDZ` (toward zero), `CU_MPFR_RNDU` (toward
+∞), `CU_MPFR_RNDD` (toward −∞), `CU_MPFR_RNDA` (away from zero), `CU_MPFR_RNDF`
(faithful) — and `cu_mpc_rnd_t` packs a real and an imaginary mode just like
MPC, with all combinations `CU_MPC_RNDNN … CU_MPC_RNDAA` and the
`CU_MPC_RND(re,im)` / `CU_MPC_RND_RE` / `CU_MPC_RND_IM` helpers. You select the
mode per call exactly as on the CPU. (Via `cu_compat.h` these are also reachable
under the plain `MPFR_RNDN` / `MPC_RNDNN` spellings.)

In contrast, the **fixed-precision** types (`cu_freal<PB>` / `cu_fcomplex<PB>`,
§8.1) are **round-to-nearest-even (RNDN) only** and take **no** rounding-mode
argument: every operation rounds RNDN at `PB` bits (the complex type rounds each
component RNDN, i.e. `MPC_RNDNN`). If you need a directed rounding mode, use the
runtime `cu_mpfr` / `cu_mpc` API.

**Device-correct division.** MPFR's generic division path miscompiles under
`nvcc` (the optimizer produces `inf` for ≥ 3-limb precision). mpc_cuda substitutes
a self-contained, correctly-rounded `mpfr_div` built on the (verified-correct)
`mpz` primitives. It matches host `mpfr_div` bit-for-bit and is what unblocks the
constants and transcendentals. This is transparent — you just call `cu_mpfr_div`.

Functions that are inherently host-only are **not** available on the device:
formatted/`*_str` I/O (`mpfr_printf`, `mpfr_set_str`, `mpfr_get_str` parsing,
`mpz_out_str`, …) and the `mpf_t`/`mpq_t` conversions absent from mini-gmp (their
device stubs trap if called).

### 8.1 Fixed-precision fast path — `cu_fp::cu_freal<PB>`

When the precision is **known at compile time**, the umbrella also exposes a
register-resident fixed-precision real type that is dramatically faster than the
runtime `cu_mpfr` path while staying **bit-exact with MPFR round-to-nearest**:

```cpp
#include "mpc_cuda.cuh"
using cu_fp::cu_freal;

__global__ void k(const double *x, const double *y, double *out, int n) {
  cu_freal<256> a = 1.5;                       // 256-bit mantissa
  for (int i = ...; i < n; ...) {
    cu_freal<256> r = a * cu_freal<256>(x[i]) + cu_freal<256>(y[i]);
    out[i] = (double) r;
  }
}
```

* **`PB` is the mantissa width in bits**, any multiple of **32** (32, 64, 96,
  128, 160, 256, 512, 1024, 2048, …). The significand is held in
  `ceil(PB/64)` limbs **in registers** — no arena, no `cudaLimitMallocHeapSize`,
  no runtime-precision dispatch. (Set only `cudaLimitStackSize` for very deep
  chains.)
* **Operations:** `operator+ - *`, or the free functions `cu_fp::cu_fmul`,
  `cu_fadd`, `cu_fsub`; conversions `cu_freal<PB>::from_double` / `(double)x`
  (implicit `cu_freal<PB>(d)` and `(double)` casts provided). Each operation
  rounds RNDN exactly like `mpfr_mul`/`mpfr_add`/`mpfr_sub` at `PB` bits.
* **Header-only**, usable on the **GPU and the host** (the host path uses
  `__uint128`); no link against `libmpc_cuda` is needed for this type alone.
* **Why it is fast:** compile-time precision lets `ptxas` keep every limb in a
  register, so the schoolbook multiply and the add/round run with zero global
  memory traffic. On a GB10 at 1024-bit this is ~**140×** faster than the
  runtime `cu_mpfr` GPU path for AXPY (and bit-identical). Validated bit-exact
  against the system MPFR for `PB` = 32…2048 across `mul`/`add`/`sub`.
* **Trade-off:** the precision must be a compile-time constant. For
  runtime-chosen precision, or for division / transcendentals, use the
  `cu_mpfr` API above.

**Fixed-precision complex — `cu_fp::cu_fcomplex<PB>`.** A complex type with the
same compile-time precision sits on top of `cu_freal`:

```cpp
using cu_fp::cu_fcomplex;
cu_fcomplex<256> a(1.5,-0.25), x(1.0,0.5), y(2.0,-1.0);
cu_fcomplex<256> z = a*x + y;            // operators + - *
double re = z.real_d(), im = z.imag_d();
```

Complex multiply is **correctly rounded per component** — `re = ∘(ar·br −
ai·bi)`, `im = ∘(ar·bi + ai·br)` — computed from the **exact** 2·N-limb partial
products (a register-resident `fmms`/`fmma`, no double rounding), so it is
**bit-exact with MPC's `MPC_RNDNN`**. Validated for `PB` = 32…2048. On a GB10 at
1024-bit, complex AXPY `z = a·x + y` runs ~**67×** faster than the runtime
`cu_mpc` GPU path (bit-identical).

`make sample-fixed` builds and runs `demos/sample_fixed.cu` (real **and**
complex, header-only, just `-Iinclude`); `make check-fixed` validates both
`cu_freal` and `cu_fcomplex` bit-exact against the system MPFR/MPC.

**Fixed-precision elementary functions** (`mpc_cuda/cu_fmath.cuh`,
`cu_fcmath.cuh`, also pulled in by the umbrella). High-accuracy (NOT
correctly-rounded — see below) elementary functions on `cu_freal<PB>` /
`cu_fcomplex<PB>`:

* real: `cu_fp::cu_`{`sqrt`,`cbrt`,`exp`,`expm1`,`log`,`log1p`,`sin`,`cos`,
  `tan`,`atan`,`sinh`,`cosh`,`asin`,`acos`,`atanh`,`pow`,`fdiv`}, plus the
  constants `cu_pi<PB>()`, `cu_ln2<PB>()`.
* complex: `cu_fp::cu_`{`cexp`,`clog`,`csqrt`,`csin`,`ccos`,`ctan`,`csinh`,
  `ccosh`,`cdiv`}.

They are computed at a working precision `PB + CU_FGUARD` (128 guard bits) with
Newton iteration + argument-reduced series, then rounded to `PB`. This is
**faithful to ~PB bits and agrees with MPFR/MPC to ≤ ~1 ULP** in the common
range — empirically **0 ULP** over the validation suite (`make check-fmath`,
`PB` = 64…1024). It is deliberately **not** correctly-rounded: the
table-maker's-dilemma cases would need an unbounded Ziv loop, incompatible with
register-resident fixed precision. Use the runtime `cu_mpfr`/`cu_mpc` API when
last-bit correct rounding is required. Real/complex **special** functions
(gamma, zeta, erf, Bessel, …) are future work. Fastest at low/medium precision
(values stay in registers); at very high `PB` the working set spills.

**Where the fixed-precision fast path wins (`make bench3`).** AXPY `r = a·x + y`
on a GB10, all bit-exact:

| mantissa bits | `cu_freal` GPU | `cu_mpfr` GPU | CPU MPFR | freal vs CPU |
|---|---|---|---|---|
| 128–1024 | ~0.005–0.008 ms (flat) | 0.05–0.39 ms | 0.19–0.68 ms | up to **83×** |
| 2048 | 0.056 ms (7× jump) | 0.51 ms | 0.32 ms | 5.7× |
| 4096 | 0.184 ms | 1.00 ms | 0.47 ms | 2.6× |
| 8192 | 0.823 ms | 1.99 ms | 0.81 ms | **1.0× (parity)** |

`cu_freal` time is flat through 1024-bit (fully register-resident), then **jumps
~7× at 2048-bit** — where the per-thread limb arrays (N = 32 limbs) spill to
local memory — and reaches parity with one CPU core by 8192-bit. So prefer the
fixed-precision path at **≤ ~1024 bits**; at very high precision use the runtime
`cu_mpfr`/`cu_mpc` path or the CPU. The complex side (`cu_fcomplex` vs `cu_mpc`
vs CPU MPC) follows the same shape, with an even larger fixed-precision lead
(peak ~48× vs `cu_mpc`, ~107× vs CPU at 1024-bit).

> **Device stack-size gotcha.** The deep `cu_mpc`/`cu_mpfr` call chain
> (`mpc_mul → mpfr_fmms → mpfr_sub → …`) needs a raised
> `cudaDeviceSetLimit(cudaLimitStackSize, …)`. On GB10 that limit has a maximum
> (≈ a few hundred KB): a too-large request (e.g. 512 KB) is **rejected with
> `cudaErrorInvalidValue`**, and if you don't check the return value the stack
> silently stays at the ~1 KB default and the kernel **stack-overflows** (seen as
> "illegal memory access"). 128 KB is accepted and sufficient. Always check the
> return value of `cudaDeviceSetLimit`.

---

## 9. Demos and benchmarks

All demos compare GPU vs CPU **time and accuracy** (the CPU baseline is this
library's own host path; every result is bit-exact). On an NVIDIA GB10, 1024-bit:

| target | what it does | result |
|--------|--------------|--------|
| `make axpy-mpfr` / `axpy-mpc` | `y = a·x + y`, real / complex | ~40× faster, bit-exact |
| `make matvec-mpfr` / `matvec-mpc` | matrix–vector `y = A·x` | up to ~16× |
| `make matmul-mpfr` / `matmul-mpc` | matrix multiply `C = A·B` | **~105×** |
| `make bench-mpfr` / `bench-mpc` | per-function elementary/transcendental benchmark | 22–77× |
| `make sample-fixed` | fixed-precision `cu_freal<PB>` / `cu_fcomplex<PB>` (real + complex) | header-only, bit-exact |
| `make check-fixed` | validate `cu_freal`/`cu_fcomplex` vs system MPFR/MPC, PB = 32…2048 | all bit-exact |
| `make check-fmath` | ULP accuracy of fixed-precision elementary functions vs MPFR/MPC | ≤1 ULP (0 in suite) |
| `make bench3` | AXPY 128..8192-bit: GPU `cu_freal` vs GPU `cu_mpfr` vs CPU MPFR | crossover sweep (below) |
| `make coexist` | `cu_mpfr` (GPU) + system `libmpfr` (CPU) in one binary | π, log 2 agree |
| `make cputest` | GPU `cu_mpfr`/`cu_mpc` vs system `libmpfr`/`libmpc` | bit-exact, all match |
| `make cpubench` | same suite, timed: GPU vs system-CPU per function | speedup + bit-exact |

`make cputest` / `make cpubench` link this library's GPU port and the ordinary
CPU `libmpfr`/`libmpc` into one binary (the `cu_` namespace makes them coexist)
and check the GPU against the CPU; they need the system libraries plus their
`<mpfr.h>`/`<mpc.h>` headers. The CPU half is `demos/cpu_ref.cpp`; both halves
share the type-free `demos/cpu_ref.h`.

Group targets: `make demos`, `make matrix`, `make bench`. Tune sizes with `-D`
overrides (`N`, `PREC`, `LBLOCKS`, `LTHREADS`, `SLAB`) when invoking the build
scripts directly.

---

## 10. How it works

`tools/cudafy_minigmp.py`, `cudafy_mpfr.py`, `cudafy_mpc.py` read the pristine
upstream sources and emit CUDA-callable versions: every device-reachable function
gets `__host__ __device__`; host-only functions (stdio/`realloc`/ctype) are
detected and left host-only; read-only tables get an address-space macro; thread
state and constant caches are made device-safe; and `realloc` is emulated with
malloc+copy+free. `tools/cu_prefix.py` then renames the exported symbols into the
`cu_` namespace. The whole pipeline is re-runnable, so a new upstream release can
be re-adapted by re-running the scripts.

`tools/build_lib.sh` compiles every generated source once; `tools/link_program.sh`
(and the installed `mpc_cuda-link`) resolve the device link closure for a user
program against those objects.

---

## 11. Limitations

* **Shared exponent/flags state.** MPFR's exponent range and flags
  (`__gmpfr_emin/emax/flags`) are shared across threads. Results match the host
  within 1e-13 up to the tested thread counts, but a fully race-free *per-thread*
  state is future work.
* **No device I/O.** Formatted printing and string parsing are host-only.
* **`mpf_t`/`mpq_t`** conversions are absent (mini-gmp build); their stubs trap.
* **Build concurrency.** Don't run multiple builds against the same tree at once
  (shared `build/` directory).
* This is an early **0.0.1** release; interfaces and packaging may change.

---

## 12. License

mpc_cuda is distributed under the **GNU Lesser General Public License, version 3
or (at your option) any later version** (LGPL-3.0-or-later), consistent with the
upstream GNU MPFR and MPC it transforms and bundles. See `COPYING.LESSER` (LGPLv3)
and `COPYING` (GPLv3).

The multiple-precision arithmetic is the work of the upstream GNU MP, MPFR and MPC
projects; see `AUTHORS`.
