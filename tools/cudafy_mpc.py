#!/usr/bin/env python3
"""cudafy_mpc.py -- mechanically adapt MPC (mpc-1.4.1) for CUDA.

MPC is the complex layer on top of MPFR.  This reuses the MPFR transform
machinery (tools/cudafy_mpfr.py) to inject `__host__ __device__` and tag
read-only data, and adds MPC-specific handling:

* `mpc.h` includes "gmp.h" and "mpfr.h"; we drop a gmp.h shim (-> the CUDA
  mini-gmp) and rely on the cudafied mpfr headers via the include path.
* MPC's `_Complex`/<complex.h> support (mpc_set_dc/get_dc, double-complex
  conversions) is not available on the device, so it is disabled by turning
  mpc.h's `#if 1` (which defines _MPC_HAVE_COMPLEX_H) into `#if 0` and by NOT
  defining HAVE_COMPLEX_H in the generated config.h.
* Function prototypes use `__MPC_DECLSPEC`; those get `__host__ __device__`.
* IO functions (out_str/inp_str/printf/logging) stay host-only via the token
  scan inherited from cudafy_mpfr.

Usage: python3 tools/cudafy_mpc.py <mpc_src_dir> <out_dir>
"""
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cudafy_mpfr as M   # reuse attribute_definitions / attribute_const_data
import cu_prefix


def attribute_mpc_prototypes(text):
    """Prepend __host__ __device__ to __MPC_DECLSPEC function prototypes
    (not to `extern` data declarations)."""
    out = []
    for line in text.split("\n"):
        s = line.lstrip()
        if s.startswith("__MPC_DECLSPEC") and "extern" not in s:
            out.append(M.ATTR + " " + line)
        else:
            out.append(line)
    return "\n".join(out)


# A minimal config.h for the device build (no complex.h, no stdio-only feature
# that the device path needs).  HAVE_COMPLEX_H is deliberately omitted.
CONFIG_H = """/* minimal config.h for the CUDA build of MPC (generated) */
#ifndef MPC_CONFIG_H
#define MPC_CONFIG_H
#define HAVE_STDLIB_H 1
#define HAVE_STDINT_H 1
#define HAVE_INTTYPES_H 1
#define HAVE_INTMAX_T 1
#define HAVE_LONG_LONG 1
/* HAVE_COMPLEX_H intentionally NOT defined: device has no _Complex support */
#endif
"""

GMP_SHIM = """/* shim: MPC's #include "gmp.h" -> CUDA mini-gmp + RODATA helper */
#include "mpc_cuda/cuda_minigmp.h"
#ifndef MPFR_RODATA
# ifdef __CUDA_ARCH__
#  define MPFR_RODATA __device__
# else
#  define MPFR_RODATA
# endif
#endif

/* mini-gmp has no random functions: MPFR/MPC need gmp_randstate_t declared. */
#ifndef gmp_randstate_t
typedef long int __gmp_randstate_struct;
typedef __gmp_randstate_struct gmp_randstate_t[1];
#endif

/* mini-gmp provides neither mpf_t (GMP float) nor mpq_t.  MPC declares a few
   conversion prototypes (mpc_set_f/_q, ...) that we do not build for the
   device; provide minimal type stand-ins so the public header parses. */
#ifndef __MPF_STRUCT_DEFINED
#define __MPF_STRUCT_DEFINED
typedef struct { int _mp_prec; int _mp_size; long _mp_exp; mp_limb_t *_mp_d; } __mpf_struct;
typedef __mpf_struct mpf_t[1];
typedef __mpf_struct *mpf_ptr;
typedef const __mpf_struct *mpf_srcptr;
typedef struct { __mpz_struct _mp_num; __mpz_struct _mp_den; } __mpq_struct;
typedef __mpq_struct mpq_t[1];
typedef __mpq_struct *mpq_ptr;
typedef const __mpq_struct *mpq_srcptr;
#endif
"""


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: cudafy_mpc.py <mpc_src_dir> <out_dir>")
    src, out = sys.argv[1:]
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)

    with open(os.path.join(out, "gmp.h"), "w") as f:
        f.write(cu_prefix.rename(GMP_SHIM))
    with open(os.path.join(out, "config.h"), "w") as f:
        f.write(CONFIG_H)

    for fn in sorted(os.listdir(src)):
        if not fn.endswith((".c", ".h")):
            continue
        with open(os.path.join(src, fn)) as f:
            text = f.read()

        if fn == "mpc.h":
            # disable the _Complex / <complex.h> code path
            text = text.replace("#if 1\n#define _MPC_HAVE_COMPLEX_H",
                                "#if 0\n#define _MPC_HAVE_COMPLEX_H")
        if fn == "mpc-impl.h":
            # MPC_ASSERT's failure path uses fprintf(stderr)+abort, neither of
            # which exists on the device; trap instead (works on host too).
            text = text.replace(
                '        fprintf (stderr, "%s:%d: MPC assertion failed: %s\\n",   \\\n'
                '                 __FILE__, __LINE__, #expr);                    \\\n'
                '        abort();                                                \\',
                '        __builtin_trap ();                                      \\')

        if fn in ("set_x.c", "set_x_x.c"):
            # These template files generate the whole mpc_set_<T>[_<T>] family in
            # one TU via token-pasting (mpfr_set_##type -> cu_mpfr_set_##type
            # after the cu_ rename).  The mpf_t/mpq_t/intmax variants have no
            # mpfr backend in a mini-gmp build, so provide trapping device stubs
            # (named to match the cu_ paste) so the file compiles; only the _d_d
            # and _fr_fr variants are used (mpc_set_fr_fr backs mpfr->mpfr copies,
            # e.g. mpc_atan), so set_fr delegates to cu_mpfr_set; the rest trap.
            stubs = (
                '#include "mpc-impl.h"\n'
                "#ifndef MPC_CUDA_MPFR_SET_STUBS\n#define MPC_CUDA_MPFR_SET_STUBS\n"
                "__host__ __device__ static inline int cu_mpfr_set_f (mpfr_ptr r, mpf_srcptr v, mpfr_rnd_t rnd)"
                " { (void) r; (void) v; (void) rnd; __builtin_trap (); return 0; }\n"
                "__host__ __device__ static inline int cu_mpfr_set_q (mpfr_ptr r, mpq_srcptr v, mpfr_rnd_t rnd)"
                " { (void) r; (void) v; (void) rnd; __builtin_trap (); return 0; }\n"
                "__host__ __device__ static inline int cu_mpfr_set_fr (mpfr_ptr r, mpfr_srcptr v, mpfr_rnd_t rnd)"
                " { return cu_mpfr_set (r, v, rnd); }\n"
                "__host__ __device__ static inline int cu_mpfr_set_uj (mpfr_ptr r, uintmax_t v, mpfr_rnd_t rnd)"
                " { (void) r; (void) v; (void) rnd; __builtin_trap (); return 0; }\n"
                "__host__ __device__ static inline int cu_mpfr_set_sj (mpfr_ptr r, intmax_t v, mpfr_rnd_t rnd)"
                " { (void) r; (void) v; (void) rnd; __builtin_trap (); return 0; }\n"
                "#endif\n")
            text = text.replace('#include "mpc-impl.h"', stubs, 1)

        text = M.attribute_const_data(text)
        if fn.endswith(".h"):
            text = attribute_mpc_prototypes(text)
            text = M.attribute_definitions(text)
        else:
            text = M.attribute_definitions(text)

        # rename exported MPC/MPFR/GMP symbols into the cu_ namespace
        text = cu_prefix.rename(text)

        with open(os.path.join(out, fn), "w") as f:
            f.write("/* CUDA-adapted from MPC by tools/cudafy_mpc.py. */\n" + text)

    print("cudafied MPC ->", out)


if __name__ == "__main__":
    main()
