#!/usr/bin/env python3
"""Generate the single-include, collision-free public headers for mpc_cuda.

The library's existing public headers (src/mpfr_cuda/mpfr.h, src/mpc_cuda/mpc.h
and include/mpc_cuda/cuda_minigmp.h) already expose *function* symbols under the
cu_ namespace (cu_mpfr_*, cu_mpc_*, cu_mpz_*), so they LINK alongside the system
libgmp/libmpfr/libmpc.  However they still use the SAME *type* names (mpfr_t,
mpc_t, __mpfr_struct ...), the SAME *enum constants* (MPFR_RNDN ...), the SAME
*public macros* (MPFR_PREC_MAX, MPC_RNDNN ...) and the SAME *include guards*
(__MPFR_H, __MPC_H) as the system headers.  That makes it impossible to include
both the system <mpc.h>/<mpfr.h>/<gmp.h> and this library in ONE translation
unit.

This script rewrites those public headers into a fully cu_/CU_-namespaced,
self-contained set:

    include/mpc_cuda/cu_gmp.h      (mini-gmp public API + arena + gmp stand-ins)
    include/mpc_cuda/cu_mpfr.h
    include/mpc_cuda/cu_mpc.h

so that everything a user touches is prefixed and cannot clash with the system
headers, in either include order, in the same .cu file.

The rewrite is purely textual (whole-word token renaming) and ABI-preserving:
function symbols keep their existing cu_ names and the renamed structs are
layout-identical to the ones the compiled objects use, so user code built
against these headers links unchanged against libmpc_cuda.

Re-run after regenerating the cudafied headers:   python3 tools/make_umbrella.py
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SRC_GMP  = os.path.join(ROOT, "include", "mpc_cuda", "cuda_minigmp.h")
SRC_SHIM = os.path.join(ROOT, "src", "mpc_cuda", "gmp.h")
SRC_MPFR = os.path.join(ROOT, "src", "mpfr_cuda", "mpfr.h")
SRC_MPC  = os.path.join(ROOT, "src", "mpc_cuda", "mpc.h")

OUT_DIR  = os.path.join(ROOT, "include", "mpc_cuda")
OUT_GMP  = os.path.join(OUT_DIR, "cu_gmp.h")
OUT_MPFR = os.path.join(OUT_DIR, "cu_mpfr.h")
OUT_MPC  = os.path.join(OUT_DIR, "cu_mpc.h")

# Lowercase bignum families whose every <family>_xxx token must become cu_<...>.
LOWER_FAMILIES = ("mpfr", "mpcr", "mpcb", "mpc", "mpz", "mpf", "mpq", "mpn")
# mp_<...>_t scalar typedefs from mini-gmp.
MP_TYPES = ("limb", "size", "bitcnt", "exp", "prec", "rnd", "ptr", "srcptr")

# Tokens we must NOT prefix: the arena/runtime helpers keep their real symbols.
def _keep(tok):
    return tok.startswith("mpc_cuda")


def rename(text):
    """Apply the whole-word namespace rewrite to one header's text."""

    # 1. Include guards (exact whole-word) -> unique, system-disjoint names.
    text = re.sub(r"\b__MINI_GMP_H__\b", "CU_MPC_CUDA_GMP_H", text)
    text = re.sub(r"\b__MPFR_H\b",       "CU_MPC_CUDA_MPFR_H", text)
    text = re.sub(r"\b__MPC_H\b",        "CU_MPC_CUDA_MPC_H", text)

    # 2. Double-underscore UPPERCASE internal macros (__MPFR_DECLSPEC,
    #    __MPFR_EXP_NAN, __GMP_DECLSPEC, __MPC_DECLSPEC ...) -> __CU_<...>.
    text = re.sub(r"\b__(MPFR|MPC|GMP)_([A-Za-z0-9_]+)\b",
                  lambda m: "__CU_%s_%s" % (m.group(1), m.group(2)), text)

    # 3. Double-underscore lowercase data/locals (__gmpfr_emin,
    #    __gmpfr_local_tab_, __mpfr_struct, __gmp_default_rounding_mode ...)
    #    -> cu___<...>.  Longer families first so the alternation is greedy.
    text = re.sub(r"\b__(gmpfr|gmpc|gmp|mpfr|mpcr|mpcb|mpc|mpz|mpf|mpq)_([A-Za-z0-9_]*)\b",
                  lambda m: "cu___%s_%s" % (m.group(1), m.group(2)), text)

    # 4. Lowercase family tokens  mpfr_x / mpc_x / mpz_x / mpn_x ...  -> cu_<...>.
    fam = "|".join(LOWER_FAMILIES)
    def _low(m):
        tok = m.group(0)
        return tok if _keep(tok) else "cu_" + tok
    text = re.sub(r"\b(?:%s)_[A-Za-z0-9_]+\b" % fam, _low, text)

    # 5. mini-gmp scalar typedefs / globals / memory hooks.
    text = re.sub(r"\bmp_(?:%s)_t\b" % "|".join(MP_TYPES),
                  lambda m: "cu_" + m.group(0), text)
    text = re.sub(r"\bmp_bits_per_limb\b", "cu_mp_bits_per_limb", text)
    text = re.sub(r"\bgmp_randstate_t\b",  "cu_gmp_randstate_t", text)
    text = re.sub(r"\b(mp_set_memory_functions|mp_get_memory_functions)\b",
                  lambda m: "cu_" + m.group(0), text)

    # 6. UPPERCASE public macros / enum constants
    #    (MPFR_RNDN, MPFR_PREC_MAX, GMP_NUMB_BITS, MPC_RNDNN ...) -> CU_<...>.
    text = re.sub(r"\b(MPFR|MPC|GMP)_([A-Za-z0-9_]+)\b",
                  lambda m: "CU_%s_%s" % (m.group(1), m.group(2)), text)

    return text


def strip_guard(text, guard):
    """Remove a header's own #ifndef/#define GUARD ... trailing #endif."""
    lines = text.splitlines(keepends=True)
    out = []
    removed_open = False
    for ln in lines:
        s = ln.strip()
        if not removed_open and (s == "#ifndef %s" % guard or
                                 s.startswith("#ifndef %s" % guard)):
            removed_open = True
            continue
        if removed_open and s.startswith("#define %s" % guard):
            continue
        out.append(ln)
    # drop the final #endif (matching the guard we removed)
    for i in range(len(out) - 1, -1, -1):
        if out[i].strip().startswith("#endif"):
            del out[i]
            break
    return "".join(out)


HEADER_NOTE = (
    "/* Auto-generated by tools/make_umbrella.py -- do NOT edit by hand.\n"
    " * Fully cu_/CU_-namespaced public header: every type, enum constant,\n"
    " * macro and include guard is prefixed so this coexists with the system\n"
    " * <gmp.h>/<mpfr.h>/<mpc.h> in the SAME translation unit. */\n")


def build_gmp():
    raw = open(SRC_GMP).read()
    body = rename(raw)
    body = strip_guard(body, "CU_MPC_CUDA_GMP_H")   # guard name AFTER rename
    # drop the <stddef.h> include (re-added below in a controlled spot)
    body = re.sub(r'^[ \t]*#\s*include\s*<stddef\.h>[ \t]*\n', "", body, flags=re.M)

    # gmp.h shim extras (randstate / mpf_t / mpq_t stand-ins, RODATA helper).
    shim = open(SRC_SHIM).read()
    shim = rename(shim)
    # the shim's first line is `#include "mpc_cuda/cuda_minigmp.h"` -> drop it
    shim = re.sub(r'^[ \t]*#\s*include\s*"[^"]*cuda_minigmp\.h"[ \t]*\n', "", shim, flags=re.M)

    out = []
    out.append(HEADER_NOTE)
    out.append("#ifndef CU_MPC_CUDA_GMP_H\n#define CU_MPC_CUDA_GMP_H\n")
    out.append("#include <stddef.h>\n")
    out.append("/* Build-time constants the cudafied library was compiled with; the public\n"
               " * headers need them too so users do not have to pass any -D flags. */\n")
    out.append("#ifndef CU_GMP_NUMB_BITS\n#define CU_GMP_NUMB_BITS 64\n#endif\n")
    out.append("#ifndef CU_MPFR_USE_MINI_GMP\n#define CU_MPFR_USE_MINI_GMP 1\n#endif\n")
    out.append(body)
    out.append("\n/* ---- gmp.h shim stand-ins (randstate / mpf_t / mpq_t) ---- */\n")
    out.append(shim)
    out.append("\n#endif /* CU_MPC_CUDA_GMP_H */\n")
    open(OUT_GMP, "w").write("".join(out))
    print("wrote", os.path.relpath(OUT_GMP, ROOT))


def build_mpfr():
    raw = open(SRC_MPFR).read()
    body = rename(raw)
    # redirect the gmp/mini-gmp include to our namespaced header
    body = re.sub(r'#\s*include\s*<\s*(?:mini-)?gmp\.h\s*>',
                  '#include "mpc_cuda/cu_gmp.h"', body)
    open(OUT_MPFR, "w").write(HEADER_NOTE + body)
    print("wrote", os.path.relpath(OUT_MPFR, ROOT))


def build_mpc():
    raw = open(SRC_MPC).read()
    body = rename(raw)
    body = re.sub(r'#\s*include\s*"\s*gmp\.h\s*"',
                  '#include "mpc_cuda/cu_gmp.h"', body)
    body = re.sub(r'#\s*include\s*"\s*mpfr\.h\s*"',
                  '#include "mpc_cuda/cu_mpfr.h"', body)
    open(OUT_MPC, "w").write(HEADER_NOTE + body)
    print("wrote", os.path.relpath(OUT_MPC, ROOT))


def main():
    for p in (SRC_GMP, SRC_SHIM, SRC_MPFR, SRC_MPC):
        if not os.path.exists(p):
            sys.exit("missing input header: %s" % p)
    build_gmp()
    build_mpfr()
    build_mpc()


if __name__ == "__main__":
    main()
