#!/usr/bin/env python3
"""cudafy_minigmp.py -- mechanically adapt mini-gmp for CUDA.

Reads the upstream mini-gmp.h / mini-gmp.c and emits CUDA-callable versions
in which every device-reachable function carries the `__host__ __device__`
qualifier.  The transformation is intentionally re-runnable so that future
mini-gmp updates can be re-adapted with one command.

Rules
-----
* Function *definitions* (GMP coding style: return type on its own line,
  function name flush-left, `(` on the same line) get `__host__ __device__`
  prepended to their return-type line.
* A function stays *host only* (no attribute) when
    - its name is in HOST_ONLY_NAMES, or
    - its body references a host-only token (realloc/FILE/stdio/ctype).
  `gmp_die` is the single exception: it is device-enabled and its body is
  rewritten so that on the device it traps instead of calling abort/fprintf.
* The dynamic allocation hooks (`gmp_alloc`/`gmp_free`/`gmp_realloc`) are
  rerouted to device-aware inline helpers (CUDA has malloc/free in device
  code but no realloc, so realloc is emulated with malloc+memcpy+free).
* Header prototypes of device-enabled functions get the attribute too, so the
  qualifier is visible at every call site inside a kernel.

Usage:  python3 tools/cudafy_minigmp.py <src_dir> <out_header> <out_source>
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cu_prefix

# Functions that must remain host-only (use realloc / stdio / ctype, or poke
# the host-side memory-function pointers).  Anything calling these from device
# code would fail to compile, so their callers stay host-only too.
# gmp_default_alloc/free are device-safe (malloc/free) and MUST be device-enabled
# because MPFR's heap temp allocator (mpfr_tmp_allocate) routes to them.
# gmp_default_realloc uses libc realloc -> rewritten device-safe below.
HOST_ONLY_NAMES = {
    "mp_set_memory_functions", "mp_get_memory_functions",
    "mpz_out_str",
    "mpz_set_str", "mpz_init_set_str",   # use isspace()/ctype
}

# Identifiers whose presence in a function body forces host-only compilation.
# Matched on word boundaries so that e.g. `gmp_realloc` is NOT mistaken for the
# libc `realloc`.
HOST_ONLY_TOKENS = ("realloc", "FILE", "fwrite", "fprintf", "fputs",
                    "isspace", "fscanf", "fread", "fputc", "getc",
                    "stdin", "stdout", "stderr")
HOST_ONLY_TOKEN_RE = re.compile(
    r"\b(?:" + "|".join(HOST_ONLY_TOKENS) + r")\b")

C_KEYWORDS = {"if", "for", "while", "switch", "return", "sizeof", "do",
              "else", "typedef", "struct", "union", "enum", "static",
              "const", "void", "int", "long", "unsigned", "double",
              "char", "extern", "signed", "float", "register", "volatile"}

# In a header prototype the first token is the *return type* (often void/int),
# so we only reject control-flow keywords and declaration markers here.
HEADER_REJECT_FIRST = {"if", "for", "while", "switch", "return", "sizeof",
                       "do", "else", "typedef", "extern", "struct", "union",
                       "enum"}

NAME_RE = re.compile(r"^([A-Za-z_]\w*)\s*\(")
PROTO_NAME_RE = re.compile(r"([A-Za-z_]\w*)\s*\(")

ATTR = "__host__ __device__"


def is_type_line(s):
    t = s.strip()
    if not t:
        return False
    if t[0] in "#*/}{":
        return False
    if t.endswith((";", ",", "{", "}")):
        return False
    if "(" in t or ")" in t:
        return False
    return re.match(r"^[A-Za-z_][\w \t\*]*$", t) is not None


ARENA_DECLS = """
/* ---- per-thread bump arena (see tools/cudafy_minigmp.py preamble) ---- */
extern __managed__ char   *mpc_cuda_arena_base;
extern __managed__ size_t  mpc_cuda_arena_slab;
extern __managed__ size_t *mpc_cuda_arena_top;
__host__ __device__ void *mpc_cuda_dev_alloc (size_t);
__host__ __device__ void  mpc_cuda_dev_free (void *);
__host__ __device__ void *mpc_cuda_dev_realloc (void *, size_t, size_t);
__device__ void mpc_cuda_arena_reset (void);
"""


def transform_header_extra(text):
    """Header-side tweaks beyond prototype attribution."""
    text = text.replace(
        "extern const int mp_bits_per_limb;",
        "extern __device__ __managed__ int mp_bits_per_limb;")
    # arena declarations go *outside* the extern "C" block (the definitions in
    # the .cu are C++-linkage), just before the include guard's final #endif.
    text = text.replace(
        "#endif /* __MINI_GMP_H__ */",
        ARENA_DECLS + "\n#endif /* __MINI_GMP_H__ */")
    return text


def find_body_span(lines, start):
    """Return (open, close) line indices of the {...} body following a name
    line at index `start`, or (None, None) for a forward declaration."""
    i = start
    n = len(lines)
    # walk forward to the first '{' (start of body) or ';' (declaration only)
    depth_paren = 0
    while i < n:
        ln = lines[i]
        # detect a declaration that ends before any body
        if "{" in ln:
            opn = i
            depth = 0
            j = i
            while j < n:
                depth += lines[j].count("{") - lines[j].count("}")
                if depth <= 0 and "{" in "".join(lines[opn:j + 1]):
                    return opn, j
                j += 1
            return opn, n - 1
        if ln.rstrip().endswith(";") and depth_paren == 0:
            # closing of a forward declaration
            return None, None
        depth_paren += ln.count("(") - ln.count(")")
        i += 1
    return None, None


def transform_source(text):
    lines = text.split("\n")
    out = []
    i = 0
    n = len(lines)
    # We rewrite gmp_alloc/free/realloc macros and gmp_die body via direct
    # textual substitution first, on the whole buffer.
    insert_before = {}    # line index -> attribute to insert

    while i < n:
        line = lines[i]
        m = NAME_RE.match(line)
        if m and line[:1] not in " \t" and m.group(1) not in C_KEYWORDS:
            name = m.group(1)
            if i > 0 and is_type_line(lines[i - 1]):
                opn, close = find_body_span(lines, i)
                host_only = name in HOST_ONLY_NAMES
                # gmp_die and gmp_default_realloc are device-enabled with
                # rewritten bodies (see below); skip the host-only token scan.
                force_device = name in ("gmp_die", "gmp_default_realloc")
                if opn is not None and not host_only and not force_device:
                    body = "\n".join(lines[opn:close + 1])
                    if HOST_ONLY_TOKEN_RE.search(body):
                        host_only = True
                if not host_only:
                    insert_before[i - 1] = ATTR
        i += 1

    for idx, line in enumerate(lines):
        if idx in insert_before:
            out.append(insert_before[idx])
        out.append(line)
    text = "\n".join(out)

    # --- reroute allocation macros -------------------------------------------
    text = text.replace(
        "#define gmp_alloc(size) ((*gmp_allocate_func)((size)))",
        "#define gmp_alloc(size) mg_cuda_alloc(size)")
    text = text.replace(
        "#define gmp_free(p, size) ((*gmp_free_func) ((p), (size)))",
        "#define gmp_free(p, size) mg_cuda_free((p))")
    text = text.replace(
        "#define gmp_realloc(ptr, old_size, size) ((*gmp_reallocate_func)(ptr, old_size, size))",
        "#define gmp_realloc(ptr, old_size, size) mg_cuda_realloc((ptr), (old_size), (size))")

    # --- device-safe gmp_default_realloc body --------------------------------
    text = text.replace(
        "  p = realloc (old, new_size);",
        "  p = mg_cuda_realloc (old, unused_old_size, new_size);")

    # --- device-safe gmp_die body --------------------------------------------
    text = text.replace(
        '  fprintf (stderr, "%s\\n", msg);\n  abort();',
        "#ifdef __CUDA_ARCH__\n  (void) msg;\n  __trap ();\n"
        "#else\n  fprintf (stderr, \"%s\\n\", msg);\n  abort ();\n#endif")

    # --- inject the CUDA adaptation preamble ---------------------------------
    preamble = r'''
/* ===================================================================== *
 *  CUDA adaptation preamble (auto-injected by tools/cudafy_minigmp.py)   *
 * ===================================================================== */
#include <cuda_runtime.h>

__host__ __device__ static void gmp_die (const char *msg);

/* --------------------------------------------------------------------- *
 *  Optional per-thread bump arena.                                      *
 *  Multiple-precision code allocates many small limb buffers per op; on *
 *  the GPU that device malloc/free churn dominates.  When an arena is    *
 *  installed (mpc_cuda_arena_base != NULL) device allocations are served *
 *  by bumping a per-thread slab and free is a no-op; mpc_cuda_arena_reset *
 *  reclaims a thread's slab (call it per work item).  With no arena      *
 *  installed, allocation falls back to the device heap (malloc/free).    *
 *  These hooks back BOTH mini-gmp and (via mpfr-gmp.c) MPFR/MPC.         */
__managed__ char   *mpc_cuda_arena_base = 0;  /* base of nthreads*slab bytes */
__managed__ size_t  mpc_cuda_arena_slab = 0;  /* bytes per thread            */
__managed__ size_t *mpc_cuda_arena_top  = 0;  /* per-thread bump offset      */

__device__ static inline unsigned long
mpc_cuda_tid (void)
{
  unsigned long bid = (unsigned long) blockIdx.x
      + (unsigned long) gridDim.x * blockIdx.y
      + (unsigned long) gridDim.x * gridDim.y * blockIdx.z;
  unsigned long bsz = (unsigned long) blockDim.x * blockDim.y * blockDim.z;
  unsigned long tib = (unsigned long) threadIdx.x
      + (unsigned long) blockDim.x * threadIdx.y
      + (unsigned long) blockDim.x * blockDim.y * threadIdx.z;
  return bid * bsz + tib;
}

__host__ __device__ void *
mpc_cuda_dev_alloc (size_t size)
{
#ifdef __CUDA_ARCH__
  if (mpc_cuda_arena_base)
    {
      unsigned long tid = mpc_cuda_tid ();
      size = (size + 15UL) & ~15UL;          /* 16-byte align */
      size_t off = mpc_cuda_arena_top[tid];
      if (off + size <= mpc_cuda_arena_slab)
        {
          char *p = mpc_cuda_arena_base + tid * mpc_cuda_arena_slab + off;
          mpc_cuda_arena_top[tid] = off + size;
          return p;
        }
      /* slab exhausted: fall back to the device heap */
    }
#endif
  return malloc (size);
}

__host__ __device__ void
mpc_cuda_dev_free (void *p)
{
#ifdef __CUDA_ARCH__
  if (mpc_cuda_arena_base)
    {
      unsigned long tid = mpc_cuda_tid ();
      char *base = mpc_cuda_arena_base + tid * mpc_cuda_arena_slab;
      if ((char *) p >= base && (char *) p < base + mpc_cuda_arena_slab)
        return;                              /* arena memory: freed by reset */
    }
#endif
  if (p) free (p);
}

__host__ __device__ void *
mpc_cuda_dev_realloc (void *old, size_t old_size, size_t new_size)
{
#ifdef __CUDA_ARCH__
  if (mpc_cuda_arena_base)
    {
      if (new_size == 0) { mpc_cuda_dev_free (old); return 0; }
      {
        void *np = mpc_cuda_dev_alloc (new_size);
        if (np && old)
          {
            size_t c = old_size < new_size ? old_size : new_size;
            memcpy (np, old, c);
            mpc_cuda_dev_free (old);
          }
        return np;
      }
    }
  if (new_size == 0) { if (old) free (old); return 0; }
  {
    void *np = malloc (new_size);
    if (np && old)
      { size_t c = old_size < new_size ? old_size : new_size; memcpy (np, old, c); free (old); }
    return np;
  }
#else
  void *p = realloc (old, new_size);
  (void) old_size;
  return p;
#endif
}

/* Reclaim the calling thread's arena slab (no-op without an arena). */
__device__ void
mpc_cuda_arena_reset (void)
{
  if (mpc_cuda_arena_base)
    mpc_cuda_arena_top[mpc_cuda_tid ()] = 0;
}

/* mini-gmp allocation hooks, routed through the arena allocator. */
__host__ __device__ static inline void *
mg_cuda_alloc (size_t size)
{
  void *p = mpc_cuda_dev_alloc (size);
  if (!p)
    gmp_die ("mg_cuda_alloc: out of memory");
  return p;
}

__host__ __device__ static inline void
mg_cuda_free (void *p)
{
  mpc_cuda_dev_free (p);
}

__host__ __device__ static inline void *
mg_cuda_realloc (void *old, size_t old_size, size_t new_size)
{
  void *p = mpc_cuda_dev_realloc (old, old_size, new_size);
  if (!p && new_size)
    gmp_die ("mg_cuda_realloc: out of memory");
  return p;
}
'''
    marker = '#include "mini-gmp.h"'
    text = text.replace(marker, '#include "cuda_minigmp.h"\n' + preamble, 1)

    # mp_bits_per_limb is referenced from device code (incl. MPFR); make it a
    # unified (host+device) variable instead of a host-only extern const.
    text = text.replace(
        "const int mp_bits_per_limb = GMP_LIMB_BITS;",
        "__device__ __managed__ int mp_bits_per_limb = GMP_LIMB_BITS;")
    return text


def transform_header(text):
    lines = text.split("\n")
    out = []
    in_comment = False
    ended = True
    for line in lines:
        stripped = line.strip()
        emit = line
        # crude block-comment tracking
        was_in_comment = in_comment
        if in_comment:
            if "*/" in stripped:
                in_comment = False
            out.append(emit)
            ended = True
            continue
        if stripped.startswith("/*") and "*/" not in stripped:
            in_comment = True
            out.append(emit)
            ended = True
            continue
        if (stripped == "" or stripped.startswith("#") or
                stripped.startswith("//") or stripped.startswith("/*") or
                stripped.startswith("*")):
            out.append(emit)
            ended = True
            continue

        start = ended
        if (start and re.match(r"^[A-Za-z_]", line) and "(" in stripped):
            first = stripped.split("(")[0].split()
            firsttok = first[0] if first else ""
            pm = PROTO_NAME_RE.search(stripped)
            pname = pm.group(1) if pm else ""
            if (firsttok not in HEADER_REJECT_FIRST and
                    pname not in HOST_ONLY_NAMES and
                    (stripped.endswith(";") or stripped.endswith(",") or
                     stripped.endswith("(")) and
                    "typedef" not in stripped):
                emit = ATTR + " " + line
        out.append(emit)
        ended = stripped.endswith((";", "{", "}"))
    return "\n".join(out)


def _dual_body(text, sig, device_body):
    """Replace the body of the function whose definition line contains `sig`
    with a dual-path body: PTX/intrinsic fast path under __CUDA_ARCH__ (64-bit
    limbs), the original portable code otherwise.  Exact integer semantics are
    unchanged, so host and device results stay bit-identical."""
    i = text.index(sig)
    b = text.index("{", i)
    depth, j = 0, b
    while True:
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    orig = text[b + 1:j]
    new = ("{\n#if defined(__CUDA_ARCH__) && defined(GMP_NUMB_BITS) && GMP_NUMB_BITS == 64\n"
           + device_body
           + "\n#else /* portable */\n" + orig + "\n#endif\n}")
    return text[:b] + new + text[j + 1:]


# PTX carry-chain fast paths for the hot mpn primitives.  E.g. one
# mad.lo.cc.u64/madc.hi.u64 pair replaces the ~20-instruction 32-bit-split
# multiply of the generic gmp_umul_ppmm.
FASTPATH_MACROS = r'''
/* ---- CUDA device fast paths (injected by cudafy_minigmp.py) ------------ */
#if defined(__CUDA_ARCH__) && defined(GMP_NUMB_BITS) && GMP_NUMB_BITS == 64
#undef gmp_umul_ppmm
#define gmp_umul_ppmm(w1, w0, u, v)                                     \
  do {                                                                  \
    mp_limb_t __cu_u = (u), __cu_v = (v);                               \
    (w0) = __cu_u * __cu_v;                                             \
    (w1) = (mp_limb_t) __umul64hi ((unsigned long long) __cu_u,         \
                                   (unsigned long long) __cu_v);        \
  } while (0)
#undef gmp_clz
#define gmp_clz(count, x)                                               \
  do { (count) = __clzll ((unsigned long long) (x)); } while (0)
#undef gmp_ctz
#define gmp_ctz(count, x)                                               \
  do { (count) = __ffsll ((unsigned long long) (x)) - 1; } while (0)
#endif

'''

BODY_ADD_N = r'''  mp_size_t i;
  mp_limb_t cy = 0;
  for (i = 0; i < n; i++)
    {
      mp_limb_t r, c;
      __asm__ ("add.cc.u64 %0, %2, %3;\n\t"
               "addc.u64   %1, 0, 0;\n\t"
               "add.cc.u64 %0, %0, %4;\n\t"
               "addc.u64   %1, %1, 0;"
               : "=&l" (r), "=&l" (c)
               : "l" (ap[i]), "l" (bp[i]), "l" (cy));
      rp[i] = r;
      cy = c;
    }
  return cy;'''

BODY_SUB_N = r'''  mp_size_t i;
  mp_limb_t cy = 0;
  for (i = 0; i < n; i++)
    {
      mp_limb_t r, m1, m2;
      __asm__ ("sub.cc.u64 %0, %3, %4;\n\t"
               "subc.u64   %1, 0, 0;\n\t"
               "sub.cc.u64 %0, %0, %5;\n\t"
               "subc.u64   %2, 0, 0;"
               : "=&l" (r), "=&l" (m1), "=&l" (m2)
               : "l" (ap[i]), "l" (bp[i]), "l" (cy));
      rp[i] = r;
      cy = (mp_limb_t) 0 - (m1 + m2);   /* the two borrows are exclusive */
    }
  return cy;'''

BODY_MUL_1 = r'''  mp_limb_t cl = 0;
  do
    {
      mp_limb_t ul = *up++, lo, hi;
      __asm__ ("mad.lo.cc.u64 %0, %2, %3, %4;\n\t"
               "madc.hi.u64   %1, %2, %3, 0;"
               : "=&l" (lo), "=&l" (hi)
               : "l" (ul), "l" (vl), "l" (cl));
      *rp++ = lo;
      cl = hi;
    }
  while (--n != 0);
  return cl;'''

BODY_ADDMUL_1 = r'''  mp_limb_t cl = 0;
  do
    {
      mp_limb_t ul = *up++, rl = *rp, lo, hi;
      __asm__ ("mad.lo.cc.u64 %0, %2, %3, %4;\n\t"
               "madc.hi.u64   %1, %2, %3, 0;\n\t"
               "add.cc.u64    %0, %0, %5;\n\t"
               "addc.u64      %1, %1, 0;"
               : "=&l" (lo), "=&l" (hi)
               : "l" (ul), "l" (vl), "l" (cl), "l" (rl));
      *rp++ = lo;
      cl = hi;
    }
  while (--n != 0);
  return cl;'''

BODY_SUBMUL_1 = r'''  mp_limb_t cl = 0;
  do
    {
      mp_limb_t ul = *up++, rl = *rp, lo, hi, m;
      __asm__ ("mad.lo.cc.u64 %0, %3, %4, %5;\n\t"
               "madc.hi.u64   %1, %3, %4, 0;\n\t"
               "sub.cc.u64    %0, %6, %0;\n\t"
               "subc.u64      %2, 0, 0;"
               : "=&l" (lo), "=&l" (hi), "=&l" (m)
               : "l" (ul), "l" (vl), "l" (cl), "l" (rl));
      *rp++ = lo;
      cl = hi - m;                      /* hi + borrow */
    }
  while (--n != 0);
  return cl;'''


def inject_device_fastpaths(text):
    """Inject the PTX fast paths into the final (cu_-renamed) source text."""
    anchor = "#define gmp_udiv_qrnnd_preinv"
    if anchor not in text:
        raise RuntimeError("fastpath injection: udiv anchor not found")
    text = text.replace(anchor, FASTPATH_MACROS + anchor, 1)
    text = _dual_body(text, "cu_mpn_add_n (mp_ptr rp, mp_srcptr ap, mp_srcptr bp, mp_size_t n)", BODY_ADD_N)
    text = _dual_body(text, "cu_mpn_sub_n (mp_ptr rp, mp_srcptr ap, mp_srcptr bp, mp_size_t n)", BODY_SUB_N)
    text = _dual_body(text, "cu_mpn_mul_1 (mp_ptr rp, mp_srcptr up, mp_size_t n, mp_limb_t vl)", BODY_MUL_1)
    text = _dual_body(text, "cu_mpn_addmul_1 (mp_ptr rp, mp_srcptr up, mp_size_t n, mp_limb_t vl)", BODY_ADDMUL_1)
    text = _dual_body(text, "cu_mpn_submul_1 (mp_ptr rp, mp_srcptr up, mp_size_t n, mp_limb_t vl)", BODY_SUBMUL_1)
    return text


def main():
    if len(sys.argv) != 4:
        sys.exit("usage: cudafy_minigmp.py <src_dir> <out_header> <out_source>")
    src_dir, out_h, out_c = sys.argv[1:]
    with open(os.path.join(src_dir, "mini-gmp.h")) as f:
        h = f.read()
    with open(os.path.join(src_dir, "mini-gmp.c")) as f:
        c = f.read()

    h2 = transform_header_extra(transform_header(h))
    # rename exported GMP symbols into the cu_ namespace (coexist with libgmp)
    h2 = cu_prefix.rename(h2)
    # guard so the include name matches and add a banner
    banner = ("/* Auto-generated from mini-gmp.h by tools/cudafy_minigmp.py."
              "  Do not edit by hand. */\n")
    with open(out_h, "w") as f:
        f.write(banner + h2)

    c2 = cu_prefix.rename(transform_source(c))
    c2 = inject_device_fastpaths(c2)
    banner_c = ("/* Auto-generated from mini-gmp.c by tools/cudafy_minigmp.py."
                "  Do not edit by hand. */\n")
    with open(out_c, "w") as f:
        f.write(banner_c + c2)
    print("wrote", out_h, "and", out_c)


if __name__ == "__main__":
    main()
