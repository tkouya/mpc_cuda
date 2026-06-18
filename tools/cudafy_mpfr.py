#!/usr/bin/env python3
"""cudafy_mpfr.py -- mechanically adapt MPFR (mpfr-4.2.2) for CUDA.

Copies the configured MPFR source tree into an output directory and rewrites it
so the library is callable from CUDA kernels:

* Function *definitions* (GMP coding style) get `__host__ __device__`.
* Function *prototypes* declared with `__MPFR_DECLSPEC` get `__host__
  __device__` (but NOT the `extern const` table declarations that also use
  `__MPFR_DECLSPEC`).
* The global thread-vars (`__gmpfr_emin/emax/flags`, default prec/rnd, constant
  caches) become `__device__ __managed__` so they are visible from host and
  device.  This relies on the build being configured `--disable-thread-safe`
  (so `MPFR_THREAD_ATTR`/`MPFR_THREAD_VAR` reduce to plain globals).
* MPFR's internal `#include "mini-gmp.h"` is redirected to the CUDA-ported
  mini-gmp via a shim placed in the output directory.
* Functions touching host-only facilities (stdio/varargs/errno/locale/FE
  rounding) stay host-only, detected by a word-boundary token scan.  The
  assertion-failure helper is device-enabled (traps) like mini-gmp's gmp_die.

The MPFR build for CUDA must be configured:
    --with-mini-gmp=<dir> --disable-thread-safe
    --disable-decimal-float --disable-float128
and compiled with the defines in tools/mpfr_cuda_defs.txt
(NOTE: never -DMPFR_HAVE_NORETURN, and DROP -DHAVE_ALLOCA so that
MPFR_ALLOCA_MAX==0 and temporaries use the heap allocator, not alloca()).

Usage: python3 tools/cudafy_mpfr.py <mpfr_src_dir> <out_dir>
"""
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cu_prefix

# Functions are __host__ __device__ (same as the mini-gmp layer): the device
# instantiation is what runs in kernels, the host one keeps MPFR's pervasive
# __builtin_constant_p() a benign warning rather than a hard error (it is fatal
# in a device-only function).  Read-only lookup tables therefore need to exist
# in BOTH address spaces: MPFR_RODATA expands to __device__ in the device
# compilation pass and to nothing in the host pass (see the shim header), so a
# single definition serves both.  Mutable global state uses __managed__.
ATTR = "__host__ __device__"

C_KEYWORDS = {"if", "for", "while", "switch", "return", "sizeof", "do",
              "else", "typedef", "struct", "union", "enum", "static",
              "const", "void", "int", "long", "unsigned", "double",
              "char", "extern", "signed", "float", "register", "volatile"}

NAME_RE = re.compile(r"^([A-Za-z_]\w*)\s*\(")

# Host-only identifiers: presence in a function body keeps it host-only.
HOST_ONLY_TOKENS = ("printf", "fprintf", "vfprintf", "vsprintf", "vsnprintf",
                    "sprintf", "snprintf", "asprintf", "vasprintf",
                    "fwrite", "fread", "fputc", "fputs", "fgetc", "getc",
                    "putc", "fopen", "fclose", "fflush", "FILE",
                    "stdin", "stdout", "stderr", "errno", "setlocale",
                    "localeconv", "fesetround", "fegetround", "fexcept",
                    "getenv", "perror", "va_start", "va_arg", "va_copy",
                    "longjmp", "setjmp", "clock", "gettimeofday", "getrusage",
                    "fscanf", "sscanf", "scanf", "fdopen", "tmpfile",
                    "strcpy", "strncpy", "strcat", "strncat")
HOST_ONLY_TOKEN_RE = re.compile(
    r"\b(?:" + "|".join(HOST_ONLY_TOKENS) + r")\b")

# Names that must stay host-only regardless (entry points we never call on the
# device and that pull in host-only machinery transitively).
HOST_ONLY_NAMES = set()

# Names force-enabled on device even though their body trips the token scan.
FORCE_DEVICE_NAMES = {"mpfr_assert_fail", "mpfr_abort_prec_max"}


def is_type_line(s):
    t = s.strip()
    if not t:
        return False
    if t[0] in "#*/}{":
        return False
    if t.endswith((";", ",", "{", "}", ":")):
        return False
    if "(" in t or ")" in t:
        return False
    return re.match(r"^[A-Za-z_][\w \t\*]*$", t) is not None


def find_body_span(lines, start):
    """(open,close) brace line indices of the body following name line `start`,
    or (None,None) for a forward declaration."""
    i, n = start, len(lines)
    depth_paren = 0
    while i < n:
        ln = lines[i]
        if "{" in ln and depth_paren + ln.count("(") - ln.count(")") <= 0:
            opn, depth, j = i, 0, i
            while j < n:
                depth += lines[j].count("{") - lines[j].count("}")
                if depth <= 0 and "{" in "".join(lines[opn:j + 1]):
                    return opn, j
                j += 1
            return opn, n - 1
        if ln.rstrip().endswith(";") and depth_paren == 0:
            return None, None
        depth_paren += ln.count("(") - ln.count(")")
        i += 1
    return None, None


def attribute_definitions(text):
    """Prepend ATTR to GMP-style function definitions (and forward decls)."""
    lines = text.split("\n")
    n = len(lines)
    insert_before = {}
    for i in range(1, n):
        line = lines[i]
        m = NAME_RE.match(line)
        if not m or line[:1] in " \t" or m.group(1) in C_KEYWORDS:
            continue
        if not is_type_line(lines[i - 1]):
            continue
        name = m.group(1)
        opn, close = find_body_span(lines, i)
        host_only = name in HOST_ONLY_NAMES
        if (opn is not None and not host_only
                and name not in FORCE_DEVICE_NAMES):
            body = "\n".join(lines[opn:close + 1])
            if HOST_ONLY_TOKEN_RE.search(body):
                host_only = True
        if not host_only:
            insert_before[i - 1] = True
    out = []
    for idx, line in enumerate(lines):
        if idx in insert_before:
            out.append(ATTR)
        out.append(line)
    return "\n".join(out)


def attribute_prototypes(text):
    """Prepend ATTR to __MPFR_DECLSPEC function prototypes, but not to the
    `extern const ...` table declarations that also use __MPFR_DECLSPEC."""
    out = []
    for line in text.split("\n"):
        s = line.lstrip()
        if s.startswith("__MPFR_DECLSPEC") and "extern" not in s:
            out.append(ATTR + " " + line)
        else:
            out.append(line)
    return "\n".join(out)


def patch_ieee_floats(text):
    """ieee_floats.h: the +/-Inf doubles are file-scope `static const union`s
    that the device cannot reference.  Use compiler builtins instead (valid on
    both host and device under nvcc)."""
    text = text.replace("#define MPFR_DBL_INFP  (dbl_infp.d)",
                        "#define MPFR_DBL_INFP  (__builtin_inf())")
    text = text.replace("#define MPFR_DBL_INFM  (dbl_infm.d)",
                        "#define MPFR_DBL_INFM  (-__builtin_inf())")
    return text


def patch_assert_fail(text):
    """mpfr_assert_fail: trap on the device instead of fprintf+abort."""
    old = ('  if (filename != NULL && filename[0] != \'\\0\')\n'
           '    {\n'
           '      fprintf (stderr, "%s:", filename);\n'
           '      if (linenum != -1)\n'
           '        fprintf (stderr, "%d: ", linenum);\n'
           '    }\n'
           '  fprintf (stderr, "MPFR assertion failed: %s\\n", expr);\n'
           '  abort();')
    new = ('#ifdef __CUDA_ARCH__\n'
           '  (void) filename; (void) linenum; (void) expr;\n'
           '  __trap ();\n'
           '#else\n'
           '  if (filename != NULL && filename[0] != \'\\0\')\n'
           '    {\n'
           '      fprintf (stderr, "%s:", filename);\n'
           '      if (linenum != -1)\n'
           '        fprintf (stderr, "%d: ", linenum);\n'
           '    }\n'
           '  fprintf (stderr, "MPFR assertion failed: %s\\n", expr);\n'
           '  abort();\n'
           '#endif')
    if old not in text:
        raise SystemExit("patch_assert_fail: body not found (upstream changed)")
    return text.replace(old, new)


def patch_mpfr_gmp_alloc(text):
    """mpfr-gmp.c: mpfr_allocate/reallocate/free_func route through
    mp_get_memory_functions(), which returns HOST function pointers (mini-gmp's
    allocation hooks).  Calling those host pointers on the device yields garbage.
    On the device, call the device heap (malloc/free) directly instead."""
    text = text.replace(
        "  mp_get_memory_functions (&allocate_func, &reallocate_func, &free_func);\n"
        "  return (*allocate_func) (alloc_size);",
        "#ifdef __CUDA_ARCH__\n"
        "  (void) allocate_func; (void) reallocate_func; (void) free_func;\n"
        "  return mpc_cuda_dev_alloc (alloc_size);\n"
        "#else\n"
        "  mp_get_memory_functions (&allocate_func, &reallocate_func, &free_func);\n"
        "  return (*allocate_func) (alloc_size);\n"
        "#endif")
    text = text.replace(
        "  mp_get_memory_functions (&allocate_func, &reallocate_func, &free_func);\n"
        "  return (*reallocate_func) (ptr, old_size, new_size);",
        "#ifdef __CUDA_ARCH__\n"
        "  (void) allocate_func; (void) reallocate_func; (void) free_func;\n"
        "  return mpc_cuda_dev_realloc (ptr, old_size, new_size);\n"
        "#else\n"
        "  mp_get_memory_functions (&allocate_func, &reallocate_func, &free_func);\n"
        "  return (*reallocate_func) (ptr, old_size, new_size);\n"
        "#endif")
    text = text.replace(
        "  mp_get_memory_functions (&allocate_func, &reallocate_func, &free_func);\n"
        "  (*free_func) (ptr, size);",
        "#ifdef __CUDA_ARCH__\n"
        "  (void) allocate_func; (void) reallocate_func; (void) free_func;\n"
        "  mpc_cuda_dev_free (ptr); (void) size;\n"
        "#else\n"
        "  mp_get_memory_functions (&allocate_func, &reallocate_func, &free_func);\n"
        "  (*free_func) (ptr, size);\n"
        "#endif")
    return text


DIV_CUDA_FUNC = r'''
/* CUDA: device-correct mpfr_div.  MPFR's generic division path (precision >= 3
   limbs) miscompiles on the device (returns inf) -- root-caused but the exact
   construct resisted isolation (see docs/phase2-mpfr.md).  This is a
   self-contained, correctly-rounded replacement built only on mpz primitives,
   which ARE verified correct on the device.  Used on the device only; the host
   keeps the upstream code. */
__host__ __device__ static int
mpfr_div_cuda (mpfr_ptr q, mpfr_srcptr u, mpfr_srcptr v, mpfr_rnd_t rnd)
{
  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (u) || MPFR_IS_SINGULAR (v)))
    {
      int sgn = MPFR_MULT_SIGN (MPFR_SIGN (u), MPFR_SIGN (v));
      if (MPFR_IS_NAN (u) || MPFR_IS_NAN (v)) { MPFR_SET_NAN (q); MPFR_RET_NAN; }
      if (MPFR_IS_INF (u))
        {
          if (MPFR_IS_INF (v)) { MPFR_SET_NAN (q); MPFR_RET_NAN; }
          MPFR_SET_INF (q); MPFR_SET_SIGN (q, sgn); MPFR_RET (0);
        }
      if (MPFR_IS_INF (v)) { MPFR_SET_ZERO (q); MPFR_SET_SIGN (q, sgn); MPFR_RET (0); }
      if (MPFR_IS_ZERO (u))
        {
          if (MPFR_IS_ZERO (v)) { MPFR_SET_NAN (q); MPFR_RET_NAN; }
          MPFR_SET_ZERO (q); MPFR_SET_SIGN (q, sgn); MPFR_RET (0);
        }
      MPFR_SET_INF (q); MPFR_SET_SIGN (q, sgn); MPFR_SET_DIVBY0 (); MPFR_RET (0);
    }
  {
    mpfr_prec_t p = MPFR_PREC (q);
    mp_size_t su = MPFR_LIMB_SIZE (u);
    mp_size_t sv = MPFR_LIMB_SIZE (v);
    mp_size_t qsize = MPFR_LIMB_SIZE (q);
    mpfr_limb_srcptr up = MPFR_MANT (u);
    mpfr_limb_srcptr vp = MPFR_MANT (v);
    int sgn = MPFR_MULT_SIGN (MPFR_SIGN (u), MPFR_SIGN (v));
    mpfr_exp_t E0 = MPFR_GET_EXP (u) - MPFR_GET_EXP (v)
                  - (mpfr_exp_t) (su - sv) * GMP_NUMB_BITS;
    mp_bitcnt_t shift = (mp_bitcnt_t) p + 2
                      + (mp_bitcnt_t) sv * GMP_NUMB_BITS;
    mpz_t MU, MV, Q, R, num, low;
    size_t nbits; mp_bitcnt_t drop;
    int round_bit, sticky, inc, t; mpfr_exp_t EXP; mp_size_t i;

    mpz_init (MU); mpz_init (MV); mpz_init (Q);
    mpz_init (R); mpz_init (num); mpz_init (low);
    mpz_import (MU, su, -1, sizeof (mp_limb_t), 0, 0, up);
    mpz_import (MV, sv, -1, sizeof (mp_limb_t), 0, 0, vp);

    mpz_mul_2exp (num, MU, shift);
    mpz_tdiv_qr (Q, R, num, MV);             /* Q = floor(MU<<shift / MV) */
    nbits = mpz_sizeinbase (Q, 2);
    drop = (mp_bitcnt_t) ((mp_size_t) nbits - (mp_size_t) p); /* >= 2 */

    round_bit = mpz_tstbit (Q, drop - 1);
    sticky = (mpz_sgn (R) != 0);
    mpz_fdiv_r_2exp (low, Q, drop - 1);
    if (mpz_sgn (low) != 0) sticky = 1;
    mpz_tdiv_q_2exp (Q, Q, drop);            /* Q -> p-bit mantissa */
    EXP = E0 + (mpfr_exp_t) nbits - (mpfr_exp_t) shift;

    inc = 0;
    if (rnd == MPFR_RNDN)
      inc = (round_bit && (sticky || mpz_tstbit (Q, 0)));
    else if (rnd == MPFR_RNDA)
      inc = (round_bit || sticky);
    else if (rnd == MPFR_RNDU)
      inc = ((round_bit || sticky) && sgn > 0);
    else if (rnd == MPFR_RNDD)
      inc = ((round_bit || sticky) && sgn < 0);
    /* MPFR_RNDZ: inc = 0 */
    t = (round_bit || sticky) ? (inc ? sgn : -sgn) : 0;

    if (inc)
      {
        mpz_add_ui (Q, Q, 1);
        if (mpz_sizeinbase (Q, 2) > (size_t) p)
          { mpz_tdiv_q_2exp (Q, Q, 1); EXP += 1; }
      }

    mpz_mul_2exp (Q, Q,
      (mp_bitcnt_t) ((mp_size_t) qsize * GMP_NUMB_BITS - (mp_size_t) p));
    {
      mpfr_limb_ptr qp = MPFR_MANT (q);
      for (i = 0; i < qsize; i++)
        qp[i] = mpz_getlimbn (Q, i);
    }
    MPFR_SET_EXP (q, EXP);
    MPFR_SET_SIGN (q, sgn);

    mpz_clear (MU); mpz_clear (MV); mpz_clear (Q);
    mpz_clear (R); mpz_clear (num); mpz_clear (low);

    return mpfr_check_range (q, t, rnd);
  }
}
'''


def patch_div_cuda(text):
    """div.c: route the public mpfr_div to a device-correct mpz-based version
    (the upstream generic path miscompiles on the device)."""
    # define the helper after the includes (all MPFR macros / mpz are in scope)
    inc = '#include "invert_limb.h"'
    if inc not in text:
        raise SystemExit("patch_div_cuda: include anchor not found")
    text = text.replace(inc, inc + "\n" + DIV_CUDA_FUNC, 1)
    # device-guard the public function body
    anchor = ("mpfr_div (mpfr_ptr q, mpfr_srcptr u, mpfr_srcptr v, "
              "mpfr_rnd_t rnd_mode)\n{")
    if anchor not in text:
        raise SystemExit("patch_div_cuda: mpfr_div definition not found")
    text = text.replace(anchor, anchor + "\n#ifdef __CUDA_ARCH__\n"
                        "  return mpfr_div_cuda (q, u, v, rnd_mode);\n#endif", 1)
    return text


LONGLONG_PTX = r'''
/* ---- CUDA device fast paths (injected by tools/cudafy_mpfr.py) ----------
 * Native 64-bit primitives for the device pass: __umul64hi / __clzll /
 * add.cc carry chains replace the generic C fallbacks (NO_ASM skips all the
 * host asm variants).  Exact integer semantics -- results stay bit-identical
 * to the host.  Defined first, so longlong.h's "#if !defined (...)" generic
 * fallbacks are skipped on the device; the host pass is unchanged. */
#if defined(__CUDA_ARCH__) && defined(GMP_NUMB_BITS) && GMP_NUMB_BITS == 64
#define umul_ppmm(w1, w0, u, v)                                         \
  do {                                                                  \
    UWtype __cu_u = (u), __cu_v = (v);                                  \
    (w0) = __cu_u * __cu_v;                                             \
    (w1) = (UWtype) __umul64hi ((unsigned long long) __cu_u,            \
                                (unsigned long long) __cu_v);           \
  } while (0)
#define add_ssaaaa(sh, sl, ah, al, bh, bl)                              \
  __asm__ ("add.cc.u64 %1, %3, %5;\n\t"                                 \
           "addc.u64   %0, %2, %4;"                                     \
           : "=&l" (sh), "=&l" (sl)                                     \
           : "l" (ah), "l" (al), "l" (bh), "l" (bl))
#define sub_ddmmss(sh, sl, ah, al, bh, bl)                              \
  __asm__ ("sub.cc.u64 %1, %3, %5;\n\t"                                 \
           "subc.u64   %0, %2, %4;"                                     \
           : "=&l" (sh), "=&l" (sl)                                     \
           : "l" (ah), "l" (al), "l" (bh), "l" (bl))
#define count_leading_zeros(count, x)                                   \
  do { (count) = __clzll ((unsigned long long) (x)); } while (0)
#define COUNT_LEADING_ZEROS_0 64
#define count_trailing_zeros(count, x)                                  \
  do { (count) = __ffsll ((unsigned long long) (x)) - 1; } while (0)
#endif

'''


def patch_longlong_ptx(text):
    """mpfr-longlong.h: prepend the device fast-path macro block."""
    return LONGLONG_PTX + text


def patch_mparam(text):
    """mparam.h: MPFR's host-arch dispatch #includes a per-CPU tuning header
    (e.g. "x86_64/mparam.h", selected because nvcc defines __x86_64__ on the
    host).  Only generic/mparam.h is copied into the device tree, so force the
    generic/default case -- the CPU tuning thresholds are irrelevant to device
    code and generic/mparam.h supplies safe defaults for every threshold."""
    # Replace the whole "#if defined(MPFR_TUNE_COVERAGE) ... #endif" dispatch
    # block (it precedes the generic/mparam.h include) with a forced default.
    start = text.find("/* Threshold when testing coverage */")
    if start == -1:
        return text
    end = text.find("/****", start)
    if end == -1:
        return text
    replacement = (
        "/* CUDA: per-arch tuning subdirs (x86_64/, x86/, arm/, ...) are not part\n"
        "   of the transformed device tree -- only generic/ is.  Force the\n"
        "   generic/default case so the host-arch dispatch never #includes a\n"
        "   missing per-arch mparam.h.  generic/mparam.h below fills in defaults. */\n"
        '#define MPFR_TUNE_CASE "default"\n\n')
    return text[:start] + replacement + text[end:]


def patch_mini_gmp_header(text):
    """mpfr-mini-gmp.h declares the shim prototypes (gmp_rand*, mpn_divrem*,
    mpz_urandomb, ...) as plain functions, but their definitions in
    mpfr-mini-gmp.c carry __host__ __device__ (added by attribute_definitions).
    nvcc warns (#20040-D) on the host-vs-host/device mismatch.  Prefix every
    function prototype here with ATTR so declaration and definition agree.
    attribute_prototypes() does not cover these because they lack the
    __MPFR_DECLSPEC marker it keys on."""
    out = []
    for line in text.split("\n"):
        s = line.lstrip()
        # start-of-prototype line: `<return type> <name> (` -- excludes
        # typedefs, preprocessor lines, already-attributed lines, and the
        # continuation lines of multi-line prototypes (which have no '(').
        if (re.match(r"[A-Za-z_][\w ]*\**\s*\w+\s*\(", s)
                and not s.startswith(("#", "typedef"))
                and ATTR not in line):
            indent = line[:len(line) - len(s)]
            out.append(indent + ATTR + " " + s)
        else:
            out.append(line)
    return "\n".join(out)


def patch_mini_gmp_shims(text):
    """mpfr-mini-gmp.c: the mpn_divrem_1 shim divides via mpz_init/clear.  That
    internal allocation corrupts control flow when the shim is called from deep
    inside MPFR's division (mpfr_div_ui) on the device -- it returns to the
    wrong place and the result comes out as inf -- even though the shim is
    correct when called standalone.  Replace it with an allocation-free 128-bit
    long division (correct and self-contained on the device)."""
    old = (
        "  mpz_t q, r, n, d;\n"
        "  mp_limb_t ret, dd[1];\n"
        "\n"
        "  d->_mp_d = dd;\n"
        "  d->_mp_d[0] = d0;\n"
        "  d->_mp_size = 1;\n"
        "  mpz_init (q);\n"
        "  mpz_init (r);\n"
        "  if (qxn == 0)\n"
        "    {\n"
        "      n->_mp_d = np;\n"
        "      n->_mp_size = nn;\n"
        "    }\n"
        "  else\n"
        "    {\n"
        "      mpz_init2 (n, (nn + qxn) * GMP_NUMB_BITS);\n"
        "      mpn_copyi (n->_mp_d + qxn, np, nn);\n"
        "      mpn_zero (n->_mp_d, qxn);\n"
        "      n->_mp_size = nn + qxn;\n"
        "    }\n"
        "  mpz_tdiv_qr (q, r, n, d);\n"
        "  if (q->_mp_size > 0)\n"
        "    mpn_copyi (qp, q->_mp_d, q->_mp_size);\n"
        "  if (q->_mp_size < nn + qxn)\n"
        "    mpn_zero (qp + q->_mp_size, nn + qxn - q->_mp_size);\n"
        "  ret = (r->_mp_size == 1) ? r->_mp_d[0] : 0;\n"
        "  mpz_clear (q);\n"
        "  mpz_clear (r);\n"
        "  if (qxn != 0)\n"
        "    mpz_clear (n);\n"
        "  return ret;")
    new = (
        "  /* CUDA: allocation-free single-limb division (the mpz-based shim\n"
        "     corrupts control flow when called from deep MPFR code on the\n"
        "     device).  Computes {np,nn}*B^qxn / d0. */\n"
        "  unsigned __int128 acc = 0;\n"
        "  mp_size_t i;\n"
        "  for (i = nn - 1; i >= 0; i--)\n"
        "    {\n"
        "      acc = (acc << GMP_NUMB_BITS) | (unsigned __int128) np[i];\n"
        "      qp[qxn + i] = (mp_limb_t) (acc / d0);\n"
        "      acc %= d0;\n"
        "    }\n"
        "  for (i = qxn - 1; i >= 0; i--)\n"
        "    {\n"
        "      acc = acc << GMP_NUMB_BITS;\n"
        "      qp[i] = (mp_limb_t) (acc / d0);\n"
        "      acc %= d0;\n"
        "    }\n"
        "  return (mp_limb_t) acc;")
    if old not in text:
        raise SystemExit("patch_mini_gmp_shims: mpn_divrem_1 body not found")
    return text.replace(old, new, 1)


def patch_impl_header(text):
    """mpfr-impl.h: make the global thread-vars __device__ __managed__ (unified
    host+device, mutable)."""
    # the 5 extern declarations (lines beginning `extern MPFR_THREAD_ATTR`)
    text = re.sub(r"extern MPFR_THREAD_ATTR\b",
                  "extern __device__ __managed__", text)
    # the MPFR_THREAD_VAR definition macro (non-thread-safe branch)
    text = text.replace(
        "  MPFR_THREAD_ATTR T N = (V);      \\",
        "  __device__ __managed__ T N = (V);      \\")
    # Constant caches (const_pi/log2/euler/catalan): mpfr_cache() cannot run on
    # the device -- it calls a compute-function *pointer* stored at host
    # static-init (a HOST address) and memoizes into a shared mutable global
    # (data races across threads).  We bypass the cache and call the *_internal
    # compute functions, but those rely on the cache's wrapper to (a) widen the
    # exponent range (MPFR_SAVE_EXPO) around the computation and (b) round/check
    # the result -- without that the constants come out WRONG.  So we redirect
    # the const_* macros to wrapper functions that reproduce exactly that, for
    # BOTH host and device (so the two paths match bit-for-bit).  Keep the cache
    # storage unified (__managed__) only so const_*.c still compiles.
    text = text.replace(
        "# define MPFR_CACHE_ATTR MPFR_THREAD_ATTR",
        "# define MPFR_CACHE_ATTR __device__ __managed__")
    wrappers = ["\n/* CUDA: cacheless constant wrappers (see cudafy_mpfr.py) */"]
    for nm in ("pi", "log2", "euler", "catalan"):
        wrappers.append(
            "__host__ __device__ static inline int mpfr_const_%s_cuda "
            "(mpfr_ptr d, mpfr_rnd_t r) {\n"
            "  MPFR_SAVE_EXPO_DECL (expo); int inex;\n"
            "  MPFR_SAVE_EXPO_MARK (expo);\n"
            "  inex = mpfr_const_%s_internal (d, r);\n"
            "  MPFR_SAVE_EXPO_FREE (expo);\n"
            "  return mpfr_check_range (d, inex, r);\n}" % (nm, nm))
        text = re.sub(
            r"#define mpfr_const_%s\(_d,\s*_r\)\s+mpfr_cache\([^\n]*\)" % nm,
            "#define mpfr_const_%s(_d,_r) mpfr_const_%s_cuda(_d,_r)" % (nm, nm),
            text)
    # inject the wrapper definitions after the *_internal declarations (so the
    # internals and MPFR_SAVE_EXPO_* are already in scope).
    anchor = ("__MPFR_DECLSPEC int mpfr_const_catalan_internal "
              "(mpfr_ptr, mpfr_rnd_t);")
    text = text.replace(anchor, anchor + "\n".join(wrappers), 1)
    return text


def attribute_const_data(text):
    """Mark file-scope read-only lookup tables / const data with __device__ so
    device code can reference them.  Only touches column-0 *data* declarations
    (those containing '[' or '=' or a bare ';'), never const-returning function
    type lines."""
    # __clz_tab (==mpfr_clz_tab, via a #define) is an *exported* table (defined
    # once in mp_clz_tab.c, declared extern in mpfr-longlong.h).  Use the
    # MPFR_RODATA scheme: a __device__ global in the device pass, resolved
    # across TUs by nvlink (-rdc).  Drop the upstream `const` mismatch.  Done
    # BEFORE the generic per-line pass so `extern const` is not re-rewritten.
    # def is `const\nunsigned char __clz_tab[129] =` (GMP style, const on its
    # own line); keep the const and add MPFR_RODATA -> `const __device__ ...`.
    text = text.replace("unsigned char __clz_tab[129] =",
                        "MPFR_RODATA unsigned char __clz_tab[129] =")
    # match the extern's const so both are `__device__ const` (compatible).
    text = text.replace(
        "extern const unsigned char __MPFR_DECLSPEC __clz_tab[129];",
        "extern MPFR_RODATA const unsigned char __MPFR_DECLSPEC __clz_tab[129];")

    out = []
    for line in text.split("\n"):
        if line[:1] not in " \t":     # column 0 only
            is_data = ("[" in line or "=" in line
                       or (line.rstrip().endswith(";") and "(" not in line))
            if is_data:
                # MPFR_RODATA -> __device__ (device pass) / nothing (host pass).
                # Covers `static const` tables AND mutable file-scope `static`
                # lookup tables (e.g. MPFR's *_ktab tuning tables, declared
                # `static short`), which the device must also be able to read.
                if line.startswith("static "):
                    line = "MPFR_RODATA " + line
                elif line.startswith("const "):
                    line = "MPFR_RODATA " + line
                elif line.startswith("extern const "):
                    line = "extern MPFR_RODATA const " + line[len("extern const "):]
                elif line.startswith("__MPFR_DECLSPEC extern const "):
                    line = ("__MPFR_DECLSPEC extern MPFR_RODATA const "
                            + line[len("__MPFR_DECLSPEC extern const "):])
        out.append(line)
    return "\n".join(out)


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: cudafy_mpfr.py <mpfr_src_dir> <out_dir>")
    src, out = sys.argv[1:]
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)

    # MPFR's mparam.h does #include "generic/mparam.h" (tuning params).
    gen_src = os.path.join(src, "generic", "mparam.h")
    if os.path.exists(gen_src):
        os.makedirs(os.path.join(out, "generic"))
        shutil.copy(gen_src, os.path.join(out, "generic", "mparam.h"))

    # mini-gmp shim: redirect MPFR's #include "mini-gmp.h" to the CUDA port.
    with open(os.path.join(out, "mini-gmp.h"), "w") as f:
        f.write('/* shim -> CUDA-ported mini-gmp + CUDA helpers */\n'
                '#include "mpc_cuda/cuda_minigmp.h"\n'
                '/* MPFR_RODATA: __device__ in the device pass, nothing on the\n'
                '   host pass, so one definition of a read-only table serves\n'
                '   both address spaces. */\n'
                '#ifndef MPFR_RODATA\n'
                '# ifdef __CUDA_ARCH__\n'
                '#  define MPFR_RODATA __device__\n'
                '# else\n'
                '#  define MPFR_RODATA\n'
                '# endif\n'
                '#endif\n')

    for fn in sorted(os.listdir(src)):
        if not fn.endswith((".c", ".h")):
            continue
        if fn == "mini-gmp.h":          # never copy the pristine one
            continue
        with open(os.path.join(src, fn)) as f:
            text = f.read()

        if fn == "mpfr-impl.h":
            text = patch_impl_header(text)
        if fn == "ieee_floats.h":
            text = patch_ieee_floats(text)
        if fn == "mpfr-gmp.c":
            text = patch_assert_fail(text)
            text = patch_mpfr_gmp_alloc(text)
        if fn == "mpfr-mini-gmp.c":
            text = patch_mini_gmp_shims(text)
        if fn == "mpfr-mini-gmp.h":
            text = patch_mini_gmp_header(text)
        if fn == "mparam.h":
            text = patch_mparam(text)
        if fn == "mpfr-longlong.h":
            text = patch_longlong_ptx(text)
        if fn in ("const_pi.c", "const_log2.c", "const_euler.c",
                  "const_catalan.c"):
            # The PUBLIC mpfr_const_* functions (reached by callers that include
            # only mpfr.h, e.g. tests / MPC) call mpfr_cache().  Route them to
            # the cacheless wrappers for host AND device so both paths match and
            # neither touches the (device-hostile) cache.
            text = re.sub(
                r"return mpfr_cache \((\w+), __gmpfr_cache_const_(\w+), (\w+)\);",
                r"return mpfr_const_\2_cuda (\1, \3);",
                text)
        if fn == "abort_prec_max.c":
            text = text.replace(
                '  fprintf (stderr, "MPFR: Maximal precision overflow\\n");\n'
                '  abort ();',
                '#ifdef __CUDA_ARCH__\n  __trap ();\n#else\n'
                '  fprintf (stderr, "MPFR: Maximal precision overflow\\n");\n'
                '  abort ();\n#endif')

        text = attribute_const_data(text)
        if fn.endswith(".h"):
            text = attribute_prototypes(text)
            # template headers (#included bodies) also carry definitions
            text = attribute_definitions(text)
        else:
            text = attribute_definitions(text)

        # applied after the attribute passes so the injected helper (which
        # already carries __host__ __device__) is not double-qualified.
        if fn == "div.c":
            text = patch_div_cuda(text)

        # rename exported MPFR/GMP symbols into the cu_ namespace (coexist
        # with the system libmpfr/libgmp in one binary)
        text = cu_prefix.rename(text)

        with open(os.path.join(out, fn), "w") as f:
            f.write("/* CUDA-adapted from MPFR by tools/cudafy_mpfr.py. */\n"
                    + text)

    print("cudafied MPFR ->", out)


if __name__ == "__main__":
    main()
