"""cu_prefix.py -- rename exported GMP/MPFR/MPC symbols to a `cu_` namespace.

The CUDA-ported library defines the same C-linkage symbols as the upstream
GMP/MPFR/MPC libraries (`mpfr_mul`, `mpc_mul`, `mpz_add`, ...).  To let a program
link BOTH this library (for the GPU) and the real system libraries (for the CPU)
without "multiple definition" collisions, every exported function/data symbol is
renamed to `cu_<name>` (so `mpfr_mul` -> `cu_mpfr_mul`, etc.).

The rename is a whole-word textual substitution applied to the generated sources
and headers, driven by tools/cu_rename_syms.txt -- the exact set of external
symbols the objects export (produced once with `nm`).  Because it is textual and
whole-word, it renames the definition, every call site, the prototype, any
function-like macro of the same name, and matching `#undef` lines uniformly --
while leaving TYPES (mpfr_t, mp_limb_t), enum/rounding macros (MPFR_RNDN), and our
own arena helpers (mpc_cuda_*) untouched (those are not in the symbol list).

Idempotent: `cu_mpfr_mul` has no word boundary before `mpfr_mul`, so re-running
never double-prefixes.
"""
import os
import re

PREFIX = "cu_"
_SYMS_FILE = os.path.join(os.path.dirname(__file__), "cu_rename_syms.txt")

_pattern = None

# Token-paste sites that BUILD a renamed symbol from a literal stem plus the `##`
# operator (e.g. MPC's `mpfr_set_ ## type` -> mpfr_set_d).  The whole-word pass
# cannot see the constructed name, so the literal stem before `##` is prefixed
# here instead.  Matches `<family>_..._` immediately followed by `##`.
_PASTE_RE = re.compile(
    r"\b((?:mpfr|mpc|mpz|mpn|mpf|mpq|gmp)_[A-Za-z0-9_]*?_)(\s*##)")


def _load():
    global _pattern
    if _pattern is not None:
        return
    with open(_SYMS_FILE) as f:
        syms = [s.strip() for s in f if s.strip() and not s.startswith("#")]
    # longest first is irrelevant with \b, but keep deterministic order
    syms.sort(key=lambda s: (-len(s), s))
    _pattern = re.compile(r"\b(" + "|".join(re.escape(s) for s in syms) + r")\b")


def symbols():
    _load()
    with open(_SYMS_FILE) as f:
        return [s.strip() for s in f if s.strip() and not s.startswith("#")]


def rename(text):
    """Return text with every exported GMP/MPFR/MPC symbol prefixed by cu_."""
    _load()
    text = _pattern.sub(lambda m: PREFIX + m.group(1), text)
    # prefix literal stems consumed by the `##` paste operator
    text = _PASTE_RE.sub(lambda m: PREFIX + m.group(1) + m.group(2), text)
    return text
