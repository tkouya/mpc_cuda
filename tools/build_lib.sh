#!/usr/bin/env bash
# build_lib.sh -- build the mpc_cuda device-object "library".
#
# Compiles every standalone CUDA-adapted device object once (mini-gmp -> MPFR ->
# MPC) from the committed source trees, then writes a symbol index used by
# tools/link_program.sh to resolve the device link closure for a user program.
#
# The CUDA-adapted sources are FIRST-CLASS, COMMITTED sources:
#   src/cuda_minigmp.cu  + include/mpc_cuda/cuda_minigmp.h   (mini-gmp)
#   src/mpfr_cuda/*.{c,h}                                    (MPFR)
#   src/mpc_cuda/*.{c,h}                                     (MPC)
# They were produced once from the upstream GMP/MPFR/MPC sources by the
# transform scripts in tools/ (cudafy_*.py).  THE UPSTREAM SOURCES ARE NO LONGER
# REQUIRED to build: this script does NOT regenerate by default.
#
# Re-running the transforms (only needed when upgrading upstream) is opt-in:
#   GEN=1  regenerate the committed trees from $GMP_SRC/$MPFR_SRC/$MPC_SRC,
#          which must then be present.  Without GEN=1 the committed trees are
#          used as-is.
#
# Honours these environment variables (configure substitutes them):
#   NVCC, PYTHON, CUDA_ARCH, BUILDDIR, GMP_SRC, MPFR_SRC, MPC_SRC
# Run from the repository root.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

NVCC="${NVCC:-nvcc}"
PYTHON="${PYTHON:-python3}"
ARCH="${CUDA_ARCH:-${ARCH:-sm_121}}"
BUILDDIR="${BUILDDIR:-build}"
GMP_SRC="${GMP_SRC:-gmp-6.3.0/mini-gmp}"
MPFR_SRC="${MPFR_SRC:-mpfr-4.2.2/src}"
MPC_SRC="${MPC_SRC:-mpc-1.4.1/src}"

# committed CUDA-adapted source trees
MPFR_CUDA="src/mpfr_cuda"
MPC_CUDA="src/mpc_cuda"

DEFS="$(grep '^-D' tools/mpfr_cuda_defs.txt | tr '\n' ' ')"
# -Xcompiler -fPIC: the objects feed BOTH the static archive (libmpc_cuda.a)
# and the shared library (libmpc_cuda.so); a .so needs position-independent
# host code, and PIC objects are equally fine inside the .a.
SUP="-diag-suppress 20011 -fmad=false -Xcompiler -fPIC"
CXX="${CXX:-g++}"
AR="${AR:-ar}"
NM="${NM:-nm}"
CUDA_LIBDIR="${CUDA_LIBDIR:-$(dirname "$(command -v "$NVCC")")/../lib64}"
# #include-only fragments (not standalone translation units)
MPFR_FRAG="add1sp1_extracted mul_1_extracted sub1sp1_extracted round_raw_generic jyn_asympt"

mkdir -p "$BUILDDIR/mpfr_cuda" "$BUILDDIR/mpc_cuda"

if [ "${GEN:-0}" = 1 ]; then
  echo ">> regenerate CUDA-adapted trees from upstream (GEN=1)"
  for d in "$GMP_SRC" "$MPFR_SRC" "$MPC_SRC"; do
    [ -d "$d" ] || { echo "!! upstream dir '$d' missing -- cannot regenerate" >&2; exit 1; }
  done
  mkdir -p "$MPFR_CUDA" "$MPC_CUDA"
  "$PYTHON" tools/cudafy_minigmp.py "$GMP_SRC" include/mpc_cuda/cuda_minigmp.h src/cuda_minigmp.cu >/dev/null || exit 1
  "$PYTHON" tools/cudafy_mpfr.py "$MPFR_SRC" "$MPFR_CUDA"                                            >/dev/null || exit 1
  "$PYTHON" tools/cudafy_mpc.py  "$MPC_SRC"  "$MPC_CUDA"                                             >/dev/null || exit 1
fi

# sanity-check the committed trees are present
[ -f src/cuda_minigmp.cu ] || { echo "!! src/cuda_minigmp.cu missing (run with GEN=1 once)" >&2; exit 1; }
ls "$MPFR_CUDA"/*.c >/dev/null 2>&1 || { echo "!! $MPFR_CUDA has no sources (run with GEN=1 once)" >&2; exit 1; }
ls "$MPC_CUDA"/*.c  >/dev/null 2>&1 || { echo "!! $MPC_CUDA has no sources (run with GEN=1 once)" >&2; exit 1; }

echo ">> compile mini-gmp"
$NVCC -x cu -dc -arch=$ARCH $DEFS -Iinclude -Iinclude/mpc_cuda $SUP \
      src/cuda_minigmp.cu -o "$BUILDDIR/cuda_minigmp.o" || exit 1

# mpfr-mini-gmp is a mandatory object: it provides the mini-gmp glue every MPFR
# function needs.  It must build or the whole library is unusable, so compile it
# loudly and abort on failure -- never let it be silently dropped like the
# optional per-function objects below.
echo ">> compile MPFR mini-gmp shim (required)"
$NVCC -x cu -dc -arch=$ARCH $DEFS -I"$MPFR_CUDA" -Iinclude $SUP \
      "$MPFR_CUDA/mpfr-mini-gmp.c" -o "$BUILDDIR/mpfr_cuda/mpfr-mini-gmp.o" || {
  echo "!! failed to compile required object $MPFR_CUDA/mpfr-mini-gmp.c" >&2; exit 1; }

echo ">> compile MPFR objects"
for f in "$MPFR_CUDA"/*.c; do b=$(basename "$f" .c)
  case " $MPFR_FRAG " in *" $b "*) continue;; esac
  [ "$b" = mpfr-mini-gmp ] && continue   # already built above (required)
  $NVCC -x cu -dc -arch=$ARCH $DEFS -I"$MPFR_CUDA" -Iinclude $SUP \
        "$f" -o "$BUILDDIR/mpfr_cuda/$b.o" 2>/dev/null || rm -f "$BUILDDIR/mpfr_cuda/$b.o"
done

echo ">> compile MPC objects"
for f in "$MPC_CUDA"/*.c; do b=$(basename "$f" .c)
  $NVCC -x cu -dc -arch=$ARCH $DEFS -I"$MPC_CUDA" -I"$MPFR_CUDA" -Iinclude $SUP \
        "$f" -o "$BUILDDIR/mpc_cuda/$b.o" 2>/dev/null || rm -f "$BUILDDIR/mpc_cuda/$b.o"
done

# --- prune to a self-consistent object set ---------------------------------
# A few upstream sources cannot be ported (FILE* I/O, missing mini-gmp helpers)
# and are silently dropped above (mpc: pow, balls, out_str, log10, logging,
# rootofunity).  Their symbols (mpc_pow, mpcb_*, ...) are therefore absent from
# the library, yet OTHER objects (eta, exp2, exp10, pow_*) reference them.  A
# single archive containing those danglers makes nvlink/ld pull them in and
# fail with "undefined reference to mpc_pow" even for programs that never use
# them.  So we drop, iteratively, every object that references one of our own
# (mpfr/mpc/gmp-namespace) symbols that nothing in the set defines, until the
# set is closed under reference.  Programs that genuinely need a dropped
# function still fail loudly at their own link -- exactly as before.
echo ">> prune to self-consistent object set"
ALLOBJ="$BUILDDIR/cuda_minigmp.o $(ls "$BUILDDIR"/mpfr_cuda/*.o "$BUILDDIR"/mpc_cuda/*.o 2>/dev/null)"
NS='^(cu_)?(__)?(mpfr|mpc|mpcb|mpcr|mpn|mpz|mpq|gmp)'
KEEP="$ALLOBJ"
for _ in 1 2 3 4 5 6 7 8; do
  $NM $KEEP 2>/dev/null | awk '$2 ~ /^[TDBRtdbrVvWw]$/ {print $3}' | sort -u > "$BUILDDIR/.defined"
  NEW=""; CHANGED=0
  for o in $KEEP; do
    miss=$($NM -u "$o" 2>/dev/null | awk '{print $NF}' | grep -E "$NS" | sort -u \
           | comm -23 - "$BUILDDIR/.defined" | grep -E "$NS")
    if [ -n "$miss" ]; then CHANGED=1; else NEW="$NEW $o"; fi
  done
  KEEP="$NEW"; [ $CHANGED -eq 0 ] && break
done
rm -f "$BUILDDIR/.defined"
NKEEP=$(echo $KEEP | wc -w); NALL=$(echo $ALLOBJ | wc -w)
echo "   keeping $NKEEP of $NALL objects ($((NALL - NKEEP)) dropped as unsatisfiable)"

# required objects must survive the prune, else the library is unusable
for req in "$BUILDDIR/cuda_minigmp.o" "$BUILDDIR/mpfr_cuda/mpfr-mini-gmp.o"; do
  case " $KEEP " in *" $req "*) ;; *)
    echo "!! required object '$req' missing/pruned -- build incomplete" >&2; exit 1;; esac
done

# --- stage kept objects under collision-free names -------------------------
# MPFR and MPC both ship e.g. acos.c -> acos.o; a flat archive keys members by
# basename, so we stage subsystem-prefixed copies before ar.
STAGE="$BUILDDIR/.ar"
rm -rf "$STAGE"; mkdir -p "$STAGE"
for o in $KEEP; do
  case "$o" in
    */mpfr_cuda/*) pfx=mpfr__ ;;
    */mpc_cuda/*)  pfx=mpc__  ;;
    *)             pfx=gmp__  ;;
  esac
  cp "$o" "$STAGE/$pfx$(basename "$o")"
done

# --- static library --------------------------------------------------------
echo ">> archive static library libmpc_cuda.a"
rm -f "$BUILDDIR/libmpc_cuda.a"
$AR rcs "$BUILDDIR/libmpc_cuda.a" "$STAGE"/*.o || exit 1

# --- shared library --------------------------------------------------------
# Device-link the (self-contained) set into one device object, then wrap the
# host code in a .so.  NOTE: nvlink cannot pull device code out of a .so, so the
# .so cannot supply __device__ functions to an external kernel's device link --
# device-linked executables must use libmpc_cuda.a.  The .so exposes the
# host-callable (__host__ __device__) cu_* API for host-only consumers.
echo ">> device-link + build shared library libmpc_cuda.so"
if $NVCC -dlink -arch=$ARCH $SUP $KEEP -o "$BUILDDIR/mpc_cuda_dlink.o" 2>"$BUILDDIR/dlink.err"; then
  if $CXX -shared $KEEP "$BUILDDIR/mpc_cuda_dlink.o" \
        -L"$CUDA_LIBDIR" -lcudart -o "$BUILDDIR/libmpc_cuda.so" 2>"$BUILDDIR/so.err"; then
    :
  else
    echo "!! shared library link failed (see $BUILDDIR/so.err); static .a is still usable" >&2
    sed 's/^/   /' "$BUILDDIR/so.err" >&2; rm -f "$BUILDDIR/libmpc_cuda.so"
  fi
else
  echo "!! device-link for .so failed (see $BUILDDIR/dlink.err); static .a is still usable" >&2
  sed 's/^/   /' "$BUILDDIR/dlink.err" >&2
fi

rm -rf "$STAGE"
echo ">> libraries ready in $BUILDDIR:"
ls -la "$BUILDDIR"/libmpc_cuda.a "$BUILDDIR"/libmpc_cuda.so 2>/dev/null | sed 's/^/   /'
