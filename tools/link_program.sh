#!/usr/bin/env bash
# link_program.sh -- compile one CUDA program against the prebuilt mpc_cuda
# static library and resolve the device link.
#
#   tools/link_program.sh <program.cu> <out-binary>
#
# Requires that tools/build_lib.sh has already produced $BUILDDIR/libmpc_cuda.a.
# The static archive is device-linked (-rdc=true): nvlink pulls exactly the
# device objects the program's kernels need out of the archive, and ld pulls the
# matching host code.  (The .so cannot be used here -- nvlink cannot read device
# code from a shared object; device-linked programs must use the .a.)
#
# Honours:
#   NVCC, CUDA_ARCH, BUILDDIR, INCDIRS, EXTRA_LINK
#   EXTRA_LINK   extra objects/libraries appended to the final link (e.g. a CPU
#                reference object plus "-lmpfr -lgmp")
# Run from the repository root (or set INCDIRS for an installed tree).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

NVCC="${NVCC:-nvcc}"
ARCH="${CUDA_ARCH:-${ARCH:-sm_121}}"
BUILDDIR="${BUILDDIR:-build}"
SRC="$1"; OUT="$2"
DEFS="$(grep '^-D' tools/mpfr_cuda_defs.txt | tr '\n' ' ')"
SUP="-diag-suppress 20011 -fmad=false"
LIB="$BUILDDIR/libmpc_cuda.a"

[ -f "$LIB" ] || { echo "   MISSING LIBRARY: $LIB" >&2
  echo "   run 'make' (or 'make clean && make') to build it first" >&2; exit 1; }

USE_MPC=0; grep -q '"mpc.h"\|<mpc.h>' "$SRC" && USE_MPC=1
INC="${INCDIRS:-}"
if [ -z "$INC" ]; then
  INC="-Isrc/mpfr_cuda -Iinclude"
  [ $USE_MPC = 1 ] && INC="-Isrc/mpc_cuda -Isrc/mpfr_cuda -Iinclude"
fi

TESTOBJ="$BUILDDIR/$(basename "$OUT").dco.o"
echo ">> compile $SRC"
$NVCC -x cu -dc -arch=$ARCH $DEFS $INC $SUP "$SRC" -o "$TESTOBJ" || exit 1

echo ">> link against $LIB"
if $NVCC -arch=$ARCH -rdc=true $SUP "$TESTOBJ" "$LIB" ${EXTRA_LINK:-} -o "$OUT" 2>"$BUILDDIR/link.err"; then
  echo "   LINK OK -> $OUT"; exit 0
fi
cat "$BUILDDIR/link.err" >&2
if grep -qi "undefined reference" "$BUILDDIR/link.err"; then
  echo "   UNRESOLVED: the program uses a function not provided by libmpc_cuda.a" >&2
  echo "   (some upstream functions cannot be ported to device code and are omitted)" >&2
fi
exit 1
