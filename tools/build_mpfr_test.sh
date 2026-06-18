#!/usr/bin/env bash
# build_mpfr_test.sh -- build the mpc_cuda library and link the MPFR-on-CUDA core
# test (tests/test_mpfr.cu) against it, in one command.
#
# Thin wrapper kept for backward compatibility.  Equivalent to
#   tools/build_lib.sh && tools/link_program.sh tests/test_mpfr.cu build/test_mpfr
# (the library is built as libmpc_cuda.a/.so and test_mpfr is device-linked
# against the static archive).  Honours SKIP_GEN and EXTRA_LINK.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

SKIP_GEN="${SKIP_GEN:-0}" tools/build_lib.sh || exit 1
tools/link_program.sh tests/test_mpfr.cu build/test_mpfr || exit 1
echo ">> run"; ./build/test_mpfr
