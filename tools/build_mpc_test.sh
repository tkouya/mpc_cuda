#!/usr/bin/env bash
# build_mpc_test.sh -- build the mpc_cuda library and link the MPC-on-CUDA core
# test (tests/test_mpc.cu) against it, in one command.
#
# Thin wrapper kept for backward compatibility.  Equivalent to
#   tools/build_lib.sh && tools/link_program.sh tests/test_mpc.cu build/test_mpc
# (the library is built as libmpc_cuda.a/.so and test_mpc is device-linked
# against the static archive).  Honours SKIP_GEN and EXTRA_LINK.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

SKIP_GEN="${SKIP_GEN:-0}" tools/build_lib.sh || exit 1
tools/link_program.sh tests/test_mpc.cu build/test_mpc || exit 1
echo ">> run"; ./build/test_mpc
