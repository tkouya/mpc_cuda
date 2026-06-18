#!/usr/bin/env bash
# build_cuda_test.sh -- convenience wrapper: build the mpc_cuda device-object
# library and link one test/demo against it, in one command.
#
#   tools/build_cuda_test.sh <program.cu> <out-binary>
#
# Equivalent to running tools/build_lib.sh followed by tools/link_program.sh.
# Honours SKIP_GEN (skip the transform step) and EXTRA_LINK (extra link inputs).
# Kept for backward compatibility; the autoconf Makefile drives build_lib.sh and
# link_program.sh separately so the library is compiled only once.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

SKIP_GEN="${SKIP_GEN:-0}" tools/build_lib.sh || exit 1
tools/link_program.sh "$1" "$2"
