#!/usr/bin/env bash
# closure_link.sh -- device-link a prebuilt .dco.o against the mpc_cuda device
# objects by iteratively resolving nvlink's undefined references object-by-object.
# Needed when a program pulls a combination of cu_mpfr + cu_mpc symbols that
# nvlink's single-pass static-archive selection fails to resolve.
#
#   tools/closure_link.sh <program.dco.o> <out-binary> ["<extra link args>"]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
NVCC="${NVCC:-nvcc}"; ARCH="${CUDA_ARCH:-sm_121}"; BUILD="${BUILDDIR:-build}"
SUP="-diag-suppress 20011 -fmad=false"
DCO="$1"; OUT="$2"; EXTRA="${3:-}"

POOL="$(ls $BUILD/cuda_minigmp.o $BUILD/mpfr_cuda/*.o $BUILD/mpc_cuda/*.o 2>/dev/null)"
echo ">> indexing $(echo "$POOL" | wc -w) device objects"
declare -A DEF
for o in $POOL; do
  while read -r sym; do [ -n "$sym" ] && DEF["$sym"]="$o"; done \
    < <(nm "$o" 2>/dev/null | awk 'NF==3 && $2!="U" {print $3}')
done

SET="$DCO"
for iter in $(seq 1 400); do
  err="$(mktemp)"
  if $NVCC -arch=$ARCH -rdc=true $SUP $SET $EXTRA -o "$OUT" 2>"$err"; then
    echo "   LINK OK -> $OUT (after $((iter-1)) resolution rounds)"; rm -f "$err"; exit 0
  fi
  # parse BOTH nvlink ("Undefined reference to 'sym'") and ld
  # ("undefined reference to `sym'") undefined-symbol messages
  undef="$(grep -oiE "undefined reference to [\`']?[A-Za-z_][A-Za-z0-9_]*" "$err" \
           | sed -E "s/.*reference to [\`']?//" | sort -u)"
  rm -f "$err"
  if [ -z "$undef" ]; then echo "   LINK FAILED (non-undefined error):"; \
    $NVCC -arch=$ARCH -rdc=true $SUP $SET $EXTRA -o "$OUT" 2>&1 | tail -8 >&2; exit 1; fi
  added=0
  for s in $undef; do
    o="${DEF[$s]:-}"
    if [ -n "$o" ] && [[ " $SET " != *" $o "* ]]; then SET="$SET $o"; added=$((added+1)); fi
  done
  if [ $added -eq 0 ]; then echo "   LINK FAILED: unresolved (not in any object):"; \
    echo "$undef" | sed 's/^/     /' >&2; exit 1; fi
done
echo "   LINK FAILED: too many rounds" >&2; exit 1
