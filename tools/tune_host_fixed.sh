#!/usr/bin/env bash
# tune_host_fixed.sh -- validate and benchmark the HOST (CPU) build variants of
# the fixed-precision cu_freal/cu_fcomplex/cu_ffused headers on THIS machine,
# so the fastest correct configuration can be chosen (e.g. on an x86-64 box).
# Needs only a C++17 compiler and GMP/MPFR/MPC (+ FLINT >= 3.1 for the nfloat
# comparison); no CUDA, no ./configure.
#
#   tools/tune_host_fixed.sh            # all variants, full benchmark
#   QUICK=1 tools/tune_host_fixed.sh    # shorter test and bench (<= 1024 bits)
#
# Environment:
#   CXX           compiler (default g++)
#   MP_PREFIX     prefix of GMP/MPFR/MPC if not in the default search path
#   FLINT_PREFIX  prefix of FLINT (default /usr/local); bench skipped if absent
#   PIN           command prefix pinning to one core, e.g. "taskset -c 2"
#   OUT           output directory (default build-host)
#   VARIANTS      space-separated subset of: default noasm gmp
#                 (on x86-64 with BMI2+ADX, "default" uses the mulx/adcx/adox
#                 multiply and adc/sbb add kernels; "noasm" the portable C)
#
# Each variant is first checked bit-exact against MPFR/MPC (test_freal_host);
# a variant that fails is reported and NOT benchmarked.  The summary table at
# the end lists cu_* ns/op per variant next to nfloat and MPFR/MPC.
set -u
cd "$(dirname "$0")/.."

CXX=${CXX:-g++}
OUT=${OUT:-build-host}
FLINT_PREFIX=${FLINT_PREFIX:-/usr/local}
PIN=${PIN:-}
QUICK=${QUICK:-0}
mkdir -p "$OUT"

INC="-Iinclude"; LIBDIR=""
if [ -n "${MP_PREFIX:-}" ]; then
  INC="$INC -I$MP_PREFIX/include"; LIBDIR="-L$MP_PREFIX/lib -Wl,-rpath,$MP_PREFIX/lib"
fi
ARCH=$(uname -m)
VARIANTS=${VARIANTS:-"default noasm gmp"}
flags_of () {
  case "$1" in
    default)    echo "" ;;
    noasm)      echo "-DCU_FP_NO_ASM" ;;
    gmp)        echo "-DCU_FP_HOST_USE_GMP" ;;
    *)          echo "unknown variant $1" >&2; exit 2 ;;
  esac
}
TESTN=100000; BENCHDEF=""
if [ "$QUICK" = 1 ]; then TESTN=20000; BENCHDEF="-DBENCH_MAXBITS=1024 -DBENCH_M=16384"; fi

HAVE_FLINT=0
[ -f "$FLINT_PREFIX/include/flint/nfloat.h" ] && HAVE_FLINT=1
echo "=== host fixed-precision tuning on $ARCH ($($CXX --version | head -1)) ==="
echo "    variants: $VARIANTS   FLINT: $([ $HAVE_FLINT = 1 ] && echo "$FLINT_PREFIX" || echo 'not found (no benchmark)')"

PASSED=""
for v in $VARIANTS; do
  F=$(flags_of "$v")
  echo; echo "--- variant $v ($F) ---"
  if ! $CXX -O2 -march=native $F $INC tools/test_freal_host.cpp -o "$OUT/test_$v" \
         $LIBDIR -lmpc -lmpfr -lgmp; then
    echo "variant $v: BUILD FAILED"; continue
  fi
  if "$OUT/test_$v" "$TESTN" > "$OUT/test_$v.txt"; then
    echo "variant $v: bit-exact OK"
  else
    echo "variant $v: *** MISMATCHES vs MPFR/MPC -- do not use ***"
    grep FAIL "$OUT/test_$v.txt" | head; continue
  fi
  PASSED="$PASSED $v"
  [ $HAVE_FLINT = 1 ] || continue
  if $CXX -O3 -march=native $F $BENCHDEF $INC -I"$FLINT_PREFIX/include" tools/bench_host_fixed.cpp \
       -o "$OUT/bench_$v" -L"$FLINT_PREFIX/lib" -Wl,-rpath,"$FLINT_PREFIX/lib" -lflint \
       $LIBDIR -lmpc -lmpfr -lgmp; then
    $PIN "$OUT/bench_$v" | tee "$OUT/bench_$v.txt" | grep -E '^ *(bits|  64|1024|4096) '
  else
    echo "variant $v: bench build failed"
  fi
done

[ $HAVE_FLINT = 1 ] || exit 0
echo; echo "=== summary: cu_* ns/op per variant (nfloat, MPFR/MPC from the first variant) ==="
first=""; files=""
for v in $PASSED; do [ -f "$OUT/bench_$v.txt" ] || continue; [ -z "$first" ] && first=$v; files="$files $v"; done
[ -n "$first" ] || exit 0
printf "%5s %-6s" bits op; for v in $files; do printf " %12s" "$v"; done; printf " %10s %10s\n" nfloat MPFR/MPC
grep -E '^ +[0-9]+ ' "$OUT/bench_$first.txt" | while read -r bits op cu nf mp rest; do
  printf "%5s %-6s" "$bits" "$op"
  for v in $files; do
    t=$(grep -E "^ +$bits +$op " "$OUT/bench_$v.txt" | awk '{print $3}')
    printf " %12s" "${t:--}"
  done
  printf " %10s %10s\n" "$nf" "$mp"
done
echo
echo "Use the fastest variant's -D flags when compiling your host code"
echo "(CU_FP_HOST_USE_GMP also needs -lgmp; thresholds: CU_FP_GMP_MIN_N,"
echo " CU_FP_MULHIGH_MIN_N, CU_FP_CMUL_FAST_MIN_N, CU_FP_CKARA_MIN_N, CU_FP_OPSCAN_MAX_N,"
echo " CU_FP_CSHORT_MAX_N, CU_FP_X86_MULX_MIN_N, CU_FP_X86_UNROLL_MAX_N)."
echo "Note: a FLINT/GMP built for a generic or misdetected CPU (e.g. configure's"
echo "config.guess reporting 'nehalem' on a newer Xeon) lacks its ADX assembly"
echo "and makes nfloat look slower than it is; check flint-config.h for"
echo "FLINT_HAVE_ASSEMBLY_x86_64_adx before comparing."
