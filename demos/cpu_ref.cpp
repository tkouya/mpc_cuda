/* cpu_ref.cpp -- the CPU reference half, backed by the *system* libmpfr /
 * libmpc (the ordinary CPU multiple-precision libraries).
 *
 * This translation unit is compiled with a host C++ compiler and includes the
 * system <mpfr.h> / <mpc.h>, so it calls the unprefixed mpfr_* / mpc_* symbols
 * that resolve from the system libmpfr.so / libmpc.so.  The GPU half (in a .cu)
 * calls this library's cu_mpfr_* / cu_mpc_* symbols.  Both are linked into the
 * SAME binary -- the cu_ namespace lets the CUDA port coexist with the system
 * libraries with no symbol collision -- so a single program can run the GPU
 * port and check it against the trusted CPU libraries.
 *
 * Built by `make cputest` / `make cpubench`; see demos/cpu_ref.h for the shared
 * (GMP/MPFR/MPC-type-free) interface.
 */
#include "cpu_ref.h"

#include <mpfr.h>
#include <mpc.h>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" const char *const CR_NAME[CR_NMPFR] =
  { "sqrt", "cbrt", "exp", "expm1", "log", "log1p",
    "sin", "cos", "tan", "atan", "sinh", "cosh" };

extern "C" const char *const CC_NAME[CC_NMPC] =
  { "sqr", "sqrt", "exp", "log", "sin", "cos", "tan",
    "sinh", "cosh", "asin", "acos", "atan" };

extern "C" double
cpu_mpfr_eval (int fid, int prec, double x)
{
  mpfr_t X, r;
  mpfr_init2 (X, prec);
  mpfr_init2 (r, prec);
  mpfr_set_d (X, x, MPFR_RNDN);
  switch (fid)
    {
    case CR_SQRT:  mpfr_sqrt  (r, X, MPFR_RNDN); break;
    case CR_CBRT:  mpfr_cbrt  (r, X, MPFR_RNDN); break;
    case CR_EXP:   mpfr_exp   (r, X, MPFR_RNDN); break;
    case CR_EXPM1: mpfr_expm1 (r, X, MPFR_RNDN); break;
    case CR_LOG:   mpfr_log   (r, X, MPFR_RNDN); break;
    case CR_LOG1P: mpfr_log1p (r, X, MPFR_RNDN); break;
    case CR_SIN:   mpfr_sin   (r, X, MPFR_RNDN); break;
    case CR_COS:   mpfr_cos   (r, X, MPFR_RNDN); break;
    case CR_TAN:   mpfr_tan   (r, X, MPFR_RNDN); break;
    case CR_ATAN:  mpfr_atan  (r, X, MPFR_RNDN); break;
    case CR_SINH:  mpfr_sinh  (r, X, MPFR_RNDN); break;
    case CR_COSH:  mpfr_cosh  (r, X, MPFR_RNDN); break;
    }
  double v = mpfr_get_d (r, MPFR_RNDN);
  mpfr_clear (X);
  mpfr_clear (r);
  return v;
}

extern "C" void
cpu_mpc_eval (int fid, int prec, double zr, double zi, double *wr, double *wi)
{
  mpc_t x, r;
  mpc_init2 (x, prec);
  mpc_init2 (r, prec);
  mpc_set_d_d (x, zr, zi, MPC_RNDNN);
  switch (fid)
    {
    case CC_SQR:  mpc_sqr  (r, x, MPC_RNDNN); break;
    case CC_SQRT: mpc_sqrt (r, x, MPC_RNDNN); break;
    case CC_EXP:  mpc_exp  (r, x, MPC_RNDNN); break;
    case CC_LOG:  mpc_log  (r, x, MPC_RNDNN); break;
    case CC_SIN:  mpc_sin  (r, x, MPC_RNDNN); break;
    case CC_COS:  mpc_cos  (r, x, MPC_RNDNN); break;
    case CC_TAN:  mpc_tan  (r, x, MPC_RNDNN); break;
    case CC_SINH: mpc_sinh (r, x, MPC_RNDNN); break;
    case CC_COSH: mpc_cosh (r, x, MPC_RNDNN); break;
    case CC_ASIN: mpc_asin (r, x, MPC_RNDNN); break;
    case CC_ACOS: mpc_acos (r, x, MPC_RNDNN); break;
    case CC_ATAN: mpc_atan (r, x, MPC_RNDNN); break;
    }
  *wr = mpfr_get_d (mpc_realref (r), MPFR_RNDN);
  *wi = mpfr_get_d (mpc_imagref (r), MPFR_RNDN);
  mpc_clear (x);
  mpc_clear (r);
}

/* --- OpenMP-parallel batch evaluation ------------------------------------- *
 * Each element is an independent eval with its own temporaries (cpu_mpfr_eval /
 * cpu_mpc_eval init+clear their own locals), so the loop parallelizes with no
 * sharing.  libmpfr/libmpc are thread-safe (per-thread constant caches via TLS)
 * when built --enable-thread-safe, which is the default on this platform.       */
extern "C" void
cpu_mpfr_eval_array (int fid, int prec, int n, const double *X, double *Y)
{
#ifdef _OPENMP
#pragma omp parallel for schedule (static)
#endif
  for (int i = 0; i < n; ++i)
    Y[i] = cpu_mpfr_eval (fid, prec, X[i]);
}

extern "C" void
cpu_mpc_eval_array (int fid, int prec, int n, const double *Zr, const double *Zi,
                    double *Wr, double *Wi)
{
#ifdef _OPENMP
#pragma omp parallel for schedule (static)
#endif
  for (int i = 0; i < n; ++i)
    cpu_mpc_eval (fid, prec, Zr[i], Zi[i], &Wr[i], &Wi[i]);
}

extern "C" int
cpu_set_max_threads (void)
{
#ifdef _OPENMP
  int t = omp_get_num_procs ();
  omp_set_num_threads (t);
  return omp_get_max_threads ();
#else
  return 1;
#endif
}

extern "C" const char *cpu_mpfr_version (void) { return mpfr_get_version (); }
extern "C" const char *cpu_mpc_version  (void) { return mpc_get_version  (); }
