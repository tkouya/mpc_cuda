/* elem_bench_mpfr.cu -- benchmark MPFR elementary / transcendental functions
 * on the GPU vs the CPU (this library's host MPFR), with accuracy check.
 *
 * For each function f, evaluate y[i] = f(x[i]) over N inputs on the GPU (fixed
 * thread pool, grid-stride, per-element arena reset) and on the CPU, and report
 * GPU/CPU time, speedup, and max relative difference.  Build: `make bench-mpfr`.
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
typedef long int gmp_randstate_t[1];
#include "mpc_cuda.cuh"

#ifndef N
#define N 4096
#endif
#ifndef PREC
#define PREC 1024
#endif
#ifndef SLAB
#define SLAB (1024 * 1024)           /* one transcendental evaluation */
#endif
#ifndef LBLOCKS
#define LBLOCKS 256
#endif
#ifndef LTHREADS
#define LTHREADS 32
#endif

enum { F_SQRT, F_CBRT, F_EXP, F_EXPM1, F_LOG, F_LOG1P,
       F_SIN, F_COS, F_TAN, F_ATAN, F_SINH, F_COSH, NFUNC };
static const char *FNAME[NFUNC] =
  { "sqrt", "cbrt", "exp", "expm1", "log", "log1p",
    "sin", "cos", "tan", "atan", "sinh", "cosh" };

__host__ __device__ static double
apply_fn (int fid, double xv)
{
  CU_MPFR_DECL_INIT (x, PREC);
  CU_MPFR_DECL_INIT (r, PREC);
  cu_mpfr_set_d (x, xv, CU_MPFR_RNDN);
  switch (fid)
    {
    case F_SQRT:  cu_mpfr_sqrt  (r, x, CU_MPFR_RNDN); break;
    case F_CBRT:  cu_mpfr_cbrt  (r, x, CU_MPFR_RNDN); break;
    case F_EXP:   cu_mpfr_exp   (r, x, CU_MPFR_RNDN); break;
    case F_EXPM1: cu_mpfr_expm1 (r, x, CU_MPFR_RNDN); break;
    case F_LOG:   cu_mpfr_log   (r, x, CU_MPFR_RNDN); break;
    case F_LOG1P: cu_mpfr_log1p (r, x, CU_MPFR_RNDN); break;
    case F_SIN:   cu_mpfr_sin   (r, x, CU_MPFR_RNDN); break;
    case F_COS:   cu_mpfr_cos   (r, x, CU_MPFR_RNDN); break;
    case F_TAN:   cu_mpfr_tan   (r, x, CU_MPFR_RNDN); break;
    case F_ATAN:  cu_mpfr_atan  (r, x, CU_MPFR_RNDN); break;
    case F_SINH:  cu_mpfr_sinh  (r, x, CU_MPFR_RNDN); break;
    case F_COSH:  cu_mpfr_cosh  (r, x, CU_MPFR_RNDN); break;
    }
  return cu_mpfr_get_d (r, CU_MPFR_RNDN);
}

__global__ void
bench_kernel (int fid, int n, const double *X, double *Y)
{
  int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    {
#ifdef __CUDA_ARCH__
      mpc_cuda_arena_reset ();
#endif
      Y[i] = apply_fn (fid, X[i]);
    }
}

int
main (void)
{
  cudaDeviceSetLimit (cudaLimitMallocHeapSize, (size_t) 256 * 1024 * 1024);
  cudaDeviceSetLimit (cudaLimitStackSize,      (size_t) 192 * 1024);

  size_t MV = (size_t) N * sizeof (double);
  double *X = (double *) malloc (MV);
  double *Yg = (double *) malloc (MV), *Yc = (double *) malloc (MV);
  for (int i = 0; i < N; ++i) X[i] = 0.5 + (double) i / N;   /* (0.5, 1.5) */

  double *dX, *dY;
  cudaMalloc (&dX, MV); cudaMalloc (&dY, MV);
  cudaMemcpy (dX, X, MV, cudaMemcpyHostToDevice);

  size_t ntot = (size_t) LBLOCKS * LTHREADS;
  char *arena; size_t *top;
  cudaMalloc (&arena, ntot * (size_t) SLAB);
  cudaMalloc (&top,   ntot * sizeof (size_t));
  cudaMemset (top, 0, ntot * sizeof (size_t));
  mpc_cuda_arena_base = arena; mpc_cuda_arena_slab = SLAB; mpc_cuda_arena_top = top;
  cudaDeviceSynchronize ();

  cudaEvent_t e0, e1; cudaEventCreate (&e0); cudaEventCreate (&e1);

  printf ("=== MPFR elementary/transcendental benchmark "
          "(N=%d inputs in (0.5,1.5), precision=%d bits) ===\n", N, PREC);
  printf ("%-8s %12s %12s %10s %14s\n",
          "func", "GPU [ms]", "CPU [ms]", "speedup", "max rel diff");

  for (int f = 0; f < NFUNC; ++f)
    {
      bench_kernel<<<LBLOCKS, LTHREADS>>> (f, N, dX, dY);    /* warm-up */
      cudaDeviceSynchronize ();
      cudaEventRecord (e0);
      bench_kernel<<<LBLOCKS, LTHREADS>>> (f, N, dX, dY);
      cudaEventRecord (e1);
      cudaError_t err = cudaDeviceSynchronize ();
      if (err != cudaSuccess) { fprintf (stderr, "%s kernel failed: %s\n", FNAME[f], cudaGetErrorString (err)); return 1; }
      float gpu_ms = 0.f; cudaEventElapsedTime (&gpu_ms, e0, e1);
      cudaMemcpy (Yg, dY, MV, cudaMemcpyDeviceToHost);

      struct timespec t0, t1;
      clock_gettime (CLOCK_MONOTONIC, &t0);
      for (int i = 0; i < N; ++i) Yc[i] = apply_fn (f, X[i]);
      clock_gettime (CLOCK_MONOTONIC, &t1);
      double cpu_ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) * 1e-6;

      double maxrel = 0.0;
      for (int i = 0; i < N; ++i)
        {
          double rel = Yc[i] != 0.0 ? fabs ((Yg[i] - Yc[i]) / Yc[i]) : fabs (Yg[i]);
          if (rel > maxrel) maxrel = rel;
        }
      printf ("%-8s %12.3f %12.3f %9.1fx %14.3e\n",
              FNAME[f], gpu_ms, cpu_ms, cpu_ms / gpu_ms, maxrel);
    }

  free (X); free (Yg); free (Yc);
  cudaFree (dX); cudaFree (dY); cudaFree (arena); cudaFree (top);
  return 0;
}
