/* elem_bench_mpc.cu -- benchmark MPC complex elementary / transcendental
 * functions on the GPU vs the CPU (this library's host MPC), with accuracy.
 *
 * Complex analogue of elem_bench_mpfr.cu: for each function f, evaluate
 * w[i] = f(z[i]) over N complex inputs and report GPU/CPU time, speedup, and
 * max relative difference.  Stack-backed cu_mpc_t operands (CU_MPC_DECL_INIT) +
 * per-element arena reset.  Build: `make bench-mpc`.
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
#define SLAB (2 * 1024 * 1024)        /* one complex transcendental eval */
#endif
#ifndef LBLOCKS
#define LBLOCKS 256
#endif
#ifndef LTHREADS
#define LTHREADS 32
#endif

#define NLIMBS ((PREC + 8 * (int) sizeof (cu_mp_limb_t) - 1) / (8 * (int) sizeof (cu_mp_limb_t)))
#define CU_MPC_DECL_INIT(z)                                                 \
  cu_mpc_t z;                                                              \
  cu_mp_limb_t z##_rl[NLIMBS], z##_il[NLIMBS];                             \
  cu_mpfr_custom_init_set (cu_mpc_realref (z), CU_MPFR_ZERO_KIND, 0, PREC, z##_rl); \
  cu_mpfr_custom_init_set (cu_mpc_imagref (z), CU_MPFR_ZERO_KIND, 0, PREC, z##_il)

enum { F_SQR, F_SQRT, F_EXP, F_LOG, F_SIN, F_COS, F_TAN,
       F_SINH, F_COSH, F_ASIN, F_ACOS, F_ATAN, NFUNC };
static const char *FNAME[NFUNC] =
  { "sqr", "sqrt", "exp", "log", "sin", "cos", "tan",
    "sinh", "cosh", "asin", "acos", "atan" };

__host__ __device__ static void
apply_fn (int fid, double zr, double zi, double *wr, double *wi)
{
  CU_MPC_DECL_INIT (x); CU_MPC_DECL_INIT (r);
  cu_mpc_set_d_d (x, zr, zi, CU_MPC_RNDNN);
  switch (fid)
    {
    case F_SQR:  cu_mpc_sqr  (r, x, CU_MPC_RNDNN); break;
    case F_SQRT: cu_mpc_sqrt (r, x, CU_MPC_RNDNN); break;
    case F_EXP:  cu_mpc_exp  (r, x, CU_MPC_RNDNN); break;
    case F_LOG:  cu_mpc_log  (r, x, CU_MPC_RNDNN); break;
    case F_SIN:  cu_mpc_sin  (r, x, CU_MPC_RNDNN); break;
    case F_COS:  cu_mpc_cos  (r, x, CU_MPC_RNDNN); break;
    case F_TAN:  cu_mpc_tan  (r, x, CU_MPC_RNDNN); break;
    case F_SINH: cu_mpc_sinh (r, x, CU_MPC_RNDNN); break;
    case F_COSH: cu_mpc_cosh (r, x, CU_MPC_RNDNN); break;
    case F_ASIN: cu_mpc_asin (r, x, CU_MPC_RNDNN); break;
    case F_ACOS: cu_mpc_acos (r, x, CU_MPC_RNDNN); break;
    case F_ATAN: cu_mpc_atan (r, x, CU_MPC_RNDNN); break;
    }
  *wr = cu_mpfr_get_d (cu_mpc_realref (r), CU_MPFR_RNDN);
  *wi = cu_mpfr_get_d (cu_mpc_imagref (r), CU_MPFR_RNDN);
}

__global__ void
bench_kernel (int fid, int n, const double *Zr, const double *Zi,
              double *Wr, double *Wi)
{
  int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    {
#ifdef __CUDA_ARCH__
      mpc_cuda_arena_reset ();
#endif
      apply_fn (fid, Zr[i], Zi[i], &Wr[i], &Wi[i]);
    }
}

int
main (void)
{
  cudaDeviceSetLimit (cudaLimitMallocHeapSize, (size_t) 256 * 1024 * 1024);
  cudaDeviceSetLimit (cudaLimitStackSize,      (size_t) 256 * 1024);

  size_t MV = (size_t) N * sizeof (double);
  double *Zr = (double *) malloc (MV), *Zi = (double *) malloc (MV);
  double *Wgr = (double *) malloc (MV), *Wgi = (double *) malloc (MV);
  double *Wcr = (double *) malloc (MV), *Wci = (double *) malloc (MV);
  for (int i = 0; i < N; ++i) { Zr[i] = 0.5 + (double) i / N; Zi[i] = 0.3 + (double) i / (2 * N); }

  double *dZr, *dZi, *dWr, *dWi;
  cudaMalloc (&dZr, MV); cudaMalloc (&dZi, MV);
  cudaMalloc (&dWr, MV); cudaMalloc (&dWi, MV);
  cudaMemcpy (dZr, Zr, MV, cudaMemcpyHostToDevice);
  cudaMemcpy (dZi, Zi, MV, cudaMemcpyHostToDevice);

  size_t ntot = (size_t) LBLOCKS * LTHREADS;
  char *arena; size_t *top;
  cudaMalloc (&arena, ntot * (size_t) SLAB);
  cudaMalloc (&top,   ntot * sizeof (size_t));
  cudaMemset (top, 0, ntot * sizeof (size_t));
  mpc_cuda_arena_base = arena; mpc_cuda_arena_slab = SLAB; mpc_cuda_arena_top = top;
  cudaDeviceSynchronize ();

  cudaEvent_t e0, e1; cudaEventCreate (&e0); cudaEventCreate (&e1);

  printf ("=== MPC complex elementary/transcendental benchmark "
          "(N=%d inputs near 1+0.3i, precision=%d bits/comp) ===\n", N, PREC);
  printf ("%-8s %12s %12s %10s %14s\n",
          "func", "GPU [ms]", "CPU [ms]", "speedup", "max rel diff");

  for (int f = 0; f < NFUNC; ++f)
    {
      bench_kernel<<<LBLOCKS, LTHREADS>>> (f, N, dZr, dZi, dWr, dWi);   /* warm-up */
      cudaError_t werr = cudaDeviceSynchronize ();
      if (werr != cudaSuccess) { fprintf (stderr, "%s warm-up kernel failed: %s\n", FNAME[f], cudaGetErrorString (werr)); return 1; }
      cudaEventRecord (e0);
      bench_kernel<<<LBLOCKS, LTHREADS>>> (f, N, dZr, dZi, dWr, dWi);
      cudaEventRecord (e1);
      cudaError_t err = cudaDeviceSynchronize ();
      if (err != cudaSuccess) { fprintf (stderr, "%s kernel failed: %s\n", FNAME[f], cudaGetErrorString (err)); return 1; }
      float gpu_ms = 0.f; cudaEventElapsedTime (&gpu_ms, e0, e1);
      cudaMemcpy (Wgr, dWr, MV, cudaMemcpyDeviceToHost);
      cudaMemcpy (Wgi, dWi, MV, cudaMemcpyDeviceToHost);

      struct timespec t0, t1;
      clock_gettime (CLOCK_MONOTONIC, &t0);
      for (int i = 0; i < N; ++i) apply_fn (f, Zr[i], Zi[i], &Wcr[i], &Wci[i]);
      clock_gettime (CLOCK_MONOTONIC, &t1);
      double cpu_ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) * 1e-6;

      double maxrel = 0.0;
      for (int i = 0; i < N; ++i)
        {
          double dr = Wcr[i] != 0.0 ? fabs ((Wgr[i] - Wcr[i]) / Wcr[i]) : fabs (Wgr[i]);
          double di = Wci[i] != 0.0 ? fabs ((Wgi[i] - Wci[i]) / Wci[i]) : fabs (Wgi[i]);
          double rel = dr > di ? dr : di; if (rel > maxrel) maxrel = rel;
        }
      printf ("%-8s %12.3f %12.3f %9.1fx %14.3e\n",
              FNAME[f], gpu_ms, cpu_ms, cpu_ms / gpu_ms, maxrel);
    }

  free (Zr); free (Zi); free (Wgr); free (Wgi); free (Wcr); free (Wci);
  cudaFree (dZr); cudaFree (dZi); cudaFree (dWr); cudaFree (dWi);
  cudaFree (arena); cudaFree (top);
  return 0;
}
