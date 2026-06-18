/* cpubench.cu -- benchmark this library's CUDA MPFR/MPC on the GPU against the
 * *system* CPU libmpfr / libmpc, both linked into one binary.
 *
 * For each real (MPFR) and complex (MPC) elementary/transcendental function,
 * evaluate y[i]=f(x[i]) over N inputs on the GPU (fixed thread pool, grid-
 * stride, per-element bump-arena reset) and on the CPU with the *system*
 * libraries, and report GPU time, CPU time, speedup, and the max relative
 * difference (which is ~0 because both are correctly rounded).
 *
 * Build/run:  make cpubench     (needs the system libmpfr/libmpc + headers)
 * Tunables (compile with -DNAME=val): N, PREC, SLAB, LBLOCKS, LTHREADS.
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
typedef long int gmp_randstate_t[1];
#include "mpc_cuda.cuh"
#include "cpu_ref.h"

#ifndef N
#define N 4096
#endif
#ifndef PREC
#define PREC 1024
#endif
#ifndef SLAB
#define SLAB (2 * 1024 * 1024)        /* sized for a complex transcendental eval */
#endif
#ifndef LBLOCKS
#define LBLOCKS 256
#endif
#ifndef LTHREADS
#define LTHREADS 32
#endif

#define NLIMBS ((PREC + 8 * (int) sizeof (cu_mp_limb_t) - 1) / (8 * (int) sizeof (cu_mp_limb_t)))
#define CU_MPC_DECL_INIT(z)                                                       \
  cu_mpc_t z;                                                                     \
  cu_mp_limb_t z##_rl[NLIMBS], z##_il[NLIMBS];                                    \
  cu_mpfr_custom_init_set (cu_mpc_realref (z), CU_MPFR_ZERO_KIND, 0, PREC, z##_rl);     \
  cu_mpfr_custom_init_set (cu_mpc_imagref (z), CU_MPFR_ZERO_KIND, 0, PREC, z##_il)

__host__ __device__ static double
gpu_mpfr_eval (int fid, double xv)
{
  CU_MPFR_DECL_INIT (x, PREC);
  CU_MPFR_DECL_INIT (r, PREC);
  cu_mpfr_set_d (x, xv, CU_MPFR_RNDN);
  switch (fid)
    {
    case CR_SQRT:  cu_mpfr_sqrt  (r, x, CU_MPFR_RNDN); break;
    case CR_CBRT:  cu_mpfr_cbrt  (r, x, CU_MPFR_RNDN); break;
    case CR_EXP:   cu_mpfr_exp   (r, x, CU_MPFR_RNDN); break;
    case CR_EXPM1: cu_mpfr_expm1 (r, x, CU_MPFR_RNDN); break;
    case CR_LOG:   cu_mpfr_log   (r, x, CU_MPFR_RNDN); break;
    case CR_LOG1P: cu_mpfr_log1p (r, x, CU_MPFR_RNDN); break;
    case CR_SIN:   cu_mpfr_sin   (r, x, CU_MPFR_RNDN); break;
    case CR_COS:   cu_mpfr_cos   (r, x, CU_MPFR_RNDN); break;
    case CR_TAN:   cu_mpfr_tan   (r, x, CU_MPFR_RNDN); break;
    case CR_ATAN:  cu_mpfr_atan  (r, x, CU_MPFR_RNDN); break;
    case CR_SINH:  cu_mpfr_sinh  (r, x, CU_MPFR_RNDN); break;
    case CR_COSH:  cu_mpfr_cosh  (r, x, CU_MPFR_RNDN); break;
    }
  return cu_mpfr_get_d (r, CU_MPFR_RNDN);
}

__global__ void
cu_mpfr_kernel (int fid, int n, const double *X, double *Y)
{
  int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    {
#ifdef __CUDA_ARCH__
      mpc_cuda_arena_reset ();
#endif
      Y[i] = gpu_mpfr_eval (fid, X[i]);
    }
}

__host__ __device__ static void
gpu_mpc_eval (int fid, double zr, double zi, double *wr, double *wi)
{
  CU_MPC_DECL_INIT (x); CU_MPC_DECL_INIT (r);
  cu_mpc_set_d_d (x, zr, zi, CU_MPC_RNDNN);
  switch (fid)
    {
    case CC_SQR:  cu_mpc_sqr  (r, x, CU_MPC_RNDNN); break;
    case CC_SQRT: cu_mpc_sqrt (r, x, CU_MPC_RNDNN); break;
    case CC_EXP:  cu_mpc_exp  (r, x, CU_MPC_RNDNN); break;
    case CC_LOG:  cu_mpc_log  (r, x, CU_MPC_RNDNN); break;
    case CC_SIN:  cu_mpc_sin  (r, x, CU_MPC_RNDNN); break;
    case CC_COS:  cu_mpc_cos  (r, x, CU_MPC_RNDNN); break;
    case CC_TAN:  cu_mpc_tan  (r, x, CU_MPC_RNDNN); break;
    case CC_SINH: cu_mpc_sinh (r, x, CU_MPC_RNDNN); break;
    case CC_COSH: cu_mpc_cosh (r, x, CU_MPC_RNDNN); break;
    case CC_ASIN: cu_mpc_asin (r, x, CU_MPC_RNDNN); break;
    case CC_ACOS: cu_mpc_acos (r, x, CU_MPC_RNDNN); break;
    case CC_ATAN: cu_mpc_atan (r, x, CU_MPC_RNDNN); break;
    }
  *wr = cu_mpfr_get_d (cu_mpc_realref (r), CU_MPFR_RNDN);
  *wi = cu_mpfr_get_d (cu_mpc_imagref (r), CU_MPFR_RNDN);
}

__global__ void
cu_mpc_kernel (int fid, int n, const double *Zr, const double *Zi,
            double *Wr, double *Wi)
{
  int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    {
#ifdef __CUDA_ARCH__
      mpc_cuda_arena_reset ();
#endif
      gpu_mpc_eval (fid, Zr[i], Zi[i], &Wr[i], &Wi[i]);
    }
}

static double
ms_since (struct timespec t0)
{
  struct timespec t1; clock_gettime (CLOCK_MONOTONIC, &t1);
  return (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) * 1e-6;
}

int
main (void)
{
  cudaDeviceSetLimit (cudaLimitMallocHeapSize, (size_t) 256 * 1024 * 1024);
  cudaDeviceSetLimit (cudaLimitStackSize,      (size_t) 192 * 1024);

  size_t MV = (size_t) N * sizeof (double);
  double *X  = (double *) malloc (MV), *Zi = (double *) malloc (MV);
  double *Yg = (double *) malloc (MV), *Yc = (double *) malloc (MV);
  double *Wgr = (double *) malloc (MV), *Wgi = (double *) malloc (MV);
  double *Wcr = (double *) malloc (MV), *Wci = (double *) malloc (MV);
  for (int i = 0; i < N; ++i) { X[i] = 0.5 + (double) i / N; Zi[i] = 0.25 + (double) i / (2 * N); }

  double *dX, *dY, *dZi, *dWr, *dWi;
  cudaMalloc (&dX, MV); cudaMalloc (&dY, MV); cudaMalloc (&dZi, MV);
  cudaMalloc (&dWr, MV); cudaMalloc (&dWi, MV);
  cudaMemcpy (dX,  X,  MV, cudaMemcpyHostToDevice);
  cudaMemcpy (dZi, Zi, MV, cudaMemcpyHostToDevice);

  /* per-thread bump arena so the hot path never touches the device heap */
  size_t ntot = (size_t) LBLOCKS * LTHREADS;
  char *arena; size_t *top;
  cudaMalloc (&arena, ntot * (size_t) SLAB);
  cudaMalloc (&top,   ntot * sizeof (size_t));
  cudaMemset (top, 0, ntot * sizeof (size_t));
  mpc_cuda_arena_base = arena; mpc_cuda_arena_slab = SLAB; mpc_cuda_arena_top = top;
  cudaDeviceSynchronize ();

  cudaEvent_t e0, e1; cudaEventCreate (&e0); cudaEventCreate (&e1);

  int cpu_threads = cpu_set_max_threads ();   /* run the CPU side on every core */

  printf ("=== GPU cu_mpfr/cu_mpc  vs  system CPU libmpfr/libmpc ===\n");
  printf ("    system libmpfr %s, libmpc %s   (N=%d inputs, precision=%d bits)\n",
          cpu_mpfr_version (), cpu_mpc_version (), N, PREC);
  printf ("    CPU side: OpenMP on %d thread%s\n\n",
          cpu_threads, cpu_threads == 1 ? "" : "s");

  printf ("MPFR (real):\n");
  printf ("%-8s %12s %12s %10s %14s\n", "func", "GPU [ms]", "CPU [ms]", "speedup", "max rel diff");
  for (int f = 0; f < CR_NMPFR; ++f)
    {
      cu_mpfr_kernel<<<LBLOCKS, LTHREADS>>> (f, N, dX, dY);     /* warm-up */
      cudaDeviceSynchronize ();
      cudaEventRecord (e0);
      cu_mpfr_kernel<<<LBLOCKS, LTHREADS>>> (f, N, dX, dY);
      cudaEventRecord (e1);
      cudaError_t err = cudaDeviceSynchronize ();
      if (err != cudaSuccess) { fprintf (stderr, "%s kernel failed: %s\n", CR_NAME[f], cudaGetErrorString (err)); return 1; }
      float gpu_ms = 0.f; cudaEventElapsedTime (&gpu_ms, e0, e1);
      cudaMemcpy (Yg, dY, MV, cudaMemcpyDeviceToHost);

      struct timespec t0; clock_gettime (CLOCK_MONOTONIC, &t0);
      cpu_mpfr_eval_array (f, PREC, N, X, Yc);     /* OpenMP over all cores */
      double cpu_ms = ms_since (t0);

      double maxrel = 0.0;
      for (int i = 0; i < N; ++i)
        { double rel = Yc[i] != 0.0 ? fabs ((Yg[i] - Yc[i]) / Yc[i]) : fabs (Yg[i]);
          if (rel > maxrel) maxrel = rel; }
      printf ("%-8s %12.3f %12.3f %9.1fx %14.3e\n",
              CR_NAME[f], gpu_ms, cpu_ms, cpu_ms / gpu_ms, maxrel);
    }

  printf ("\nMPC (complex):\n");
  printf ("%-8s %12s %12s %10s %14s\n", "func", "GPU [ms]", "CPU [ms]", "speedup", "max rel diff");
  for (int f = 0; f < CC_NMPC; ++f)
    {
      cu_mpc_kernel<<<LBLOCKS, LTHREADS>>> (f, N, dX, dZi, dWr, dWi);  /* warm-up */
      cudaDeviceSynchronize ();
      cudaEventRecord (e0);
      cu_mpc_kernel<<<LBLOCKS, LTHREADS>>> (f, N, dX, dZi, dWr, dWi);
      cudaEventRecord (e1);
      cudaError_t err = cudaDeviceSynchronize ();
      if (err != cudaSuccess) { fprintf (stderr, "%s kernel failed: %s\n", CC_NAME[f], cudaGetErrorString (err)); return 1; }
      float gpu_ms = 0.f; cudaEventElapsedTime (&gpu_ms, e0, e1);
      cudaMemcpy (Wgr, dWr, MV, cudaMemcpyDeviceToHost);
      cudaMemcpy (Wgi, dWi, MV, cudaMemcpyDeviceToHost);

      struct timespec t0; clock_gettime (CLOCK_MONOTONIC, &t0);
      cpu_mpc_eval_array (f, PREC, N, X, Zi, Wcr, Wci);   /* OpenMP over all cores */
      double cpu_ms = ms_since (t0);

      double maxrel = 0.0;
      for (int i = 0; i < N; ++i)
        {
          double cr = Wcr[i], ci = Wci[i];
          double dr = cr != 0.0 ? fabs ((Wgr[i] - cr) / cr) : fabs (Wgr[i]);
          double di = ci != 0.0 ? fabs ((Wgi[i] - ci) / ci) : fabs (Wgi[i]);
          double rel = dr > di ? dr : di;
          if (rel > maxrel) maxrel = rel;
        }
      printf ("%-8s %12.3f %12.3f %9.1fx %14.3e\n",
              CC_NAME[f], gpu_ms, cpu_ms, cpu_ms / gpu_ms, maxrel);
    }

  free (X); free (Zi); free (Yg); free (Yc); free (Wgr); free (Wgi); free (Wcr); free (Wci);
  cudaFree (dX); cudaFree (dY); cudaFree (dZi); cudaFree (dWr); cudaFree (dWi);
  cudaFree (arena); cudaFree (top);
  return 0;
}
