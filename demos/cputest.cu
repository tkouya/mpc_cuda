/* cputest.cu -- correctness test of this library's CUDA MPFR/MPC against the
 * *system* CPU libmpfr / libmpc, both linked into one binary.
 *
 * The GPU half (this file) evaluates a battery of real (MPFR) and complex (MPC)
 * elementary/transcendental functions inside CUDA kernels using the
 * cu_mpfr_* / cu_mpc_* API.  The CPU half (demos/cpu_ref.cpp) evaluates the same
 * with the system libmpfr / libmpc.  Both libraries are correctly rounded, so
 * the results must agree to the last bit of the returned double; the test fails
 * if any element differs.
 *
 * Build/run:  make cputest      (needs the system libmpfr/libmpc + headers)
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
typedef long int gmp_randstate_t[1];
#include "mpc_cuda.cuh"
#include "cpu_ref.h"              /* shared enums + CPU-reference prototypes    */

#ifndef N
#define N 256
#endif
#ifndef PREC
#define PREC 200
#endif

/* --- real (MPFR) ----------------------------------------------------------- */
#define RLIMBS ((PREC + 8 * (int) sizeof (cu_mp_limb_t) - 1) / (8 * (int) sizeof (cu_mp_limb_t)))

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
cu_mpfr_kernel (int n, const double *X, double *Y)   /* Y[f*n + i] = f(X[i]) */
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    for (int f = 0; f < CR_NMPFR; ++f)
      Y[f * n + i] = gpu_mpfr_eval (f, X[i]);
}

/* --- complex (MPC) --------------------------------------------------------- */
#define CLIMBS ((PREC + 8 * (int) sizeof (cu_mp_limb_t) - 1) / (8 * (int) sizeof (cu_mp_limb_t)))
#define CU_MPC_DECL_INIT(z)                                                       \
  cu_mpc_t z;                                                                     \
  cu_mp_limb_t z##_rl[CLIMBS], z##_il[CLIMBS];                                    \
  cu_mpfr_custom_init_set (cu_mpc_realref (z), CU_MPFR_ZERO_KIND, 0, PREC, z##_rl);     \
  cu_mpfr_custom_init_set (cu_mpc_imagref (z), CU_MPFR_ZERO_KIND, 0, PREC, z##_il)

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
cu_mpc_kernel (int n, const double *Zr, const double *Zi, double *Wr, double *Wi)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    for (int f = 0; f < CC_NMPC; ++f)
      gpu_mpc_eval (f, Zr[i], Zi[i], &Wr[f * n + i], &Wi[f * n + i]);
}

static int
report (const char *name, double maxrel, long exact, long total)
{
  int ok = (maxrel == 0.0);
  printf ("  %-6s  max rel diff %10.3e   bit-exact %ld/%ld   %s\n",
          name, maxrel, exact, total, ok ? "OK" : "MISMATCH");
  return ok;
}

int
main (void)
{
  cudaDeviceSetLimit (cudaLimitMallocHeapSize, (size_t) 512 * 1024 * 1024);
  cudaDeviceSetLimit (cudaLimitStackSize,      (size_t) 192 * 1024);

  int cpu_threads = cpu_set_max_threads ();   /* run the CPU reference on every core */

  printf ("=== GPU cu_mpfr/cu_mpc  vs  system CPU libmpfr/libmpc ===\n");
  printf ("    system libmpfr %s, libmpc %s   (N=%d inputs, precision=%d bits)\n",
          cpu_mpfr_version (), cpu_mpc_version (), N, PREC);
  printf ("    CPU reference: OpenMP on %d thread%s\n\n",
          cpu_threads, cpu_threads == 1 ? "" : "s");

  /* inputs in (0.5, 1.5): inside the domain of every function tested */
  double *X  = (double *) malloc (N * sizeof (double));
  double *Zi = (double *) malloc (N * sizeof (double));
  for (int i = 0; i < N; ++i) { X[i] = 0.5 + (double) i / N; Zi[i] = 0.25 + (double) i / (2 * N); }

  int allok = 1;

  /* ---- real ---- */
  {
    size_t MV = (size_t) N * sizeof (double), MA = (size_t) CR_NMPFR * MV;
    double *dX, *dY; cudaMalloc (&dX, MV); cudaMalloc (&dY, MA);
    cudaMemcpy (dX, X, MV, cudaMemcpyHostToDevice);
    cu_mpfr_kernel<<<(N + 63) / 64, 64>>> (N, dX, dY);
    cudaError_t e = cudaDeviceSynchronize ();
    if (e != cudaSuccess) { fprintf (stderr, "mpfr kernel failed: %s\n", cudaGetErrorString (e)); return 1; }
    double *Yg = (double *) malloc (MA); cudaMemcpy (Yg, dY, MA, cudaMemcpyDeviceToHost);
    double *Yc = (double *) malloc (MV);            /* CPU reference, one func at a time */

    printf ("MPFR (real):\n");
    for (int f = 0; f < CR_NMPFR; ++f)
      {
        cpu_mpfr_eval_array (f, PREC, N, X, Yc);    /* OpenMP over all cores */
        double maxrel = 0.0; long exact = 0;
        for (int i = 0; i < N; ++i)
          {
            double c = Yc[i], g = Yg[f * N + i];
            if (g == c) ++exact;
            double rel = c != 0.0 ? fabs ((g - c) / c) : fabs (g);
            if (rel > maxrel) maxrel = rel;
          }
        allok &= report (CR_NAME[f], maxrel, exact, N);
      }
    free (Yc); free (Yg); cudaFree (dX); cudaFree (dY);
  }

  /* ---- complex ---- */
  {
    size_t MV = (size_t) N * sizeof (double), MA = (size_t) CC_NMPC * MV;
    double *dZr, *dZi, *dWr, *dWi;
    cudaMalloc (&dZr, MV); cudaMalloc (&dZi, MV); cudaMalloc (&dWr, MA); cudaMalloc (&dWi, MA);
    cudaMemcpy (dZr, X,  MV, cudaMemcpyHostToDevice);
    cudaMemcpy (dZi, Zi, MV, cudaMemcpyHostToDevice);
    cu_mpc_kernel<<<(N + 63) / 64, 64>>> (N, dZr, dZi, dWr, dWi);
    cudaError_t e = cudaDeviceSynchronize ();
    if (e != cudaSuccess) { fprintf (stderr, "mpc kernel failed: %s\n", cudaGetErrorString (e)); return 1; }
    double *Wgr = (double *) malloc (MA), *Wgi = (double *) malloc (MA);
    double *Wcr = (double *) malloc (MV), *Wci = (double *) malloc (MV);
    cudaMemcpy (Wgr, dWr, MA, cudaMemcpyDeviceToHost);
    cudaMemcpy (Wgi, dWi, MA, cudaMemcpyDeviceToHost);

    printf ("\nMPC (complex):\n");
    for (int f = 0; f < CC_NMPC; ++f)
      {
        cpu_mpc_eval_array (f, PREC, N, X, Zi, Wcr, Wci);   /* OpenMP over all cores */
        double maxrel = 0.0; long exact = 0;
        for (int i = 0; i < N; ++i)
          {
            double cr = Wcr[i], ci = Wci[i];
            double gr = Wgr[f * N + i], gi = Wgi[f * N + i];
            if (gr == cr && gi == ci) ++exact;
            double dr = cr != 0.0 ? fabs ((gr - cr) / cr) : fabs (gr);
            double di = ci != 0.0 ? fabs ((gi - ci) / ci) : fabs (gi);
            double rel = dr > di ? dr : di;
            if (rel > maxrel) maxrel = rel;
          }
        allok &= report (CC_NAME[f], maxrel, exact, N);
      }
    free (Wcr); free (Wci); free (Wgr); free (Wgi);
    cudaFree (dZr); cudaFree (dZi); cudaFree (dWr); cudaFree (dWi);
  }

  free (X); free (Zi);
  printf ("\n%s\n", allok ? "ALL MATCH -- GPU port agrees with the system CPU libraries"
                          : "SOME MISMATCH");
  return allok ? 0 : 1;
}
