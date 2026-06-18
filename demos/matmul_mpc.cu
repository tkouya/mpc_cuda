/* matmul_mpc.cu -- multiple-precision COMPLEX matrix multiply  C = A*B  with
 * MPC, run on the GPU and on the CPU, comparing time and accuracy.
 *
 * Complex analogue of matmul_mpfr.cu.  Stack-backed cu_mpc_t accumulator/scratch
 * (CU_MPC_DECL_INIT) + per-iteration arena reset; the GPU grid-strides a fixed
 * thread pool over the N*N output elements.  Build: `make matmul-mpc`.
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
typedef long int gmp_randstate_t[1];
#include "mpc_cuda.cuh"

#ifndef N
#define N 64
#endif
#ifndef PREC
#define PREC 1024
#endif
#ifndef SLAB
#define SLAB (64 * 1024)
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

/* C[i,j] = sum_k A[i,k] * B[k,j]  (complex) */
__host__ __device__ static void
cdot_ij (const double *Ar, const double *Ai, const double *Br, const double *Bi,
         int n, int i, int j, double *cr, double *ci)
{
  CU_MPC_DECL_INIT (acc); CU_MPC_DECL_INIT (a); CU_MPC_DECL_INIT (b); CU_MPC_DECL_INIT (t);
  for (int k = 0; k < n; ++k)
    {
#ifdef __CUDA_ARCH__
      mpc_cuda_arena_reset ();
#endif
      size_t ia = (size_t) i * n + k, ib = (size_t) k * n + j;
      cu_mpc_set_d_d (a, Ar[ia], Ai[ia], CU_MPC_RNDNN);
      cu_mpc_set_d_d (b, Br[ib], Bi[ib], CU_MPC_RNDNN);
      cu_mpc_mul (t, a, b, CU_MPC_RNDNN);
      cu_mpc_add (acc, acc, t, CU_MPC_RNDNN);
    }
  *cr = cu_mpfr_get_d (cu_mpc_realref (acc), CU_MPFR_RNDN);
  *ci = cu_mpfr_get_d (cu_mpc_imagref (acc), CU_MPFR_RNDN);
}

__global__ void
matmul_kernel (int n, const double *Ar, const double *Ai,
               const double *Br, const double *Bi, double *Cr, double *Ci)
{
  int stride = gridDim.x * blockDim.x;
  long total = (long) n * n;
  for (long e = blockIdx.x * blockDim.x + threadIdx.x; e < total; e += stride)
    cdot_ij (Ar, Ai, Br, Bi, n, (int) (e / n), (int) (e % n), &Cr[e], &Ci[e]);
}

int
main (void)
{
  cudaDeviceSetLimit (cudaLimitMallocHeapSize, (size_t) 64 * 1024 * 1024);
  cudaDeviceSetLimit (cudaLimitStackSize,      (size_t) 160 * 1024);

  size_t M = (size_t) N * N * sizeof (double);
  double *Ar = (double *) malloc (M), *Ai = (double *) malloc (M);
  double *Br = (double *) malloc (M), *Bi = (double *) malloc (M);
  double *Cgr = (double *) malloc (M), *Cgi = (double *) malloc (M);
  double *Ccr = (double *) malloc (M), *Cci = (double *) malloc (M);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      {
        size_t e = (size_t) i * N + j;
        Ar[e] = 1.0 + (i + 2 * j) * 1e-4;  Ai[e] = -0.5 + (2 * i + j) * 1e-4;
        Br[e] = 0.5 + (2 * i + j) * 1e-4;  Bi[e] =  0.25 - (i + j) * 1e-4;
      }

  double *dAr,*dAi,*dBr,*dBi,*dCr,*dCi;
  cudaMalloc (&dAr, M); cudaMalloc (&dAi, M); cudaMalloc (&dBr, M);
  cudaMalloc (&dBi, M); cudaMalloc (&dCr, M); cudaMalloc (&dCi, M);
  cudaMemcpy (dAr, Ar, M, cudaMemcpyHostToDevice); cudaMemcpy (dAi, Ai, M, cudaMemcpyHostToDevice);
  cudaMemcpy (dBr, Br, M, cudaMemcpyHostToDevice); cudaMemcpy (dBi, Bi, M, cudaMemcpyHostToDevice);

  size_t ntot = (size_t) LBLOCKS * LTHREADS;
  char *arena; size_t *top;
  cudaMalloc (&arena, ntot * (size_t) SLAB);
  cudaMalloc (&top,   ntot * sizeof (size_t));
  cudaMemset (top, 0, ntot * sizeof (size_t));
  mpc_cuda_arena_base = arena; mpc_cuda_arena_slab = SLAB; mpc_cuda_arena_top = top;
  cudaDeviceSynchronize ();

  cudaEvent_t e0, e1; cudaEventCreate (&e0); cudaEventCreate (&e1);
  matmul_kernel<<<LBLOCKS, LTHREADS>>> (N, dAr, dAi, dBr, dBi, dCr, dCi);  /* warm-up */
  cudaDeviceSynchronize ();
  cudaEventRecord (e0);
  matmul_kernel<<<LBLOCKS, LTHREADS>>> (N, dAr, dAi, dBr, dBi, dCr, dCi);
  cudaEventRecord (e1);
  cudaError_t err = cudaDeviceSynchronize ();
  if (err != cudaSuccess) { fprintf (stderr, "kernel failed: %s\n", cudaGetErrorString (err)); return 1; }
  float gpu_ms = 0.f; cudaEventElapsedTime (&gpu_ms, e0, e1);
  cudaMemcpy (Cgr, dCr, M, cudaMemcpyDeviceToHost); cudaMemcpy (Cgi, dCi, M, cudaMemcpyDeviceToHost);

  struct timespec t0, t1;
  clock_gettime (CLOCK_MONOTONIC, &t0);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      cdot_ij (Ar, Ai, Br, Bi, N, i, j, &Ccr[(size_t) i * N + j], &Cci[(size_t) i * N + j]);
  clock_gettime (CLOCK_MONOTONIC, &t1);
  double cpu_ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) * 1e-6;

  double maxrel = 0.0; int mism = 0; long tot = (long) N * N;
  for (long e = 0; e < tot; ++e)
    {
      double dr = Ccr[e] != 0.0 ? fabs ((Cgr[e] - Ccr[e]) / Ccr[e]) : fabs (Cgr[e]);
      double di = Cci[e] != 0.0 ? fabs ((Cgi[e] - Cci[e]) / Cci[e]) : fabs (Cgi[e]);
      double rel = dr > di ? dr : di; if (rel > maxrel) maxrel = rel; if (rel > 1e-13) ++mism;
    }

  printf ("=== MPC complex matrix multiply  C = A*B,  N=%d, precision=%d bits/comp ===\n", N, PREC);
  printf ("GPU time : %8.3f ms   (per-thread arena + grid-stride)\n", gpu_ms);
  printf ("CPU time : %8.3f ms   (this library's MPC on the host)\n", cpu_ms);
  printf ("speedup  : %8.2fx (GPU vs CPU)\n", cpu_ms / gpu_ms);
  printf ("accuracy : max relative |GPU-CPU| = %.3e over %ld elems (%d exceed 1e-13)\n",
          maxrel, tot, mism);
  printf ("sample   : C[0][0]=(%.12g, %.12g)\n", Cgr[0], Cgi[0]);

  return mism ? 1 : 0;
}
