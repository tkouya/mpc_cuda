/* matmul_mpfr.cu -- multiple-precision real matrix multiply  C = A*B  with
 * MPFR, run on the GPU and on the CPU, comparing time and accuracy.
 *
 * A, B, C are N x N (row-major), PREC bits.  C[i][j] is a length-N dot product
 * over A's row i and B's column j; the GPU grid-strides a fixed thread pool
 * over the N*N output elements.  Stack-backed accumulator (CU_MPFR_DECL_INIT) +
 * per-iteration arena reset keep the arena tiny.  Build: `make matmul-mpfr`.
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
typedef long int gmp_randstate_t[1];
#include "mpc_cuda.cuh"

#ifndef N
#define N 96
#endif
#ifndef PREC
#define PREC 1024
#endif
#ifndef SLAB
#define SLAB (16 * 1024)
#endif
#ifndef LBLOCKS
#define LBLOCKS 256
#endif
#ifndef LTHREADS
#define LTHREADS 32
#endif

/* C[i][j] = sum_k A[i*n+k] * B[k*n+j] */
__host__ __device__ static double
dot_ij (const double *A, const double *B, int n, int i, int j)
{
  CU_MPFR_DECL_INIT (acc, PREC);
  CU_MPFR_DECL_INIT (a, PREC);
  CU_MPFR_DECL_INIT (b, PREC);
  CU_MPFR_DECL_INIT (t, PREC);
  cu_mpfr_set_zero (acc, 1);
  for (int k = 0; k < n; ++k)
    {
#ifdef __CUDA_ARCH__
      mpc_cuda_arena_reset ();
#endif
      cu_mpfr_set_d (a, A[(size_t) i * n + k], CU_MPFR_RNDN);
      cu_mpfr_set_d (b, B[(size_t) k * n + j], CU_MPFR_RNDN);
      cu_mpfr_mul (t, a, b, CU_MPFR_RNDN);
      cu_mpfr_add (acc, acc, t, CU_MPFR_RNDN);
    }
  return cu_mpfr_get_d (acc, CU_MPFR_RNDN);
}

__global__ void
matmul_kernel (int n, const double *A, const double *B, double *C)
{
  int stride = gridDim.x * blockDim.x;
  long total = (long) n * n;
  for (long e = blockIdx.x * blockDim.x + threadIdx.x; e < total; e += stride)
    C[e] = dot_ij (A, B, n, (int) (e / n), (int) (e % n));
}

int
main (void)
{
  cudaDeviceSetLimit (cudaLimitMallocHeapSize, (size_t) 64 * 1024 * 1024);
  cudaDeviceSetLimit (cudaLimitStackSize,      (size_t) 64 * 1024);

  size_t M = (size_t) N * N * sizeof (double);
  double *A = (double *) malloc (M), *B = (double *) malloc (M);
  double *Cg = (double *) malloc (M), *Cc = (double *) malloc (M);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      {
        A[(size_t) i * N + j] = 1.0 + (i + 2 * j) * 1e-4;
        B[(size_t) i * N + j] = 0.5 + (2 * i + j) * 1e-4;
      }

  double *dA, *dB, *dC;
  cudaMalloc (&dA, M); cudaMalloc (&dB, M); cudaMalloc (&dC, M);
  cudaMemcpy (dA, A, M, cudaMemcpyHostToDevice);
  cudaMemcpy (dB, B, M, cudaMemcpyHostToDevice);

  size_t ntot = (size_t) LBLOCKS * LTHREADS;
  char *arena; size_t *top;
  cudaMalloc (&arena, ntot * (size_t) SLAB);
  cudaMalloc (&top,   ntot * sizeof (size_t));
  cudaMemset (top, 0, ntot * sizeof (size_t));
  mpc_cuda_arena_base = arena; mpc_cuda_arena_slab = SLAB; mpc_cuda_arena_top = top;
  cudaDeviceSynchronize ();

  cudaEvent_t e0, e1; cudaEventCreate (&e0); cudaEventCreate (&e1);
  matmul_kernel<<<LBLOCKS, LTHREADS>>> (N, dA, dB, dC);   /* warm-up */
  cudaDeviceSynchronize ();
  cudaEventRecord (e0);
  matmul_kernel<<<LBLOCKS, LTHREADS>>> (N, dA, dB, dC);
  cudaEventRecord (e1);
  cudaError_t err = cudaDeviceSynchronize ();
  if (err != cudaSuccess) { fprintf (stderr, "kernel failed: %s\n", cudaGetErrorString (err)); return 1; }
  float gpu_ms = 0.f; cudaEventElapsedTime (&gpu_ms, e0, e1);
  cudaMemcpy (Cg, dC, M, cudaMemcpyDeviceToHost);

  struct timespec t0, t1;
  clock_gettime (CLOCK_MONOTONIC, &t0);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      Cc[(size_t) i * N + j] = dot_ij (A, B, N, i, j);
  clock_gettime (CLOCK_MONOTONIC, &t1);
  double cpu_ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) * 1e-6;

  double maxrel = 0.0; int mism = 0; long tot = (long) N * N;
  for (long e = 0; e < tot; ++e)
    {
      double rel = Cc[e] != 0.0 ? fabs ((Cg[e] - Cc[e]) / Cc[e]) : fabs (Cg[e]);
      if (rel > maxrel) maxrel = rel; if (rel > 1e-13) ++mism;
    }

  printf ("=== MPFR matrix multiply  C = A*B,  N=%d, precision=%d bits ===\n", N, PREC);
  printf ("GPU time : %8.3f ms   (per-thread arena + grid-stride)\n", gpu_ms);
  printf ("CPU time : %8.3f ms   (this library's MPFR on the host)\n", cpu_ms);
  printf ("speedup  : %8.2fx (GPU vs CPU)\n", cpu_ms / gpu_ms);
  printf ("accuracy : max relative |GPU-CPU| = %.3e over %ld elems (%d exceed 1e-13)\n",
          maxrel, tot, mism);
  printf ("sample   : C[0][0]=%.15g  C[N-1][N-1]=%.15g\n", Cg[0], Cg[tot - 1]);

  free (A); free (B); free (Cg); free (Cc);
  cudaFree (dA); cudaFree (dB); cudaFree (dC); cudaFree (arena); cudaFree (top);
  return mism ? 1 : 0;
}
