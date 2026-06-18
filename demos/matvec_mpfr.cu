/* matvec_mpfr.cu -- multiple-precision real matrix-vector product  y = A*x
 * with MPFR, run on the GPU and on the CPU, comparing time and accuracy.
 *
 * A is N x N (row-major), x and y are length N, PREC bits.  One output row is a
 * length-N dot product; the GPU grid-strides a fixed thread pool over the rows.
 * The dot-product accumulator and scratch are stack-backed cu_mpfr_t
 * (CU_MPFR_DECL_INIT), so the per-thread bump arena only has to hold ONE
 * multiply-add's temporaries and is reset every inner iteration -- this keeps
 * the arena tiny regardless of N.  Build: `make matvec-mpfr`.
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
typedef long int gmp_randstate_t[1];
#include "mpc_cuda.cuh"

#ifndef N
#define N 256
#endif
#ifndef PREC
#define PREC 1024
#endif
#ifndef SLAB
#define SLAB (16 * 1024)             /* arena bytes per thread (one madd) */
#endif
#ifndef LBLOCKS
#define LBLOCKS 128
#endif
#ifndef LTHREADS
#define LTHREADS 32
#endif

/* y_i = sum_j A[i*n + j] * x[j] */
__host__ __device__ static double
dot_row (const double *Arow, const double *x, int n)
{
  CU_MPFR_DECL_INIT (acc, PREC);
  CU_MPFR_DECL_INIT (a, PREC);
  CU_MPFR_DECL_INIT (b, PREC);
  CU_MPFR_DECL_INIT (t, PREC);
  cu_mpfr_set_zero (acc, 1);
  for (int j = 0; j < n; ++j)
    {
#ifdef __CUDA_ARCH__
      mpc_cuda_arena_reset ();        /* reclaim the previous madd's scratch */
#endif
      cu_mpfr_set_d (a, Arow[j], CU_MPFR_RNDN);
      cu_mpfr_set_d (b, x[j], CU_MPFR_RNDN);
      cu_mpfr_mul (t, a, b, CU_MPFR_RNDN);
      cu_mpfr_add (acc, acc, t, CU_MPFR_RNDN);
    }
  return cu_mpfr_get_d (acc, CU_MPFR_RNDN);
}

__global__ void
matvec_kernel (int n, const double *A, const double *x, double *y)
{
  int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    y[i] = dot_row (A + (size_t) i * n, x, n);
}

int
main (void)
{
  cudaDeviceSetLimit (cudaLimitMallocHeapSize, (size_t) 64 * 1024 * 1024);
  cudaDeviceSetLimit (cudaLimitStackSize,      (size_t) 64 * 1024);

  size_t MA = (size_t) N * N * sizeof (double), MV = (size_t) N * sizeof (double);
  double *A = (double *) malloc (MA);
  double *x = (double *) malloc (MV);
  double *yg = (double *) malloc (MV), *yc = (double *) malloc (MV);
  for (int i = 0; i < N; ++i)
    {
      x[i] = 1.0 + i * 1e-3;
      for (int j = 0; j < N; ++j) A[(size_t) i * N + j] = 1.0 + (i + 2 * j) * 1e-4;
    }

  double *dA, *dx, *dy;
  cudaMalloc (&dA, MA); cudaMalloc (&dx, MV); cudaMalloc (&dy, MV);
  cudaMemcpy (dA, A, MA, cudaMemcpyHostToDevice);
  cudaMemcpy (dx, x, MV, cudaMemcpyHostToDevice);

  size_t ntot = (size_t) LBLOCKS * LTHREADS;
  char *arena; size_t *top;
  cudaMalloc (&arena, ntot * (size_t) SLAB);
  cudaMalloc (&top,   ntot * sizeof (size_t));
  cudaMemset (top, 0, ntot * sizeof (size_t));
  mpc_cuda_arena_base = arena; mpc_cuda_arena_slab = SLAB; mpc_cuda_arena_top = top;
  cudaDeviceSynchronize ();

  cudaEvent_t e0, e1; cudaEventCreate (&e0); cudaEventCreate (&e1);
  matvec_kernel<<<LBLOCKS, LTHREADS>>> (N, dA, dx, dy);   /* warm-up */
  cudaDeviceSynchronize ();
  cudaEventRecord (e0);
  matvec_kernel<<<LBLOCKS, LTHREADS>>> (N, dA, dx, dy);
  cudaEventRecord (e1);
  cudaError_t err = cudaDeviceSynchronize ();
  if (err != cudaSuccess) { fprintf (stderr, "kernel failed: %s\n", cudaGetErrorString (err)); return 1; }
  float gpu_ms = 0.f; cudaEventElapsedTime (&gpu_ms, e0, e1);
  cudaMemcpy (yg, dy, MV, cudaMemcpyDeviceToHost);

  struct timespec t0, t1;
  clock_gettime (CLOCK_MONOTONIC, &t0);
  for (int i = 0; i < N; ++i) yc[i] = dot_row (A + (size_t) i * N, x, N);
  clock_gettime (CLOCK_MONOTONIC, &t1);
  double cpu_ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) * 1e-6;

  double maxrel = 0.0; int mism = 0;
  for (int i = 0; i < N; ++i)
    {
      double rel = yc[i] != 0.0 ? fabs ((yg[i] - yc[i]) / yc[i]) : fabs (yg[i]);
      if (rel > maxrel) maxrel = rel; if (rel > 1e-13) ++mism;
    }

  printf ("=== MPFR matrix-vector  y = A*x,  N=%d, precision=%d bits ===\n", N, PREC);
  printf ("GPU time : %8.3f ms   (per-thread arena + grid-stride)\n", gpu_ms);
  printf ("CPU time : %8.3f ms   (this library's MPFR on the host)\n", cpu_ms);
  printf ("speedup  : %8.2fx (GPU vs CPU)\n", cpu_ms / gpu_ms);
  printf ("accuracy : max relative |GPU-CPU| = %.3e over %d elems (%d exceed 1e-13)\n",
          maxrel, N, mism);
  printf ("sample   : y[0]=%.15g  y[N-1]=%.15g\n", yg[0], yg[N-1]);

  free (A); free (x); free (yg); free (yc);
  cudaFree (dA); cudaFree (dx); cudaFree (dy); cudaFree (arena); cudaFree (top);
  return mism ? 1 : 0;
}
