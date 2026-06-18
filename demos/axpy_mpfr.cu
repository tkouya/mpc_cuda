/* axpy_mpfr.cu -- multiple-precision real AXPY (y = a*x + y) with MPFR,
 * run on the GPU and on the CPU, comparing execution time and accuracy.
 *
 * Port of cump_sm121/demos/axpy.cu to the (faithful) MPFR semantics of this
 * library.  The CPU reference is the SAME ported MPFR running on the host.
 * The GPU launches a FIXED pool of threads and grid-strides over the N
 * elements; a per-thread bump arena of bounded size (LAUNCH*SLAB) backs the
 * limb allocations (tools/cudafy_minigmp.py), removing the device-malloc churn
 * that would otherwise dominate.  Build: `make axpy-mpfr`.
 *
 *   y_i <- a * x_i + y_i,   i = 0..N-1,   a scalar, x,y vectors, PREC bits.
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
typedef long int gmp_randstate_t[1];
#include "mpc_cuda.cuh"

#ifndef N
#define N 16384                      /* problem size (vector length) */
#endif
#ifndef PREC
#define PREC 1024
#endif
#ifndef SLAB
#define SLAB (32 * 1024)             /* arena bytes per resident thread */
#endif
#ifndef LBLOCKS
#define LBLOCKS 256
#endif
#ifndef LTHREADS
#define LTHREADS 64
#endif

__host__ __device__ static double
axpy_elt (double a, double x, double y)
{
  cu_mpfr_t ma, mx, my, t;
  cu_mpfr_init2 (ma, PREC); cu_mpfr_init2 (mx, PREC);
  cu_mpfr_init2 (my, PREC); cu_mpfr_init2 (t,  PREC);
  cu_mpfr_set_d (ma, a, CU_MPFR_RNDN);
  cu_mpfr_set_d (mx, x, CU_MPFR_RNDN);
  cu_mpfr_set_d (my, y, CU_MPFR_RNDN);
  cu_mpfr_mul (t, ma, mx, CU_MPFR_RNDN);    /* t = a*x     */
  cu_mpfr_add (my, t, my, CU_MPFR_RNDN);    /* y = a*x + y */
  double r = cu_mpfr_get_d (my, CU_MPFR_RNDN);
  cu_mpfr_clear (ma); cu_mpfr_clear (mx); cu_mpfr_clear (my); cu_mpfr_clear (t);
  return r;
}

__global__ void
axpy_kernel (int n, double a, const double *x, const double *y, double *out)
{
  int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    {
      mpc_cuda_arena_reset ();
      out[i] = axpy_elt (a, x[i], y[i]);
    }
}

int
main (void)
{
  cudaDeviceSetLimit (cudaLimitMallocHeapSize, (size_t) 64 * 1024 * 1024);
  cudaDeviceSetLimit (cudaLimitStackSize,      (size_t) 64 * 1024);

  double a = 1.5;
  double *x = (double *) malloc (N * sizeof (double));
  double *y = (double *) malloc (N * sizeof (double));
  double *gpu = (double *) malloc (N * sizeof (double));
  double *cpu = (double *) malloc (N * sizeof (double));
  for (int i = 0; i < N; ++i) { x[i] = 1.0 + i * 1e-3; y[i] = 2.0 - i * 7e-4; }

  double *dx, *dy, *dout;
  cudaMalloc (&dx, N*sizeof(double)); cudaMalloc (&dy, N*sizeof(double));
  cudaMalloc (&dout, N*sizeof(double));
  cudaMemcpy (dx, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (dy, y, N*sizeof(double), cudaMemcpyHostToDevice);

  size_t ntot = (size_t) LBLOCKS * LTHREADS;
  char *arena; size_t *top;
  cudaMalloc (&arena, ntot * (size_t) SLAB);
  cudaMalloc (&top,   ntot * sizeof (size_t));
  cudaMemset (top, 0, ntot * sizeof (size_t));
  mpc_cuda_arena_base = arena; mpc_cuda_arena_slab = SLAB; mpc_cuda_arena_top = top;
  cudaDeviceSynchronize ();

  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  axpy_kernel<<<LBLOCKS, LTHREADS>>> (N, a, dx, dy, dout);  /* warm-up */
  cudaDeviceSynchronize ();
  cudaEventRecord (e0);
  axpy_kernel<<<LBLOCKS, LTHREADS>>> (N, a, dx, dy, dout);
  cudaEventRecord (e1);
  cudaError_t err = cudaDeviceSynchronize ();
  if (err != cudaSuccess) { fprintf (stderr, "kernel failed: %s\n", cudaGetErrorString (err)); return 1; }
  float gpu_ms = 0.f; cudaEventElapsedTime (&gpu_ms, e0, e1);
  cudaMemcpy (gpu, dout, N*sizeof(double), cudaMemcpyDeviceToHost);

  struct timespec t0, t1;
  clock_gettime (CLOCK_MONOTONIC, &t0);
  for (int i = 0; i < N; ++i) cpu[i] = axpy_elt (a, x[i], y[i]);
  clock_gettime (CLOCK_MONOTONIC, &t1);
  double cpu_ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) * 1e-6;

  double maxrel = 0.0; int mism = 0;
  for (int i = 0; i < N; ++i)
    {
      double rel = cpu[i] != 0.0 ? fabs ((gpu[i] - cpu[i]) / cpu[i]) : fabs (gpu[i]);
      if (rel > maxrel) maxrel = rel;
      if (rel > 1e-13) ++mism;
    }

  printf ("=== MPFR real AXPY (y = a*x + y),  N=%d, precision=%d bits ===\n", N, PREC);
  printf ("launch   : %d blocks x %d threads = %zu resident; arena %zu MB\n",
          LBLOCKS, LTHREADS, ntot, (ntot * (size_t) SLAB) >> 20);
  printf ("GPU time : %8.3f ms   (per-thread arena + grid-stride)\n", gpu_ms);
  printf ("CPU time : %8.3f ms   (this library's MPFR on the host)\n", cpu_ms);
  printf ("speedup  : %8.2fx (GPU vs CPU)\n", cpu_ms / gpu_ms);
  printf ("accuracy : max relative |GPU-CPU| = %.3e over %d elems (%d exceed 1e-13)\n",
          maxrel, N, mism);
  printf ("sample   : y[0]=%.15g  y[1]=%.15g  y[N-1]=%.15g\n", gpu[0], gpu[1], gpu[N-1]);

  free (x); free (y); free (gpu); free (cpu);
  cudaFree (dx); cudaFree (dy); cudaFree (dout); cudaFree (arena); cudaFree (top);
  return mism ? 1 : 0;
}
