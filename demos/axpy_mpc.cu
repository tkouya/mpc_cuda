/* axpy_mpc.cu -- multiple-precision COMPLEX AXPY (y = a*x + y) with MPC,
 * run on the GPU and on the CPU, comparing execution time and accuracy.
 *
 * Complex analogue of axpy_mpfr.cu.  The CPU reference is this library's MPC
 * running on the host (same __host__ __device__ code).
 *
 * The GPU launches a FIXED pool of threads and grid-strides over the N
 * elements, so a per-thread bump arena of bounded size (LAUNCH*SLAB) backs the
 * limb allocations regardless of N.  Enough resident threads are needed to hide
 * the global-memory latency of the deep cu_mpc_mul -> cu_mpfr_fmms -> ... chain; with
 * that, the GPU is dramatically faster than the (sequential) CPU.
 *
 *   y_i <- a * x_i + y_i  (complex),  i = 0..N-1,  PREC bits per component.
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
#define SLAB (256 * 1024)            /* arena bytes per resident thread */
#endif
#ifndef LBLOCKS
#define LBLOCKS 256                  /* launched grid: LBLOCKS*LTHREADS threads */
#endif
#ifndef LTHREADS
#define LTHREADS 32
#endif

__host__ __device__ static void
axpy_elt (double ar, double ai, double xr, double xi,
          double yr, double yi, double *orr, double *ori)
{
  cu_mpc_t a, x, y, t;
  cu_mpc_init2 (a, PREC); cu_mpc_init2 (x, PREC);
  cu_mpc_init2 (y, PREC); cu_mpc_init2 (t, PREC);
  cu_mpc_set_d_d (a, ar, ai, CU_MPC_RNDNN);
  cu_mpc_set_d_d (x, xr, xi, CU_MPC_RNDNN);
  cu_mpc_set_d_d (y, yr, yi, CU_MPC_RNDNN);
  cu_mpc_mul (t, a, x, CU_MPC_RNDNN);     /* t = a*x      */
  cu_mpc_add (y, t, y, CU_MPC_RNDNN);     /* y = a*x + y  */
  *orr = cu_mpfr_get_d (cu_mpc_realref (y), CU_MPFR_RNDN);
  *ori = cu_mpfr_get_d (cu_mpc_imagref (y), CU_MPFR_RNDN);
  cu_mpc_clear (a); cu_mpc_clear (x); cu_mpc_clear (y); cu_mpc_clear (t);
}

__global__ void
axpy_kernel (int n, double ar, double ai,
             const double *xr, const double *xi,
             const double *yr, const double *yi,
             double *orr, double *ori)
{
  int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    {
      mpc_cuda_arena_reset ();                /* reclaim this thread's slab */
      axpy_elt (ar, ai, xr[i], xi[i], yr[i], yi[i], &orr[i], &ori[i]);
    }
}

int
main (void)
{
  cudaDeviceSetLimit (cudaLimitMallocHeapSize, (size_t) 64 * 1024 * 1024);
  cudaDeviceSetLimit (cudaLimitStackSize,      (size_t) 128 * 1024);

  double ar = 1.5, ai = -0.5;
  size_t B = N * sizeof (double);
  double *xr=(double*)malloc(B), *xi=(double*)malloc(B);
  double *yr=(double*)malloc(B), *yi=(double*)malloc(B);
  double *gr=(double*)malloc(B), *gi=(double*)malloc(B);
  double *cr=(double*)malloc(B), *ci=(double*)malloc(B);
  for (int i = 0; i < N; ++i)
    { xr[i]=1.0+i*1e-3; xi[i]=0.5-i*5e-4; yr[i]=2.0-i*7e-4; yi[i]=-1.0+i*3e-4; }

  double *dxr,*dxi,*dyr,*dyi,*dorr,*dori;
  cudaMalloc(&dxr,B);cudaMalloc(&dxi,B);cudaMalloc(&dyr,B);cudaMalloc(&dyi,B);
  cudaMalloc(&dorr,B);cudaMalloc(&dori,B);
  cudaMemcpy(dxr,xr,B,cudaMemcpyHostToDevice);cudaMemcpy(dxi,xi,B,cudaMemcpyHostToDevice);
  cudaMemcpy(dyr,yr,B,cudaMemcpyHostToDevice);cudaMemcpy(dyi,yi,B,cudaMemcpyHostToDevice);

  /* fixed launch + bounded per-thread arena (LAUNCH*SLAB) */
  size_t ntot = (size_t) LBLOCKS * LTHREADS;
  char *arena; size_t *atop;
  cudaMalloc (&arena, ntot * (size_t) SLAB);
  cudaMalloc (&atop,  ntot * sizeof (size_t));
  cudaMemset (atop, 0, ntot * sizeof (size_t));
  mpc_cuda_arena_base = arena; mpc_cuda_arena_slab = SLAB; mpc_cuda_arena_top = atop;
  cudaDeviceSynchronize ();

  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  axpy_kernel<<<LBLOCKS,LTHREADS>>>(N,ar,ai,dxr,dxi,dyr,dyi,dorr,dori); /* warm-up */
  cudaDeviceSynchronize();
  cudaEventRecord(e0);
  axpy_kernel<<<LBLOCKS,LTHREADS>>>(N,ar,ai,dxr,dxi,dyr,dyi,dorr,dori);
  cudaEventRecord(e1);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) { fprintf (stderr, "kernel failed: %s\n", cudaGetErrorString (err)); return 1; }
  float gpu_ms=0.f; cudaEventElapsedTime(&gpu_ms,e0,e1);
  cudaMemcpy(gr,dorr,B,cudaMemcpyDeviceToHost); cudaMemcpy(gi,dori,B,cudaMemcpyDeviceToHost);

  struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
  for (int i=0;i<N;++i) axpy_elt(ar,ai,xr[i],xi[i],yr[i],yi[i],&cr[i],&ci[i]);
  clock_gettime(CLOCK_MONOTONIC,&t1);
  double cpu_ms=(t1.tv_sec-t0.tv_sec)*1e3+(t1.tv_nsec-t0.tv_nsec)*1e-6;

  double maxrel=0.0; int mism=0;
  for (int i=0;i<N;++i){
    double dr = cr[i]!=0.0?fabs((gr[i]-cr[i])/cr[i]):fabs(gr[i]);
    double di = ci[i]!=0.0?fabs((gi[i]-ci[i])/ci[i]):fabs(gi[i]);
    double rel = dr>di?dr:di; if(rel>maxrel)maxrel=rel; if(rel>1e-13)++mism;
  }

  printf ("=== MPC complex AXPY (y = a*x + y),  N=%d, precision=%d bits/comp ===\n", N, PREC);
  printf ("launch   : %d blocks x %d threads = %zu resident; arena %zu MB\n",
          LBLOCKS, LTHREADS, ntot, (ntot * (size_t) SLAB) >> 20);
  printf ("GPU time : %8.3f ms   (per-thread arena + grid-stride)\n", gpu_ms);
  printf ("CPU time : %8.3f ms   (this library's MPC on the host)\n", cpu_ms);
  printf ("speedup  : %8.2fx (GPU vs CPU)\n", cpu_ms / gpu_ms);
  printf ("accuracy : max relative |GPU-CPU| = %.3e over %d elems (%d exceed 1e-13)\n",
          maxrel, N, mism);
  printf ("sample   : y[0]=(%.12g, %.12g)  y[N-1]=(%.12g, %.12g)\n",
          gr[0], gi[0], gr[N-1], gi[N-1]);
  free(xr);free(xi);free(yr);free(yi);free(gr);free(gi);free(cr);free(ci);
  cudaFree(dxr);cudaFree(dxi);cudaFree(dyr);cudaFree(dyi);cudaFree(dorr);cudaFree(dori);
  cudaFree(arena);cudaFree(atop);
  return mism ? 1 : 0;
}
