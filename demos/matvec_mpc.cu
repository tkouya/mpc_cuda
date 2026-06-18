/* matvec_mpc.cu -- multiple-precision COMPLEX matrix-vector product y = A*x
 * with MPC, run on the GPU and on the CPU, comparing time and accuracy.
 *
 * Complex analogue of matvec_mpfr.cu.  The complex accumulator/scratch are
 * stack-backed cu_mpc_t (CU_MPC_DECL_INIT, built from cu_mpfr_custom_init_set on the two
 * components), so the per-thread arena only holds one complex multiply-add's
 * scratch and is reset every inner iteration.  Build: `make matvec-mpc`.
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
typedef long int gmp_randstate_t[1];
#include "mpc_cuda.cuh"

#ifndef N
#define N 128
#endif
#ifndef PREC
#define PREC 1024
#endif
#ifndef SLAB
#define SLAB (64 * 1024)             /* one complex madd's scratch */
#endif
#ifndef LBLOCKS
#define LBLOCKS 128
#endif
#ifndef LTHREADS
#define LTHREADS 32
#endif

#define NLIMBS ((PREC + 8 * (int) sizeof (cu_mp_limb_t) - 1) / (8 * (int) sizeof (cu_mp_limb_t)))
/* stack-backed cu_mpc_t initialised to +0 */
#define CU_MPC_DECL_INIT(z)                                                 \
  cu_mpc_t z;                                                              \
  cu_mp_limb_t z##_rl[NLIMBS], z##_il[NLIMBS];                             \
  cu_mpfr_custom_init_set (cu_mpc_realref (z), CU_MPFR_ZERO_KIND, 0, PREC, z##_rl); \
  cu_mpfr_custom_init_set (cu_mpc_imagref (z), CU_MPFR_ZERO_KIND, 0, PREC, z##_il)

/* (yr,yi) = sum_j A[i,j] * x[j]  (complex) */
__host__ __device__ static void
cdot_row (const double *Ar, const double *Ai,
          const double *xr, const double *xi, int n,
          double *yr, double *yi)
{
  CU_MPC_DECL_INIT (acc); CU_MPC_DECL_INIT (a); CU_MPC_DECL_INIT (xx); CU_MPC_DECL_INIT (t);
  for (int j = 0; j < n; ++j)
    {
#ifdef __CUDA_ARCH__
      mpc_cuda_arena_reset ();
#endif
      cu_mpc_set_d_d (a,  Ar[j], Ai[j], CU_MPC_RNDNN);
      cu_mpc_set_d_d (xx, xr[j], xi[j], CU_MPC_RNDNN);
      cu_mpc_mul (t, a, xx, CU_MPC_RNDNN);
      cu_mpc_add (acc, acc, t, CU_MPC_RNDNN);
    }
  *yr = cu_mpfr_get_d (cu_mpc_realref (acc), CU_MPFR_RNDN);
  *yi = cu_mpfr_get_d (cu_mpc_imagref (acc), CU_MPFR_RNDN);
}

__global__ void
matvec_kernel (int n, const double *Ar, const double *Ai,
               const double *xr, const double *xi, double *yr, double *yi)
{
  int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    cdot_row (Ar + (size_t) i * n, Ai + (size_t) i * n, xr, xi, n, &yr[i], &yi[i]);
}

int
main (void)
{
  cudaDeviceSetLimit (cudaLimitMallocHeapSize, (size_t) 64 * 1024 * 1024);
  cudaDeviceSetLimit (cudaLimitStackSize,      (size_t) 160 * 1024);

  size_t MA = (size_t) N * N * sizeof (double), MV = (size_t) N * sizeof (double);
  double *Ar = (double *) malloc (MA), *Ai = (double *) malloc (MA);
  double *xr = (double *) malloc (MV), *xi = (double *) malloc (MV);
  double *ygr = (double *) malloc (MV), *ygi = (double *) malloc (MV);
  double *ycr = (double *) malloc (MV), *yci = (double *) malloc (MV);
  for (int i = 0; i < N; ++i)
    {
      xr[i] = 1.0 + i * 1e-3; xi[i] = 0.5 - i * 5e-4;
      for (int j = 0; j < N; ++j)
        { Ar[(size_t) i * N + j] = 1.0 + (i + 2 * j) * 1e-4;
          Ai[(size_t) i * N + j] = -0.5 + (2 * i + j) * 1e-4; }
    }

  double *dAr, *dAi, *dxr, *dxi, *dyr, *dyi;
  cudaMalloc (&dAr, MA); cudaMalloc (&dAi, MA);
  cudaMalloc (&dxr, MV); cudaMalloc (&dxi, MV);
  cudaMalloc (&dyr, MV); cudaMalloc (&dyi, MV);
  cudaMemcpy (dAr, Ar, MA, cudaMemcpyHostToDevice); cudaMemcpy (dAi, Ai, MA, cudaMemcpyHostToDevice);
  cudaMemcpy (dxr, xr, MV, cudaMemcpyHostToDevice); cudaMemcpy (dxi, xi, MV, cudaMemcpyHostToDevice);

  size_t ntot = (size_t) LBLOCKS * LTHREADS;
  char *arena; size_t *top;
  cudaMalloc (&arena, ntot * (size_t) SLAB);
  cudaMalloc (&top,   ntot * sizeof (size_t));
  cudaMemset (top, 0, ntot * sizeof (size_t));
  mpc_cuda_arena_base = arena; mpc_cuda_arena_slab = SLAB; mpc_cuda_arena_top = top;
  cudaDeviceSynchronize ();

  cudaEvent_t e0, e1; cudaEventCreate (&e0); cudaEventCreate (&e1);
  matvec_kernel<<<LBLOCKS, LTHREADS>>> (N, dAr, dAi, dxr, dxi, dyr, dyi);  /* warm-up */
  cudaDeviceSynchronize ();
  cudaEventRecord (e0);
  matvec_kernel<<<LBLOCKS, LTHREADS>>> (N, dAr, dAi, dxr, dxi, dyr, dyi);
  cudaEventRecord (e1);
  cudaError_t err = cudaDeviceSynchronize ();
  if (err != cudaSuccess) { fprintf (stderr, "kernel failed: %s\n", cudaGetErrorString (err)); return 1; }
  float gpu_ms = 0.f; cudaEventElapsedTime (&gpu_ms, e0, e1);
  cudaMemcpy (ygr, dyr, MV, cudaMemcpyDeviceToHost); cudaMemcpy (ygi, dyi, MV, cudaMemcpyDeviceToHost);

  struct timespec t0, t1;
  clock_gettime (CLOCK_MONOTONIC, &t0);
  for (int i = 0; i < N; ++i)
    cdot_row (Ar + (size_t) i * N, Ai + (size_t) i * N, xr, xi, N, &ycr[i], &yci[i]);
  clock_gettime (CLOCK_MONOTONIC, &t1);
  double cpu_ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) * 1e-6;

  double maxrel = 0.0; int mism = 0;
  for (int i = 0; i < N; ++i)
    {
      double dr = ycr[i] != 0.0 ? fabs ((ygr[i] - ycr[i]) / ycr[i]) : fabs (ygr[i]);
      double di = yci[i] != 0.0 ? fabs ((ygi[i] - yci[i]) / yci[i]) : fabs (ygi[i]);
      double rel = dr > di ? dr : di; if (rel > maxrel) maxrel = rel; if (rel > 1e-13) ++mism;
    }

  printf ("=== MPC complex matrix-vector  y = A*x,  N=%d, precision=%d bits/comp ===\n", N, PREC);
  printf ("GPU time : %8.3f ms   (per-thread arena + grid-stride)\n", gpu_ms);
  printf ("CPU time : %8.3f ms   (this library's MPC on the host)\n", cpu_ms);
  printf ("speedup  : %8.2fx (GPU vs CPU)\n", cpu_ms / gpu_ms);
  printf ("accuracy : max relative |GPU-CPU| = %.3e over %d elems (%d exceed 1e-13)\n",
          maxrel, N, mism);
  printf ("sample   : y[0]=(%.12g, %.12g)  y[N-1]=(%.12g, %.12g)\n",
          ygr[0], ygi[0], ygr[N - 1], ygi[N - 1]);

  return mism ? 1 : 0;
}
