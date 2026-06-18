/* test_mpc.cu -- verify CUDA-ported MPC (complex) against host (cudafied) MPC
 * and against hand-computed values.
 *
 * Each thread computes the complex  r = a*x + y  with MPC at PREC bits and
 * returns Re(r), Im(r) as doubles.  Device results are compared to the host
 * instantiation of the same __host__ __device__ MPC, and thread 0 is checked
 * against the exact complex value.
 */
#include <cstdio>
#include <cstring>
#include "mpc.h"
#include "mpc_cuda/cu_compat.h"

#define NTHREADS 256
#define PREC     200

__host__ __device__ static void
axpy_one (double ar, double ai, double xr, double xi,
          double yr, double yi, double *outr, double *outi)
{
  mpc_t a, x, y, t;
  mpc_init2 (a, PREC);
  mpc_init2 (x, PREC);
  mpc_init2 (y, PREC);
  mpc_init2 (t, PREC);

  mpc_set_d_d (a, ar, ai, MPC_RNDNN);
  mpc_set_d_d (x, xr, xi, MPC_RNDNN);
  mpc_set_d_d (y, yr, yi, MPC_RNDNN);

  mpc_mul (t, a, x, MPC_RNDNN);     /* t = a*x      */
  mpc_add (y, t, y, MPC_RNDNN);     /* y = a*x + y  */

  *outr = mpfr_get_d (mpc_realref (y), MPFR_RNDN);
  *outi = mpfr_get_d (mpc_imagref (y), MPFR_RNDN);

  mpc_clear (a); mpc_clear (x); mpc_clear (y); mpc_clear (t);
}

__global__ void
kernel (double *ar, double *ai, double *xr, double *xi,
        double *yr, double *yi, double *outr, double *outi)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < NTHREADS)
    axpy_one (ar[i], ai[i], xr[i], xi[i], yr[i], yi[i], &outr[i], &outi[i]);
}

int
main (void)
{
  cudaDeviceSetLimit (cudaLimitMallocHeapSize, (size_t) 256 * 1024 * 1024);
  /* MPC's complex mul chains deep into MPFR (mpc_mul -> mpfr_fmms -> mpfr_sub
   * -> mpfr_sub1) with large stack frames; raise the per-thread stack. */
  cudaDeviceSetLimit (cudaLimitStackSize, (size_t) 128 * 1024);

  double ar[NTHREADS], ai[NTHREADS], xr[NTHREADS], xi[NTHREADS],
         yr[NTHREADS], yi[NTHREADS];
  for (int i = 0; i < NTHREADS; ++i)
    {
      ar[i] = 1.0 + (i % 7) * 0.5;   ai[i] = 0.5 - (i % 3) * 0.25;
      xr[i] = 3.0 + (i % 11) * 0.25; xi[i] = -1.0 + (i % 4) * 0.5;
      yr[i] = -2.0 + (i % 5) * 1.5;  yi[i] = 2.0 - (i % 6) * 0.5;
    }

  double *d[8]; size_t sz = NTHREADS * sizeof (double);
  for (int k = 0; k < 8; ++k) cudaMalloc (&d[k], sz);
  cudaMemcpy (d[0], ar, sz, cudaMemcpyHostToDevice);
  cudaMemcpy (d[1], ai, sz, cudaMemcpyHostToDevice);
  cudaMemcpy (d[2], xr, sz, cudaMemcpyHostToDevice);
  cudaMemcpy (d[3], xi, sz, cudaMemcpyHostToDevice);
  cudaMemcpy (d[4], yr, sz, cudaMemcpyHostToDevice);
  cudaMemcpy (d[5], yi, sz, cudaMemcpyHostToDevice);

  int threads = 32, blocks = (NTHREADS + threads - 1) / threads;
  kernel<<<blocks, threads>>> (d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7]);
  cudaError_t err = cudaDeviceSynchronize ();
  if (err != cudaSuccess)
    { fprintf (stderr, "kernel failed: %s\n", cudaGetErrorString (err)); return 1; }

  double hr[NTHREADS], hi[NTHREADS];
  cudaMemcpy (hr, d[6], sz, cudaMemcpyDeviceToHost);
  cudaMemcpy (hi, d[7], sz, cudaMemcpyDeviceToHost);

  int fails = 0;
  for (int i = 0; i < NTHREADS; ++i)
    {
      double rr, ri;
      axpy_one (ar[i], ai[i], xr[i], xi[i], yr[i], yi[i], &rr, &ri);
      if (memcmp (&rr, &hr[i], 8) != 0 || memcmp (&ri, &hi[i], 8) != 0)
        {
          if (fails < 5)
            printf ("  MISMATCH i=%d dev=(%.17g,%.17g) host=(%.17g,%.17g)\n",
                    i, hr[i], hi[i], rr, ri);
          ++fails;
        }
    }

  printf ("MPC CUDA axpy test: %d/%d threads matched host MPC.\n",
          NTHREADS - fails, NTHREADS);
  /* thread 0 exact: a*x+y with a=(1,0.5) x=(3,-1) y=(-2,2)
     a*x = (1*3-0.5*-1, 1*-1+0.5*3) = (3.5, 0.5); +y = (1.5, 2.5) */
  double er = ar[0]*xr[0]-ai[0]*xi[0]+yr[0], ei = ar[0]*xi[0]+ai[0]*xr[0]+yi[0];
  printf ("thread 0: dev=(%.17g,%.17g) exact=(%.17g,%.17g) %s\n",
          hr[0], hi[0], er, ei,
          (hr[0]==er && hi[0]==ei) ? "OK" : "DIFF");

  for (int k = 0; k < 8; ++k) cudaFree (d[k]);
  return fails ? 1 : 0;
}
