/* test_mpfr.cu -- verify CUDA-ported MPFR core against the host (cudafied)
 * path and against hand-computed values.
 *
 * Each thread computes  r = a*x + y  with MPFR at PREC bits, then converts the
 * result to double.  The device results are compared to the same computation
 * run on the host (the host instantiation of the __host__ __device__ MPFR), and
 * thread 0 is checked against the exact value a0*x0+y0.
 */
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
/* MPFR built against mini-gmp needs gmp_randstate_t defined before mpfr.h
 * (mini-gmp has no random functions). See mpfr doc/mini-gmp. */
typedef long int gmp_randstate_t[1];
#include "mpfr.h"
#include "mpc_cuda/cu_compat.h"

#define NTHREADS 256
#define PREC     200

__host__ __device__ static double
axpy_one (double a, double x, double y)
{
  mpfr_t ma, mx, my, t;
  mpfr_init2 (ma, PREC);
  mpfr_init2 (mx, PREC);
  mpfr_init2 (my, PREC);
  mpfr_init2 (t,  PREC);

  mpfr_set_d (ma, a, MPFR_RNDN);
  mpfr_set_d (mx, x, MPFR_RNDN);
  mpfr_set_d (my, y, MPFR_RNDN);

  mpfr_mul (t, ma, mx, MPFR_RNDN);     /* t = a*x       */
  mpfr_add (my, t, my, MPFR_RNDN);     /* y = a*x + y   */

  double r = mpfr_get_d (my, MPFR_RNDN);

  mpfr_clear (ma); mpfr_clear (mx); mpfr_clear (my); mpfr_clear (t);
  return r;
}

__global__ void
kernel (double *a, double *x, double *y, double *out)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < NTHREADS)
    out[i] = axpy_one (a[i], x[i], y[i]);
}

int
main (void)
{
  cudaDeviceSetLimit (cudaLimitMallocHeapSize, (size_t) 256 * 1024 * 1024);

  double a[NTHREADS], x[NTHREADS], y[NTHREADS];
  for (int i = 0; i < NTHREADS; ++i)
    {
      a[i] = 1.0 + (i % 7) * 0.5;
      x[i] = 3.0 + (i % 11) * 0.25;
      y[i] = -2.0 + (i % 5) * 1.5;
    }

  double *da, *dx, *dy, *dout;
  size_t sz = NTHREADS * sizeof (double);
  cudaMalloc (&da, sz); cudaMalloc (&dx, sz);
  cudaMalloc (&dy, sz); cudaMalloc (&dout, sz);
  cudaMemcpy (da, a, sz, cudaMemcpyHostToDevice);
  cudaMemcpy (dx, x, sz, cudaMemcpyHostToDevice);
  cudaMemcpy (dy, y, sz, cudaMemcpyHostToDevice);

  int threads = 32, blocks = (NTHREADS + threads - 1) / threads;
  kernel<<<blocks, threads>>> (da, dx, dy, dout);
  cudaError_t err = cudaDeviceSynchronize ();
  if (err != cudaSuccess)
    {
      fprintf (stderr, "kernel failed: %s\n", cudaGetErrorString (err));
      return 1;
    }

  double hout[NTHREADS];
  cudaMemcpy (hout, dout, sz, cudaMemcpyDeviceToHost);

  int fails = 0;
  for (int i = 0; i < NTHREADS; ++i)
    {
      double ref = axpy_one (a[i], x[i], y[i]);   /* host (cudafied) MPFR */
      if (memcmp (&ref, &hout[i], sizeof (double)) != 0)
        {
          if (fails < 5)
            printf ("  MISMATCH i=%d  dev=%.17g  host=%.17g\n",
                    i, hout[i], ref);
          ++fails;
        }
    }

  printf ("MPFR CUDA axpy test: %d/%d threads matched host MPFR.\n",
          NTHREADS - fails, NTHREADS);
  double exact0 = a[0] * x[0] + y[0];
  printf ("thread 0: a=%.1f x=%.1f y=%.1f -> dev=%.17g  exact=%.17g  %s\n",
          a[0], x[0], y[0], hout[0], exact0,
          (hout[0] == exact0) ? "OK" : "DIFF");

  cudaFree (da); cudaFree (dx); cudaFree (dy); cudaFree (dout);
  return fails ? 1 : 0;
}
