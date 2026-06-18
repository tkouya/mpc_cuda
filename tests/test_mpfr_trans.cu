/* test_mpfr_trans.cu -- verify CUDA-ported MPFR transcendental functions
 * against the host (cudafied) MPFR and known double values.
 *
 * Covers exp / log / sin / cos / atan / sqrt and the AGM constant pi.  These
 * depend on a correctly-rounded mpfr_div, whose generic path miscompiles on the
 * device; this library substitutes an mpz-based device mpfr_div (see
 * docs/phase2-mpfr.md), after which the whole transcendental set works.
 */
#include <cstdio>
#include <cstring>
#include <cmath>
typedef long int gmp_randstate_t[1];
#include "mpfr.h"
#include "mpc_cuda/cu_compat.h"

#define NTHREADS 128
#define PREC     200
#define NOUT     7

__host__ __device__ static void
compute (double seed, double *out)
{
  mpfr_t x, r, pi;
  mpfr_init2 (x, PREC);
  mpfr_init2 (r, PREC);
  mpfr_init2 (pi, PREC);

  mpfr_const_pi (pi, MPFR_RNDN);          out[0] = mpfr_get_d (pi, MPFR_RNDN);
  mpfr_set_d (x, 1.0 + seed, MPFR_RNDN);
  mpfr_exp  (r, x, MPFR_RNDN);            out[1] = mpfr_get_d (r, MPFR_RNDN);
  mpfr_set_d (x, 2.0 + seed, MPFR_RNDN);
  mpfr_log  (r, x, MPFR_RNDN);            out[2] = mpfr_get_d (r, MPFR_RNDN);
  mpfr_set_d (x, 0.5 + seed, MPFR_RNDN);
  mpfr_sin  (r, x, MPFR_RNDN);            out[3] = mpfr_get_d (r, MPFR_RNDN);
  mpfr_set_d (x, 0.5 + seed, MPFR_RNDN);
  mpfr_cos  (r, x, MPFR_RNDN);            out[4] = mpfr_get_d (r, MPFR_RNDN);
  mpfr_set_d (x, 0.5 + seed, MPFR_RNDN);
  mpfr_atan (r, x, MPFR_RNDN);            out[5] = mpfr_get_d (r, MPFR_RNDN);
  mpfr_set_d (x, 2.0 + seed, MPFR_RNDN);
  mpfr_sqrt (r, x, MPFR_RNDN);            out[6] = mpfr_get_d (r, MPFR_RNDN);

  mpfr_clear (x); mpfr_clear (r); mpfr_clear (pi);
}

__global__ void kernel (double *out)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < NTHREADS) compute (i * 0.01, out + i * NOUT);
}

int main (void)
{
  cudaDeviceSetLimit (cudaLimitMallocHeapSize, (size_t) 512 * 1024 * 1024);
  cudaDeviceSetLimit (cudaLimitStackSize,      (size_t) 192 * 1024);

  double *d; size_t sz = NTHREADS * NOUT * sizeof (double);
  cudaMalloc (&d, sz);
  kernel<<<(NTHREADS + 31) / 32, 32>>> (d);
  cudaError_t err = cudaDeviceSynchronize ();
  if (err != cudaSuccess) { fprintf (stderr, "kernel failed: %s\n", cudaGetErrorString (err)); return 1; }

  double *h = (double *) malloc (sz);
  cudaMemcpy (h, d, sz, cudaMemcpyDeviceToHost);

  int fails = 0;
  for (int i = 0; i < NTHREADS; ++i)
    {
      double ref[NOUT]; compute (i * 0.01, ref);
      for (int k = 0; k < NOUT; ++k)
        {
          double rel = ref[k] != 0.0 ? fabs ((h[i*NOUT+k] - ref[k]) / ref[k]) : fabs (h[i*NOUT+k]);
          if (rel > 1e-13)
            { if (fails < 5) printf ("  MISMATCH i=%d k=%d dev=%.17g host=%.17g\n", i, k, h[i*NOUT+k], ref[k]); ++fails; }
        }
    }
  printf ("MPFR transcendental CUDA test (pi/exp/log/sin/cos/atan/sqrt): "
          "%d/%d threads matched host MPFR (rel<1e-13).\n", NTHREADS - fails, NTHREADS);
  printf ("thread 0: pi=%.15f exp(1)=%.15f log(2)=%.15f sin(.5)=%.15f cos(.5)=%.15f atan(.5)=%.15f\n",
          h[0], h[1], h[2], h[3], h[4], h[5]);
  printf ("known   : pi=%.15f exp(1)=%.15f log(2)=%.15f sin(.5)=%.15f cos(.5)=%.15f atan(.5)=%.15f\n",
          M_PI, M_E, log(2.0), sin(0.5), cos(0.5), atan(0.5));
  free (h); cudaFree (d);
  return fails ? 1 : 0;
}
