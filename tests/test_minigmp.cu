/* test_minigmp.cu -- verify CUDA-ported mini-gmp (mpz) against host mini-gmp.
 *
 * Each device thread independently computes  c = a*b + (a-b)  and  q = c / a
 * for thread-dependent operands, formats the results as decimal strings, and
 * writes them out.  The host computes the same with the (host-compiled) ported
 * mini-gmp and the results are compared string-by-string.
 */
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include "mpc_cuda/cuda_minigmp.h"
#include "mpc_cuda/cu_compat.h"

#define NTHREADS 256
#define STRLEN   256

__host__ __device__ static void
compute (unsigned long seed, char *c_out, char *q_out)
{
  mpz_t a, b, c, t, q, r;
  mpz_init (a); mpz_init (b); mpz_init (c);
  mpz_init (t); mpz_init (q); mpz_init (r);

  /* a = (seed+3)^7 ,  b = (seed+2)^5   -> multi-limb operands */
  mpz_set_ui (a, seed + 3);
  mpz_pow_ui (a, a, 7);
  mpz_set_ui (b, seed + 2);
  mpz_pow_ui (b, b, 5);

  mpz_mul (c, a, b);          /* c = a*b            */
  mpz_sub (t, a, b);          /* t = a-b            */
  mpz_add (c, c, t);          /* c = a*b + (a-b)    */

  mpz_tdiv_qr (q, r, c, a);   /* q = c / a (trunc)  */

  mpz_get_str (c_out, 10, c);
  mpz_get_str (q_out, 10, q);

  mpz_clear (a); mpz_clear (b); mpz_clear (c);
  mpz_clear (t); mpz_clear (q); mpz_clear (r);
}

__global__ void
kernel (char *c_all, char *q_all)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < NTHREADS)
    compute ((unsigned long) idx, c_all + idx * STRLEN, q_all + idx * STRLEN);
}

int
main (void)
{
  /* mini-gmp uses the device heap (malloc/free) for limb storage. */
  cudaDeviceSetLimit (cudaLimitMallocHeapSize, (size_t) 64 * 1024 * 1024);

  size_t bytes = (size_t) NTHREADS * STRLEN;
  char *d_c, *d_q;
  cudaMalloc (&d_c, bytes);
  cudaMalloc (&d_q, bytes);

  int threads = 64;
  int blocks  = (NTHREADS + threads - 1) / threads;
  kernel<<<blocks, threads>>> (d_c, d_q);
  cudaError_t err = cudaDeviceSynchronize ();
  if (err != cudaSuccess)
    {
      fprintf (stderr, "kernel failed: %s\n", cudaGetErrorString (err));
      return 1;
    }

  char *h_c = (char *) malloc (bytes);
  char *h_q = (char *) malloc (bytes);
  cudaMemcpy (h_c, d_c, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy (h_q, d_q, bytes, cudaMemcpyDeviceToHost);

  int fails = 0;
  for (int i = 0; i < NTHREADS; ++i)
    {
      char ref_c[STRLEN], ref_q[STRLEN];
      compute ((unsigned long) i, ref_c, ref_q);
      if (strcmp (ref_c, h_c + i * STRLEN) != 0 ||
          strcmp (ref_q, h_q + i * STRLEN) != 0)
        {
          if (fails < 5)
            printf ("  MISMATCH i=%d\n    host c=%s\n    dev  c=%s\n"
                    "    host q=%s\n    dev  q=%s\n",
                    i, ref_c, h_c + i * STRLEN, ref_q, h_q + i * STRLEN);
          ++fails;
        }
    }

  printf ("mini-gmp CUDA test: %d/%d threads matched host mini-gmp.\n",
          NTHREADS - fails, NTHREADS);
  if (fails == 0)
    {
      printf ("sample (thread 7): c = %s\n", h_c + 7 * STRLEN);
      printf ("                   q = %s\n", h_q + 7 * STRLEN);
    }

  free (h_c); free (h_q);
  cudaFree (d_c); cudaFree (d_q);
  return fails ? 1 : 0;
}
