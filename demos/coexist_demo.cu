/* coexist_demo.cu -- the GPU half of the coexistence demo, and the program's
 * main().
 *
 * This translation unit uses THIS library's CUDA MPFR (the cu_mpfr_* symbols)
 * to compute pi and log(2) inside a kernel, then calls system_mpfr_ref() (in
 * coexist_cpu.cpp), which computes the same constants with the *system* libmpfr
 * on the CPU.  Both halves are linked into ONE binary -- proof that the
 * cu_-prefixed library coexists with the system GMP/MPFR/MPC with no symbol
 * collision.
 *
 * Build (see `make coexist`):
 *   g++   -c demos/coexist_cpu.cpp -o build/coexist_cpu.o
 *   EXTRA_LINK="build/coexist_cpu.o -lmpfr -lgmp" \
 *     bash tools/build_cuda_test.sh demos/coexist_demo.cu build/coexist_demo
 */
#include <cstdio>
typedef long int gmp_randstate_t[1];
#include "mpc_cuda.cuh"

/* from coexist_cpu.cpp, backed by the SYSTEM libmpfr */
extern "C" void        system_mpfr_ref (int prec, double *pi, double *log2);
extern "C" const char *system_mpfr_version (void);

__global__ void
gpu_kernel (double *o)
{
  cu_mpfr_t r, x;                     /* -> cu_mpfr_t via our header */
  cu_mpfr_init2 (r, 200);
  cu_mpfr_init2 (x, 200);
  cu_mpfr_const_pi (r, CU_MPFR_RNDN);          o[0] = cu_mpfr_get_d (r, CU_MPFR_RNDN);
  cu_mpfr_set_ui  (x, 2, CU_MPFR_RNDN);
  cu_mpfr_log     (r, x, CU_MPFR_RNDN);        o[1] = cu_mpfr_get_d (r, CU_MPFR_RNDN);
  cu_mpfr_clear (r);
  cu_mpfr_clear (x);
}

int
main (void)
{
  cudaDeviceSetLimit (cudaLimitMallocHeapSize, (size_t) 256 * 1024 * 1024);
  cudaDeviceSetLimit (cudaLimitStackSize,      (size_t) 192 * 1024);

  double *d; cudaMalloc (&d, 2 * sizeof (double));
  gpu_kernel<<<1, 1>>> (d);
  cudaError_t e = cudaDeviceSynchronize ();
  if (e != cudaSuccess) { fprintf (stderr, "kernel failed: %s\n", cudaGetErrorString (e)); return 1; }
  double g[2]; cudaMemcpy (g, d, 2 * sizeof (double), cudaMemcpyDeviceToHost);

  double c[2];
  system_mpfr_ref (200, &c[0], &c[1]);   /* system libmpfr on the CPU */

  printf ("=== Coexistence in ONE binary ===\n");
  printf ("GPU side : this library's cu_mpfr (CUDA kernel)\n");
  printf ("CPU side : system libmpfr %s (unprefixed cu_mpfr_*)\n\n", system_mpfr_version ());
  printf ("  %-10s %-22s %-22s\n", "", "GPU cu_mpfr", "CPU system mpfr");
  printf ("  %-10s %-22.15f %-22.15f  %s\n", "pi",     g[0], c[0], g[0] == c[0] ? "MATCH" : "differ");
  printf ("  %-10s %-22.15f %-22.15f  %s\n", "log(2)", g[1], c[1], g[1] == c[1] ? "MATCH" : "differ");

  cudaFree (d);
  return (g[0] == c[0] && g[1] == c[1]) ? 0 : 1;
}
