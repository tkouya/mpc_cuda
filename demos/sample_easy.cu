/* sample_easy.cu -- the "just include one header" demo for mpc_cuda.
 *
 * This single .cu file uses BOTH stacks at once, with no name clashes:
 *
 *     - the system CPU GMP/MPFR/MPC   via  <mpfr.h> / <mpc.h>   (plain names)
 *     - this GPU library              via  "mpc_cuda.cuh"       (cu_ names)
 *
 * The GPU computes pi (real, MPFR) and a complex product a*x+y (MPC) inside a
 * kernel; the CPU recomputes the same with the system libraries; we check that
 * the results agree to the last bit.
 *
 * Build (installed tree -- only the install include root + the system headers):
 *     nvcc -arch=sm_121 -rdc=true -fmad=false sample_easy.cu \
 *          -I<prefix>/include -lmpc_cuda_helper... -lmpc -lmpfr -lgmp -o sample_easy
 * In this source tree just run:   make sample
 */

/* ---- system CPU multiple precision (plain names) ---- */
#include <mpfr.h>
#include <mpc.h>

/* ---- the GPU library: ONE include, every cu_ function/type/constant ---- */
#include "mpc_cuda.cuh"

#include <cstdio>
#include <cmath>

#ifndef PREC
#define PREC 256          /* working precision in bits */
#endif

/* ----------------------------------------------------------------------- *
 * GPU kernel: everything here is cu_-prefixed and runs on the device.     *
 * ----------------------------------------------------------------------- */
__global__ void gpu_kernel(double *pi_out,
                           double *re_out, double *im_out)
{
    if (threadIdx.x || blockIdx.x) return;   /* one thread is enough here */

    /* --- real: pi via MPFR --- */
    cu_mpfr_t pi;
    cu_mpfr_init2(pi, PREC);
    cu_mpfr_const_pi(pi, CU_MPFR_RNDN);
    *pi_out = cu_mpfr_get_d(pi, CU_MPFR_RNDN);
    cu_mpfr_clear(pi);

    /* --- complex: y = a*x + y via MPC --- */
    cu_mpc_t a, x, y;
    cu_mpc_init2(a, PREC);
    cu_mpc_init2(x, PREC);
    cu_mpc_init2(y, PREC);
    cu_mpc_set_d_d(a, 1.5, -0.25, CU_MPC_RNDNN);   /* a = 1.5 - 0.25 i */
    cu_mpc_set_d_d(x, 2.0,  3.0,  CU_MPC_RNDNN);   /* x = 2   + 3    i */
    cu_mpc_set_d_d(y, 0.5,  0.5,  CU_MPC_RNDNN);   /* y = 0.5 + 0.5  i */
    cu_mpc_fma(y, a, x, y, CU_MPC_RNDNN);          /* y = a*x + y */
    *re_out = cu_mpfr_get_d(cu_mpc_realref(y), CU_MPFR_RNDN);
    *im_out = cu_mpfr_get_d(cu_mpc_imagref(y), CU_MPFR_RNDN);
    cu_mpc_clear(a); cu_mpc_clear(x); cu_mpc_clear(y);
}

/* ----------------------------------------------------------------------- *
 * CPU reference using the SYSTEM libraries (plain names, no cu_ prefix).   *
 * ----------------------------------------------------------------------- */
static void cpu_reference(double *pi_out, double *re_out, double *im_out)
{
    mpfr_t pi;
    mpfr_init2(pi, PREC);
    mpfr_const_pi(pi, MPFR_RNDN);
    *pi_out = mpfr_get_d(pi, MPFR_RNDN);
    mpfr_clear(pi);

    mpc_t a, x, y;
    mpc_init2(a, PREC);
    mpc_init2(x, PREC);
    mpc_init2(y, PREC);
    mpc_set_d_d(a, 1.5, -0.25, MPC_RNDNN);
    mpc_set_d_d(x, 2.0,  3.0,  MPC_RNDNN);
    mpc_set_d_d(y, 0.5,  0.5,  MPC_RNDNN);
    mpc_fma(y, a, x, y, MPC_RNDNN);
    *re_out = mpfr_get_d(mpc_realref(y), MPFR_RNDN);
    *im_out = mpfr_get_d(mpc_imagref(y), MPFR_RNDN);
    mpc_clear(a); mpc_clear(x); mpc_clear(y);
}

int main(void)
{
    /* The deep MPC/MPFR call chains need a bigger thread stack than the CUDA
     * default; the limb allocator also needs device heap.  (No bump arena is
     * installed here, so cu_mpfr_init2 falls back to device malloc.) */
    cudaDeviceSetLimit(cudaLimitStackSize,      128 * 1024);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 64 * 1024 * 1024);

    double *d;  cudaMallocManaged(&d, 3 * sizeof(double));
    gpu_kernel<<<1, 1>>>(&d[0], &d[1], &d[2]);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(e));
        return 1;
    }

    double cpi, cre, cim;
    cpu_reference(&cpi, &cre, &cim);

    printf("            %-22s %-22s\n", "GPU (cu_mpc_cuda)", "CPU (system MPFR/MPC)");
    printf("pi        = %-22.15f %-22.15f\n", d[0], cpi);
    printf("Re(a*x+y) = %-22.15f %-22.15f\n", d[1], cre);
    printf("Im(a*x+y) = %-22.15f %-22.15f\n", d[2], cim);

    int ok = (d[0] == cpi) && (d[1] == cre) && (d[2] == cim);
    printf("\n%s (GPU and CPU agree bit-for-bit)\n", ok ? "PASS" : "FAIL");

    cudaFree(d);
    return ok ? 0 : 1;
}
