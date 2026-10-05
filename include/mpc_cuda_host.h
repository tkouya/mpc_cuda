/* mpc_cuda_host.h -- single include for the HOST (CPU) part of mpc_cuda: the
 * header-only fixed-precision types, usable with any C++17 compiler and
 * without CUDA (no nvcc, no libmpc_cuda, no libcudart):
 *
 *   cu_fp::cu_freal<PB>, cu_fp::cu_fcomplex<PB>      + - *, bit-exact with
 *                                                    MPFR / MPC round-to-nearest
 *   cu_fp::cu_ffma_cr / cu_fdot / cu_cfma_cr / cu_cdot   correctly rounded fused
 *                                                    ops (= mpfr_fma/mpfr_dot/
 *                                                    mpc_fma/mpc_dot)
 *   cu_fp::cu_exp, cu_log, cu_sin, ..., cu_cexp, ...  elementary functions
 *                                                    (<= 1 ULP vs MPFR/MPC)
 *
 *   #include "mpc_cuda_host.h"
 *   cu_fp::cu_freal<256> a = 1.5, b = 2.25;
 *   double d = (double)(a*b + a);
 *
 * Compile with  g++ -std=c++17 -O2 -march=native -Iinclude ...   (no libraries
 * needed; -lgmp only with -DCU_FP_HOST_USE_GMP).  The same headers also compile
 * with nvcc for the GPU (see mpc_cuda.cuh).
 */
#ifndef MPC_CUDA_HOST_H
#define MPC_CUDA_HOST_H

#include "mpc_cuda/cu_freal.cuh"
#include "mpc_cuda/cu_fcomplex.cuh"
#include "mpc_cuda/cu_fmath.cuh"
#include "mpc_cuda/cu_fcmath.cuh"
#include "mpc_cuda/cu_ffused.cuh"

#endif /* MPC_CUDA_HOST_H */
