/* mpc_cuda.h -- convenience alias for the umbrella header.
 *
 * The mpc_cuda API is CUDA device/host code (functions are __host__ __device__),
 * so it must be compiled with nvcc.  This .h simply forwards to mpc_cuda.cuh so
 * that either spelling works:
 *
 *     #include "mpc_cuda.h"      // or
 *     #include "mpc_cuda.cuh"
 */
#ifndef MPC_CUDA_UMBRELLA_H
#define MPC_CUDA_UMBRELLA_H
#include "mpc_cuda.cuh"
#endif /* MPC_CUDA_UMBRELLA_H */
