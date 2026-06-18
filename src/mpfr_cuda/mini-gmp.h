/* shim -> CUDA-ported mini-gmp + CUDA helpers */
#include "mpc_cuda/cuda_minigmp.h"
/* MPFR_RODATA: __device__ in the device pass, nothing on the
   host pass, so one definition of a read-only table serves
   both address spaces. */
#ifndef MPFR_RODATA
# ifdef __CUDA_ARCH__
#  define MPFR_RODATA __device__
# else
#  define MPFR_RODATA
# endif
#endif
