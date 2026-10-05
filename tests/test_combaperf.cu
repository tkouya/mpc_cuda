/* test_combaperf.cu -- library-level throughput of device cu_mpn_mul, with the
 * product written to a per-thread GLOBAL scratch slab (exactly the regime of
 * the MPFR general path: cu_mpfr_tmp_allocate arena).  Build twice:
 *   with comba (default)  and  with -DCU_MPN_MUL_NO_COMBA (generic path)
 * to get an apples-to-apples library-level speedup.
 */
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include "mpc_cuda/cuda_minigmp.h"

typedef unsigned long long limb;
#define NP   (1 << 16)
#define REPS 64

__global__ void
kern (const limb *U, const limb *V, limb *acc, limb *slab, int n, int np)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x, stride = gridDim.x*blockDim.x;
  limb *s = slab + (size_t)tid * 2 * 34;
  for (int rep = 0; rep < REPS; rep++)
    for (int k = tid; k < np; k += stride)
      {
        cu_mpn_mul ((mp_ptr)s, (mp_srcptr)(U+(size_t)k*n), n,
                    (mp_srcptr)(V+(size_t)k*n), n);
        limb x = 0; for (int i = 0; i < 2*n; i++) x ^= s[i]; acc[k] ^= x;
      }
}

static limb xs (limb *s){ limb x=*s; x^=x<<13; x^=x>>7; x^=x<<17; return *s=x; }

int main (void)
{
  limb seed = 0x77;
  limb *U=(limb*)malloc((size_t)NP*34*sizeof(limb));
  limb *V=(limb*)malloc((size_t)NP*34*sizeof(limb));
  for (size_t i=0;i<(size_t)NP*34;i++){ U[i]=xs(&seed); V[i]=xs(&seed); }
  limb *dU,*dV,*dAcc,*dSlab;
  cudaMalloc(&dU,(size_t)NP*34*sizeof(limb)); cudaMalloc(&dV,(size_t)NP*34*sizeof(limb));
  cudaMalloc(&dAcc,NP*sizeof(limb)); cudaMemset(dAcc,0,NP*sizeof(limb));
  cudaMemcpy(dU,U,(size_t)NP*34*sizeof(limb),cudaMemcpyHostToDevice);
  cudaMemcpy(dV,V,(size_t)NP*34*sizeof(limb),cudaMemcpyHostToDevice);
  int block=128, grid=0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&grid,kern,block,0);
  int sm=0; cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,0); grid*=sm;
  cudaMalloc(&dSlab,(size_t)grid*block*2*34*sizeof(limb));

  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  int sizes[] = {4,6,8,12,16,24,32};
  printf("grid=%d block=%d  (%.1f M mul/s figures)\n", grid, block, (double)NP*REPS/1e6);
  for (unsigned si=0; si<sizeof(sizes)/sizeof(int); si++)
    {
      int n = sizes[si];
      kern<<<grid,block>>>(dU,dV,dAcc,dSlab,n,NP); cudaDeviceSynchronize();
      cudaEventRecord(a); kern<<<grid,block>>>(dU,dV,dAcc,dSlab,n,NP);
      cudaEventRecord(b); cudaEventSynchronize(b);
      float t=0; cudaEventElapsedTime(&t,a,b);
      printf("n=%2d (%5d-bit): %8.3f ms   %8.1f M mul/s\n",
             n, n*64, t, (double)NP*REPS/1e3/t);
    }
  return 0;
}
