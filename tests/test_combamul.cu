/* test_combamul.cu -- validate the device register-resident comba base
 * multiply (cu_mpn_mul on device) against the portable schoolbook reference
 * (cu_mpn_mul on host), on random full-width limbs, across every operand size.
 *
 * The device path dispatches un==vn in {4..16,20,24,28,32} to the 32-bit comba;
 * the host path always runs the portable mpn_mul_1/addmul_1 loop.  Bit-identical
 * products across all sizes (including fallback sizes like 17..19) validate the
 * integration end to end.
 */
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include "mpc_cuda/cuda_minigmp.h"

typedef unsigned long long limb;

#define MAXN 34
#define NP   4096          /* operand pairs per size */

__global__ void
kern_mul (const limb *U, const limb *V, limb *R, int n, int np)
{
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < np)
    cu_mpn_mul ((mp_ptr)(R + (size_t)k * 2 * n),
                (mp_srcptr)(U + (size_t)k * n), n,
                (mp_srcptr)(V + (size_t)k * n), n);
}

static limb xs (limb *s){ limb x=*s; x^=x<<13; x^=x>>7; x^=x<<17; return *s=x; }

int
main (void)
{
  limb seed = 0xC0FFEEULL;
  limb *U=(limb*)malloc(NP*MAXN*sizeof(limb));
  limb *V=(limb*)malloc(NP*MAXN*sizeof(limb));
  limb *Rd=(limb*)malloc(NP*2*MAXN*sizeof(limb));   /* device result */
  limb *Rh=(limb*)malloc(2*MAXN*sizeof(limb));      /* host reference (per pair) */
  limb *dU,*dV,*dR;
  cudaMalloc(&dU,NP*MAXN*sizeof(limb));
  cudaMalloc(&dV,NP*MAXN*sizeof(limb));
  cudaMalloc(&dR,NP*2*MAXN*sizeof(limb));

  int total_fail = 0;
  for (int n = 2; n <= MAXN; n++)
    {
      for (size_t i=0;i<(size_t)NP*n;i++){ U[i]=xs(&seed); V[i]=xs(&seed); }
      cudaMemcpy(dU,U,(size_t)NP*n*sizeof(limb),cudaMemcpyHostToDevice);
      cudaMemcpy(dV,V,(size_t)NP*n*sizeof(limb),cudaMemcpyHostToDevice);
      int th=128, bl=(NP+th-1)/th;
      kern_mul<<<bl,th>>>(dU,dV,dR,n,NP);
      cudaError_t err=cudaDeviceSynchronize();
      if (err!=cudaSuccess){ printf("n=%d KERNEL FAIL: %s\n",n,cudaGetErrorString(err)); total_fail++; continue; }
      cudaMemcpy(Rd,dR,(size_t)NP*2*n*sizeof(limb),cudaMemcpyDeviceToHost);

      size_t mism=0;
      for (int k=0;k<NP;k++)
        {
          cu_mpn_mul((mp_ptr)Rh,(mp_srcptr)(U+(size_t)k*n),n,(mp_srcptr)(V+(size_t)k*n),n);
          for (int j=0;j<2*n;j++)
            if (Rh[j]!=Rd[(size_t)k*2*n+j]) mism++;
        }
      printf("n=%2d (%5d-bit) : %s (%zu mismatched limbs)\n",
             n, n*64, mism?"FAIL":"OK", mism);
      if (mism) total_fail++;
    }
  printf(total_fail ? "\nSOME SIZES FAILED\n" : "\nALL SIZES OK\n");
  return total_fail?1:0;
}
