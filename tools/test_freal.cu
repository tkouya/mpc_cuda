/* test_freal.cu -- validate cu_freal<PB> (mul/add/sub, RNDN) bit-exact against
 * system MPFR across several precisions, including sub-limb (SB=32) cases.
 *
 *   nvcc -arch=sm_121 -O3 -fmad=false -Iinclude -I/usr/local/include \
 *        -o /tmp/test_freal tools/test_freal.cu -L/usr/local/lib -lmpfr -lgmp
 */
#include <cstdio>
#include <cstdlib>
#include "mpc_cuda/cu_freal.cuh"
using namespace cu_fp;

#include <gmp.h>
#include <mpfr.h>

static cu_limb xs (cu_limb *s){ cu_limb x=*s; x^=x<<13; x^=x>>7; x^=x<<17; return *s=x; }

template<int PB>
static cu_freal<PB> randf (cu_limb *s, int spread)
{
  cu_freal<PB> r; const int N=cu_freal<PB>::N, SB=cu_freal<PB>::SB;
  r.sign=(xs(s)&1)?1:-1;
  for (int i=0;i<N;i++) r.m[i]=xs(s);
  r.m[N-1] |= 1ULL<<63;
  if (SB) r.m[0] &= ~(((cu_limb)1<<SB)-1);    /* keep low SB bits zero */
  r.exp=(long)(xs(s)%(2*spread+1))-spread;
  return r;
}
template<int PB>
static void to_mpfr (mpfr_t out, const cu_freal<PB> &x)
{
  const int N=cu_freal<PB>::N;
  if (x.is_zero()){ mpfr_set_zero(out,1); return; }
  mpz_t M; mpz_init(M);
  mpz_import (M, N, -1, sizeof(cu_limb), 0, 0, x.m);
  mpfr_set_z (out, M, MPFR_RNDN);
  mpfr_mul_2si (out, out, x.exp - N*64, MPFR_RNDN);
  if (x.sign<0) mpfr_neg(out,out,MPFR_RNDN);
  mpz_clear(M);
}

/* device kernel computing mul/add/sub for one precision */
template<int PB>
__global__ void kern (int n, const cu_freal<PB>*A, const cu_freal<PB>*B,
                      cu_freal<PB>*MU, cu_freal<PB>*AD, cu_freal<PB>*SU)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<n){ MU[i]=cu_fmul<PB>(A[i],B[i]); AD[i]=cu_fadd<PB>(A[i],B[i]); SU[i]=cu_fsub<PB>(A[i],B[i]); }
}

template<int PB>
static int run (int M, cu_limb seed)
{
  typedef cu_freal<PB> F;
  F *A=(F*)malloc(M*sizeof(F)), *B=(F*)malloc(M*sizeof(F));
  for (int i=0;i<M;i++){ int sp=(i&3)?40:(PB+200); A[i]=randf<PB>(&seed,sp); B[i]=randf<PB>(&seed,sp); }
  F *dA,*dB,*dMU,*dAD,*dSU;
  cudaMalloc(&dA,M*sizeof(F)); cudaMalloc(&dB,M*sizeof(F));
  cudaMalloc(&dMU,M*sizeof(F)); cudaMalloc(&dAD,M*sizeof(F)); cudaMalloc(&dSU,M*sizeof(F));
  cudaMemcpy(dA,A,M*sizeof(F),cudaMemcpyHostToDevice);
  cudaMemcpy(dB,B,M*sizeof(F),cudaMemcpyHostToDevice);
  kern<PB><<<(M+127)/128,128>>>(M,dA,dB,dMU,dAD,dSU);
  cudaError_t e=cudaDeviceSynchronize();
  if (e){ printf("PB=%4d  KERNEL ERROR %s\n",PB,cudaGetErrorString(e)); return 1; }
  F *MU=(F*)malloc(M*sizeof(F)),*AD=(F*)malloc(M*sizeof(F)),*SU=(F*)malloc(M*sizeof(F));
  cudaMemcpy(MU,dMU,M*sizeof(F),cudaMemcpyDeviceToHost);
  cudaMemcpy(AD,dAD,M*sizeof(F),cudaMemcpyDeviceToHost);
  cudaMemcpy(SU,dSU,M*sizeof(F),cudaMemcpyDeviceToHost);

  mpfr_t ma,mb,ref,mine; mpfr_inits2(PB,ma,mb,ref,mine,(mpfr_ptr)0);
  long bm=0,ba=0,bs=0;
  for (int i=0;i<M;i++){
    to_mpfr<PB>(ma,A[i]); to_mpfr<PB>(mb,B[i]);
    mpfr_mul(ref,ma,mb,MPFR_RNDN); to_mpfr<PB>(mine,MU[i]); if(!mpfr_equal_p(ref,mine)) bm++;
    mpfr_add(ref,ma,mb,MPFR_RNDN); to_mpfr<PB>(mine,AD[i]); if(!mpfr_equal_p(ref,mine)) ba++;
    mpfr_sub(ref,ma,mb,MPFR_RNDN); to_mpfr<PB>(mine,SU[i]); if(!mpfr_equal_p(ref,mine)) bs++;
  }
  int SB=cu_freal<PB>::SB;
  printf("PB=%4d (N=%2d SB=%2d): mul %ld  add %ld  sub %ld  / %d   %s\n",
         PB,cu_freal<PB>::N,SB,bm,ba,bs,M,(bm||ba||bs)?"FAIL":"OK");
  mpfr_clears(ma,mb,ref,mine,(mpfr_ptr)0);
  free(A);free(B);free(MU);free(AD);free(SU);
  cudaFree(dA);cudaFree(dB);cudaFree(dMU);cudaFree(dAD);cudaFree(dSU);
  return (bm||ba||bs)?1:0;
}

int main (void)
{
  cudaDeviceSetLimit(cudaLimitStackSize,(size_t)64*1024);
  const int M=100000;
  int bad=0;
  printf("=== cu_freal<PB> vs system MPFR (RNDN), %d random pairs each ===\n",M);
  bad|=run<32>  (M,0x1111ull);
  bad|=run<64>  (M,0x2222ull);
  bad|=run<96>  (M,0x3333ull);   /* N=2 SB=32 */
  bad|=run<128> (M,0x4444ull);
  bad|=run<160> (M,0x5555ull);   /* N=3 SB=32 */
  bad|=run<256> (M,0x6666ull);
  bad|=run<288> (M,0x7777ull);   /* N=5 SB=32 */
  bad|=run<512> (M,0x8888ull);
  bad|=run<1024>(M,0x9999ull);
  bad|=run<1056>(M,0xaaaaull);   /* N=17 SB=32 */
  bad|=run<2048>(M,0xbbbbull);
  printf("%s\n", bad?"*** FAILURES ***":"ALL PRECISIONS BIT-EXACT");
  return bad;
}
