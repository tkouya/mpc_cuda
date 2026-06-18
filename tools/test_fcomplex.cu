/* test_fcomplex.cu -- validate cu_fcomplex<PB> (mul/add/sub) bit-exact against
 * system MPC (MPC_RNDNN) across several precisions, incl sub-limb (SB=32).
 *
 *   nvcc -arch=sm_121 -O3 -fmad=false -Iinclude -I/usr/local/include \
 *        -o /tmp/test_fcomplex tools/test_fcomplex.cu -L/usr/local/lib -lmpc -lmpfr -lgmp
 */
#include <cstdio>
#include <cstdlib>
#include "mpc_cuda/cu_fcomplex.cuh"
using namespace cu_fp;
#include <gmp.h>
#include <mpfr.h>
#include <mpc.h>

static cu_limb xs (cu_limb *s){ cu_limb x=*s; x^=x<<13; x^=x>>7; x^=x<<17; return *s=x; }

template<int PB>
static cu_freal<PB> randf (cu_limb *s, int spread)
{
  cu_freal<PB> r; const int N=cu_freal<PB>::N, SB=cu_freal<PB>::SB;
  r.sign=(xs(s)&1)?1:-1;
  for (int i=0;i<N;i++) r.m[i]=xs(s);
  r.m[N-1]|=1ULL<<63; if(SB) r.m[0]&=~(((cu_limb)1<<SB)-1);
  r.exp=(long)(xs(s)%(2*spread+1))-spread;
  return r;
}
template<int PB>
static void f_to_mpfr (mpfr_t out, const cu_freal<PB> &x)
{
  const int N=cu_freal<PB>::N;
  if (x.is_zero()){ mpfr_set_zero(out,1); return; }
  mpz_t M; mpz_init(M);
  mpz_import(M,N,-1,sizeof(cu_limb),0,0,x.m);
  mpfr_set_z(out,M,MPFR_RNDN); mpfr_mul_2si(out,out,x.exp-N*64,MPFR_RNDN);
  if(x.sign<0) mpfr_neg(out,out,MPFR_RNDN);
  mpz_clear(M);
}
template<int PB>
static void c_to_mpc (mpc_t out, const cu_fcomplex<PB> &z)
{ f_to_mpfr<PB>(mpc_realref(out),z.re); f_to_mpfr<PB>(mpc_imagref(out),z.im); }

template<int PB>
__global__ void kern (int n, const cu_fcomplex<PB>*A, const cu_fcomplex<PB>*B,
                      cu_fcomplex<PB>*MU, cu_fcomplex<PB>*AD, cu_fcomplex<PB>*SU)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<n){ MU[i]=cu_cmul<PB>(A[i],B[i]); AD[i]=cu_cadd<PB>(A[i],B[i]); SU[i]=cu_csub<PB>(A[i],B[i]); }
}

template<int PB>
static int run (int M, cu_limb seed)
{
  typedef cu_fcomplex<PB> C;
  C *A=(C*)malloc(M*sizeof(C)), *B=(C*)malloc(M*sizeof(C));
  for (int i=0;i<M;i++){ int sp=(i&3)?40:(PB+200);
    A[i].re=randf<PB>(&seed,sp); A[i].im=randf<PB>(&seed,sp);
    B[i].re=randf<PB>(&seed,sp); B[i].im=randf<PB>(&seed,sp); }
  C *dA,*dB,*dMU,*dAD,*dSU;
  cudaMalloc(&dA,M*sizeof(C)); cudaMalloc(&dB,M*sizeof(C));
  cudaMalloc(&dMU,M*sizeof(C)); cudaMalloc(&dAD,M*sizeof(C)); cudaMalloc(&dSU,M*sizeof(C));
  cudaMemcpy(dA,A,M*sizeof(C),cudaMemcpyHostToDevice);
  cudaMemcpy(dB,B,M*sizeof(C),cudaMemcpyHostToDevice);
  kern<PB><<<(M+127)/128,128>>>(M,dA,dB,dMU,dAD,dSU);
  cudaError_t e=cudaDeviceSynchronize();
  if (e){ printf("PB=%4d KERNEL ERR %s\n",PB,cudaGetErrorString(e)); return 1; }
  C *MU=(C*)malloc(M*sizeof(C)),*AD=(C*)malloc(M*sizeof(C)),*SU=(C*)malloc(M*sizeof(C));
  cudaMemcpy(MU,dMU,M*sizeof(C),cudaMemcpyDeviceToHost);
  cudaMemcpy(AD,dAD,M*sizeof(C),cudaMemcpyDeviceToHost);
  cudaMemcpy(SU,dSU,M*sizeof(C),cudaMemcpyDeviceToHost);

  mpc_t ma,mb,ref,mine; mpc_init2(ma,PB);mpc_init2(mb,PB);mpc_init2(ref,PB);mpc_init2(mine,PB);
  long bm=0,ba=0,bs=0;
  for (int i=0;i<M;i++){
    c_to_mpc<PB>(ma,A[i]); c_to_mpc<PB>(mb,B[i]);
    mpc_mul(ref,ma,mb,MPC_RNDNN); c_to_mpc<PB>(mine,MU[i]); if(mpc_cmp(ref,mine)!=0) bm++;
    mpc_add(ref,ma,mb,MPC_RNDNN); c_to_mpc<PB>(mine,AD[i]); if(mpc_cmp(ref,mine)!=0) ba++;
    mpc_sub(ref,ma,mb,MPC_RNDNN); c_to_mpc<PB>(mine,SU[i]); if(mpc_cmp(ref,mine)!=0) bs++;
  }
  printf("PB=%4d (N=%2d SB=%2d): mul %ld  add %ld  sub %ld  / %d   %s\n",
         PB,cu_freal<PB>::N,cu_freal<PB>::SB,bm,ba,bs,M,(bm||ba||bs)?"FAIL":"OK");
  mpc_clear(ma);mpc_clear(mb);mpc_clear(ref);mpc_clear(mine);
  free(A);free(B);free(MU);free(AD);free(SU);
  cudaFree(dA);cudaFree(dB);cudaFree(dMU);cudaFree(dAD);cudaFree(dSU);
  return (bm||ba||bs)?1:0;
}

int main(void)
{
  cudaDeviceSetLimit(cudaLimitStackSize,(size_t)128*1024);
  const int M=100000; int bad=0;
  printf("=== cu_fcomplex<PB> vs system MPC (MPC_RNDNN), %d random pairs each ===\n",M);
  bad|=run<32>  (M,0x11ull);
  bad|=run<96>  (M,0x33ull);
  bad|=run<128> (M,0x44ull);
  bad|=run<160> (M,0x55ull);
  bad|=run<256> (M,0x66ull);
  bad|=run<288> (M,0x77ull);
  bad|=run<512> (M,0x88ull);
  bad|=run<1024>(M,0x99ull);
  bad|=run<1056>(M,0xaaull);
  bad|=run<2048>(M,0xbbull);
  printf("%s\n", bad?"*** FAILURES ***":"ALL PRECISIONS BIT-EXACT");
  return bad;
}
