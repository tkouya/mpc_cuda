/* test_fmath3.cu -- Phase 3 (asin/acos/atanh/pow) max ULP vs MPFR. */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "mpc_cuda/cu_fmath.cuh"
using namespace cu_fp;
#include <gmp.h>
#include <mpfr.h>
static cu_limb xs(cu_limb*s){cu_limb x=*s;x^=x<<13;x^=x>>7;x^=x<<17;return *s=x;}
template<int PB> static cu_freal<PB> mk(cu_limb*s,long expv,bool pos){
  cu_freal<PB> r;const int N=cu_freal<PB>::N,SB=cu_freal<PB>::SB;
  r.sign=(pos||(xs(s)&1))?1:-1; for(int i=0;i<N;i++)r.m[i]=xs(s);
  r.m[N-1]|=1ULL<<63; if(SB)r.m[0]&=~(((cu_limb)1<<SB)-1); r.exp=expv; return r;
}
template<int PB> static void to_mpfr(mpfr_t o,const cu_freal<PB>&x){
  const int N=cu_freal<PB>::N; if(x.is_zero()){mpfr_set_zero(o,1);return;}
  mpz_t M;mpz_init(M);mpz_import(M,N,-1,sizeof(cu_limb),0,0,x.m);
  mpfr_set_z(o,M,MPFR_RNDN);mpfr_mul_2si(o,o,x.exp-N*64,MPFR_RNDN);
  if(x.sign<0)mpfr_neg(o,o,MPFR_RNDN);mpz_clear(M);
}
template<int PB> static double ulp(mpfr_t ref,const cu_freal<PB>&mine){
  if(mpfr_zero_p(ref))return mine.is_zero()?0.0:1e30;
  mpfr_t m;mpfr_init2(m,PB+80);to_mpfr<PB>(m,mine);
  mpfr_t d;mpfr_init2(d,PB+80);mpfr_sub(d,ref,m,MPFR_RNDN);mpfr_abs(d,d,MPFR_RNDN);
  mpfr_mul_2si(d,d,(long)PB-mpfr_get_exp(ref),MPFR_RNDN);
  double u=mpfr_get_d(d,MPFR_RNDN);mpfr_clear(m);mpfr_clear(d);return u;
}
enum{ASIN,ACOS,ATANH,POW,NF};
const char*NM[NF]={"asin","acos","atanh","pow"};
template<int PB> __global__ void kern(int n,int fn,const cu_freal<PB>*A,const cu_freal<PB>*B,cu_freal<PB>*R){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  if(fn==ASIN)R[i]=cu_asin<PB>(A[i]); else if(fn==ACOS)R[i]=cu_acos<PB>(A[i]);
  else if(fn==ATANH)R[i]=cu_atanh<PB>(A[i]); else R[i]=cu_pow<PB>(A[i],B[i]);
}
template<int PB> static void run(int M,cu_limb seed){
  typedef cu_freal<PB> F; F*A=(F*)malloc(M*sizeof(F)),*B=(F*)malloc(M*sizeof(F)),*R=(F*)malloc(M*sizeof(F));
  F*dA,*dB,*dR;cudaMalloc(&dA,M*sizeof(F));cudaMalloc(&dB,M*sizeof(F));cudaMalloc(&dR,M*sizeof(F));
  mpfr_t a,b,ref;mpfr_inits2(PB,a,b,ref,(mpfr_ptr)0); double mx[NF];for(int f=0;f<NF;f++)mx[f]=0;
  for(int fn=0;fn<NF;fn++){
    for(int i=0;i<M;i++){
      if(fn==POW){ A[i]=mk<PB>(&seed,(long)(xs(&seed)%7)-3,true);  /* x>0, exp[-3,3] */
                   B[i]=mk<PB>(&seed,(long)(xs(&seed)%5)-2,false);} /* y exp[-2,2] */
      else { A[i]=mk<PB>(&seed,-(long)(xs(&seed)%5),false); }       /* |x|<1: exp[-4,0] */
    }
    cudaMemcpy(dA,A,M*sizeof(F),cudaMemcpyHostToDevice);cudaMemcpy(dB,B,M*sizeof(F),cudaMemcpyHostToDevice);
    kern<PB><<<(M+127)/128,128>>>(M,fn,dA,dB,dR);
    cudaError_t e=cudaDeviceSynchronize();if(e){printf("PB=%d %s ERR %s\n",PB,NM[fn],cudaGetErrorString(e));continue;}
    cudaMemcpy(R,dR,M*sizeof(F),cudaMemcpyDeviceToHost);
    for(int i=0;i<M;i++){ to_mpfr<PB>(a,A[i]);
      if(fn==ASIN)mpfr_asin(ref,a,MPFR_RNDN);
      else if(fn==ACOS)mpfr_acos(ref,a,MPFR_RNDN);
      else if(fn==ATANH)mpfr_atanh(ref,a,MPFR_RNDN);
      else { to_mpfr<PB>(b,B[i]); mpfr_pow(ref,a,b,MPFR_RNDN); }
      double u=ulp<PB>(ref,R[i]); if(u>mx[fn])mx[fn]=u;
    }
  }
  printf("PB=%4d :",PB); for(int f=0;f<NF;f++)printf(" %s %.2f",NM[f],mx[f]); printf("  ULP\n");
  mpfr_clears(a,b,ref,(mpfr_ptr)0);
  free(A);free(B);free(R);cudaFree(dA);cudaFree(dB);cudaFree(dR);
}
int main(void){
  cudaDeviceSetLimit(cudaLimitStackSize,(size_t)512*1024);
  const int M=5000; printf("=== cu_fp Phase-3 (asin/acos/atanh/pow) max ULP vs MPFR, %d samples ===\n",M);
  run<64>(M,0x11);run<128>(M,0x22);run<256>(M,0x33);run<512>(M,0x44);run<1024>(M,0x55);
  return 0;
}
