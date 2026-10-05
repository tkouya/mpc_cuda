/* test_fcmath.cu -- Phase 4 complex elementary, max ULP vs system MPC. */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "cuda_host_shim.h"   /* also builds as plain C++ (CPU-only) */
#include "mpc_cuda/cu_fcmath.cuh"
using namespace cu_fp;
#include <gmp.h>
#include <mpfr.h>
#include <mpc.h>
static cu_limb xs(cu_limb*s){cu_limb x=*s;x^=x<<13;x^=x>>7;x^=x<<17;return *s=x;}
template<int PB> static cu_freal<PB> mk(cu_limb*s,long ex){
  cu_freal<PB> r;const int N=cu_freal<PB>::N,SB=cu_freal<PB>::SB;
  r.sign=(xs(s)&1)?1:-1; for(int i=0;i<N;i++)r.m[i]=xs(s);
  r.m[N-1]|=1ULL<<63; if(SB)r.m[0]&=~(((cu_limb)1<<SB)-1); r.exp=ex; return r;
}
template<int PB> static void f2m(mpfr_t o,const cu_freal<PB>&x){
  const int N=cu_freal<PB>::N; if(x.is_zero()){mpfr_set_zero(o,1);return;}
  mpz_t M;mpz_init(M);mpz_import(M,N,-1,sizeof(cu_limb),0,0,x.m);
  mpfr_set_z(o,M,MPFR_RNDN);mpfr_mul_2si(o,o,x.exp-N*64,MPFR_RNDN);
  if(x.sign<0)mpfr_neg(o,o,MPFR_RNDN);mpz_clear(M);
}
template<int PB> static double ulp1(mpfr_t ref,const cu_freal<PB>&mine){
  if(mpfr_zero_p(ref))return mine.is_zero()?0.0:1e30;
  mpfr_t m;mpfr_init2(m,PB+80);f2m<PB>(m,mine);
  mpfr_t d;mpfr_init2(d,PB+80);mpfr_sub(d,ref,m,MPFR_RNDN);mpfr_abs(d,d,MPFR_RNDN);
  mpfr_mul_2si(d,d,(long)PB-mpfr_get_exp(ref),MPFR_RNDN);
  double u=mpfr_get_d(d,MPFR_RNDN);mpfr_clear(m);mpfr_clear(d);return u;
}
template<int PB> static double culp(mpc_t ref,const cu_fcomplex<PB>&z){
  double a=ulp1<PB>(mpc_realref(ref),z.re), b=ulp1<PB>(mpc_imagref(ref),z.im);
  return a>b?a:b;
}
enum{CEXP,CLOG,CSQRT,CSIN,CCOS,CTAN,CSINH,CCOSH,NF};
const char*NM[NF]={"cexp","clog","csqrt","csin","ccos","ctan","csinh","ccosh"};
template<int PB> __global__ void kern(int n,int fn,const cu_fcomplex<PB>*A,cu_fcomplex<PB>*R){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;cu_fcomplex<PB> z=A[i];
  switch(fn){case CEXP:R[i]=cu_cexp<PB>(z);break;case CLOG:R[i]=cu_clog<PB>(z);break;
    case CSQRT:R[i]=cu_csqrt<PB>(z);break;case CSIN:R[i]=cu_csin<PB>(z);break;
    case CCOS:R[i]=cu_ccos<PB>(z);break;case CTAN:R[i]=cu_ctan<PB>(z);break;
    case CSINH:R[i]=cu_csinh<PB>(z);break;case CCOSH:R[i]=cu_ccosh<PB>(z);break;}
}
template<int PB> static void run(int M,cu_limb seed){
  typedef cu_fcomplex<PB> C; C*A=(C*)malloc(M*sizeof(C)),*R=(C*)malloc(M*sizeof(C));
  C*dA,*dR;cudaMalloc(&dA,M*sizeof(C));cudaMalloc(&dR,M*sizeof(C));
  mpc_t ref;mpc_init2(ref,PB); mpc_t a;mpc_init2(a,PB);
  double mx[NF];for(int f=0;f<NF;f++)mx[f]=0;
  for(int fn=0;fn<NF;fn++){
    for(int i=0;i<M;i++){ long ex=(fn==CEXP||fn==CSIN||fn==CCOS||fn==CTAN)?((long)(xs(&seed)%5)-2):((long)(xs(&seed)%7)-3);
      A[i].re=mk<PB>(&seed,ex); A[i].im=mk<PB>(&seed,ex); }
    cudaMemcpy(dA,A,M*sizeof(C),cudaMemcpyHostToDevice);
    CU_LAUNCH(kern<PB>, M, M,fn,dA,dR);
    cudaError_t e=cudaDeviceSynchronize();if(e){printf("PB=%d %s ERR %s\n",PB,NM[fn],cudaGetErrorString(e));continue;}
    cudaMemcpy(R,dR,M*sizeof(C),cudaMemcpyDeviceToHost);
    for(int i=0;i<M;i++){ f2m<PB>(mpc_realref(a),A[i].re); f2m<PB>(mpc_imagref(a),A[i].im);
      switch(fn){case CEXP:mpc_exp(ref,a,MPC_RNDNN);break;case CLOG:mpc_log(ref,a,MPC_RNDNN);break;
        case CSQRT:mpc_sqrt(ref,a,MPC_RNDNN);break;case CSIN:mpc_sin(ref,a,MPC_RNDNN);break;
        case CCOS:mpc_cos(ref,a,MPC_RNDNN);break;case CTAN:mpc_tan(ref,a,MPC_RNDNN);break;
        case CSINH:mpc_sinh(ref,a,MPC_RNDNN);break;case CCOSH:mpc_cosh(ref,a,MPC_RNDNN);break;}
      double u=culp<PB>(ref,R[i]); if(u>mx[fn])mx[fn]=u;
    }
  }
  printf("PB=%4d :",PB);for(int f=0;f<NF;f++)printf(" %s %.2f",NM[f],mx[f]);printf("  ULP\n");
  mpc_clear(ref);mpc_clear(a);free(A);free(R);cudaFree(dA);cudaFree(dR);
}
int main(void){
  cudaDeviceSetLimit(cudaLimitStackSize,(size_t)512*1024);
  const int M=3000; printf("=== cu_fp Phase-4 complex elementary, max ULP vs MPC, %d samples ===\n",M);
  run<64>(M,0x11);run<128>(M,0x22);run<256>(M,0x33);
  return 0;
}
