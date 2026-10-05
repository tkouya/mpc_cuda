/* test_fmath.cu -- accuracy (max ULP) of cu_fp elementary functions vs MPFR.
 *   nvcc -arch=sm_121 -O3 -fmad=false -Iinclude -I/usr/local/include \
 *        -o /tmp/test_fmath tools/test_fmath.cu -L/usr/local/lib -lmpfr -lgmp
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "cuda_host_shim.h"   /* also builds as plain C++ (CPU-only) */
#include "mpc_cuda/cu_fmath.cuh"
using namespace cu_fp;
#include <gmp.h>
#include <mpfr.h>

static cu_limb xs(cu_limb*s){cu_limb x=*s;x^=x<<13;x^=x>>7;x^=x<<17;return *s=x;}
/* random value with mantissa full-width, exponent in [-spread,spread], sign per flag */
template<int PB> static cu_freal<PB> randf(cu_limb*s,int spread,bool pos){
  cu_freal<PB> r;const int N=cu_freal<PB>::N,SB=cu_freal<PB>::SB;
  r.sign=(pos||(xs(s)&1))?1:-1; for(int i=0;i<N;i++)r.m[i]=xs(s);
  r.m[N-1]|=1ULL<<63; if(SB)r.m[0]&=~(((cu_limb)1<<SB)-1);
  r.exp=(long)(xs(s)%(2*spread+1))-spread; return r;
}
template<int PB> static void to_mpfr(mpfr_t o,const cu_freal<PB>&x){
  const int N=cu_freal<PB>::N; if(x.is_zero()){mpfr_set_zero(o,1);return;}
  mpz_t M;mpz_init(M);mpz_import(M,N,-1,sizeof(cu_limb),0,0,x.m);
  mpfr_set_z(o,M,MPFR_RNDN);mpfr_mul_2si(o,o,x.exp-N*64,MPFR_RNDN);
  if(x.sign<0)mpfr_neg(o,o,MPFR_RNDN);mpz_clear(M);
}
template<int PB> static double ulp_err(mpfr_t ref,const cu_freal<PB>&mine){
  if(mpfr_zero_p(ref)) return mine.is_zero()?0.0:1e30;
  mpfr_t m;mpfr_init2(m,PB+80);to_mpfr<PB>(m,mine);
  mpfr_t d;mpfr_init2(d,PB+80);mpfr_sub(d,ref,m,MPFR_RNDN);mpfr_abs(d,d,MPFR_RNDN);
  mpfr_mul_2si(d,d,(long)PB-mpfr_get_exp(ref),MPFR_RNDN);
  double u=mpfr_get_d(d,MPFR_RNDN);mpfr_clear(m);mpfr_clear(d);return u;
}

enum {EXP,LOG,EXPM1,LOG1P,SIN,COS,TAN,ATAN,SINH,COSH,NF};
const char* NM[NF]={"exp","log","expm1","log1p","sin","cos","tan","atan","sinh","cosh"};
template<int PB> __global__ void kern(int n,int fn,const cu_freal<PB>*A,cu_freal<PB>*R){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; cu_freal<PB> x=A[i];
  switch(fn){
    case EXP:R[i]=cu_exp<PB>(x);break; case LOG:R[i]=cu_log<PB>(cu_abs<PB>(x));break;
    case EXPM1:R[i]=cu_expm1<PB>(x);break; case LOG1P:R[i]=cu_log1p<PB>(x);break;
    case SIN:R[i]=cu_sin<PB>(x);break; case COS:R[i]=cu_cos<PB>(x);break;
    case TAN:R[i]=cu_tan<PB>(x);break; case ATAN:R[i]=cu_atan<PB>(x);break;
    case SINH:R[i]=cu_sinh<PB>(x);break; case COSH:R[i]=cu_cosh<PB>(x);break;
  }
}
template<int PB> static void run(int M,cu_limb seed){
  typedef cu_freal<PB> F;
  F*A=(F*)malloc(M*sizeof(F)),*R=(F*)malloc(M*sizeof(F));
  F*dA,*dR;cudaMalloc(&dA,M*sizeof(F));cudaMalloc(&dR,M*sizeof(F));
  mpfr_t a,ref;mpfr_inits2(PB,a,ref,(mpfr_ptr)0);
  double mx[NF]; for(int f=0;f<NF;f++)mx[f]=0;
  for(int fn=0;fn<NF;fn++){
    /* per-function arg range */
    int spread = (fn==LOG)?20 : (fn==SIN||fn==COS||fn==TAN)?5 : 3;
    bool pos = (fn==LOG);
    bool small = (fn==EXPM1||fn==LOG1P);
    for(int i=0;i<M;i++){ A[i]=randf<PB>(&seed,small?6:spread,pos);
      if(small){ A[i].exp -= 6; if(fn==LOG1P) {} } }   /* expm1/log1p: smaller args */
    cudaMemcpy(dA,A,M*sizeof(F),cudaMemcpyHostToDevice);
    CU_LAUNCH(kern<PB>, M, M,fn,dA,dR);
    cudaError_t e=cudaDeviceSynchronize(); if(e){printf("PB=%d %s ERR %s\n",PB,NM[fn],cudaGetErrorString(e));continue;}
    cudaMemcpy(R,dR,M*sizeof(F),cudaMemcpyDeviceToHost);
    for(int i=0;i<M;i++){ to_mpfr<PB>(a,A[i]);
      switch(fn){
        case EXP:mpfr_exp(ref,a,MPFR_RNDN);break;
        case LOG:mpfr_abs(a,a,MPFR_RNDN);mpfr_log(ref,a,MPFR_RNDN);break;
        case EXPM1:mpfr_expm1(ref,a,MPFR_RNDN);break;
        case LOG1P:if(mpfr_cmp_si(a,-1)<=0)continue;mpfr_log1p(ref,a,MPFR_RNDN);break;
        case SIN:mpfr_sin(ref,a,MPFR_RNDN);break; case COS:mpfr_cos(ref,a,MPFR_RNDN);break;
        case TAN:mpfr_tan(ref,a,MPFR_RNDN);break; case ATAN:mpfr_atan(ref,a,MPFR_RNDN);break;
        case SINH:mpfr_sinh(ref,a,MPFR_RNDN);break; case COSH:mpfr_cosh(ref,a,MPFR_RNDN);break;
      }
      double u=ulp_err<PB>(ref,R[i]); if(u>mx[fn])mx[fn]=u;
    }
  }
  printf("PB=%4d :",PB); for(int f=0;f<NF;f++)printf(" %s %.2f",NM[f],mx[f]); printf("  ULP\n");
  mpfr_clears(a,ref,(mpfr_ptr)0);
  free(A);free(R);cudaFree(dA);cudaFree(dR);
}
int main(void){
  cudaDeviceSetLimit(cudaLimitStackSize,(size_t)512*1024);
  const int M=5000;
  printf("=== cu_fp Phase-2 elementary, max ULP vs MPFR, %d samples ===\n",M);
  run<64>(M,0x11);run<128>(M,0x22);run<256>(M,0x33);run<512>(M,0x44);run<1024>(M,0x55);
  return 0;
}
