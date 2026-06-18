/* bench_fmath.cu -- throughput of cu_fp elementary functions (register-resident)
 * vs system MPFR on the CPU.  PB compile-time.
 *   nvcc -arch=sm_121 -O3 -fmad=false -Iinclude -I/usr/local/include \
 *        -o /tmp/bench_fmath tools/bench_fmath.cu -L/usr/local/lib -lmpfr -lgmp
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include "mpc_cuda/cu_fmath.cuh"
using namespace cu_fp;
#include <gmp.h>
#include <mpfr.h>
#ifndef PB
#define PB 256
#endif
#ifndef NN
#define NN 200000
#endif
#define LB 256
#define LT 64
typedef cu_freal<PB> F;
enum{EXP,LOG,SIN};
__global__ void k(int n,int fn,const double*X,double*O){
  int s=gridDim.x*blockDim.x;
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=s){
    F x=F::from_double(X[i]); F r;
    if(fn==EXP)r=cu_exp<PB>(x); else if(fn==LOG)r=cu_log<PB>(cu_abs<PB>(x)); else r=cu_sin<PB>(x);
    O[i]=r.to_double();
  }
}
static float tk(int fn,int n,const double*dX,double*dO){
  cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);
  k<<<LB,LT>>>(n,fn,dX,dO);cudaDeviceSynchronize();
  cudaEventRecord(a);for(int i=0;i<10;i++)k<<<LB,LT>>>(n,fn,dX,dO);cudaEventRecord(b);cudaEventSynchronize(b);
  float t=0;cudaEventElapsedTime(&t,a,b);return t/10;
}
int main(void){
  cudaDeviceSetLimit(cudaLimitStackSize,(size_t)256*1024);
  int n=NN; double*X=(double*)malloc(n*sizeof(double)),*O=(double*)malloc(n*sizeof(double));
  for(int i=0;i<n;i++)X[i]=0.5+ (i%1000)*1e-3;
  double*dX,*dO;cudaMalloc(&dX,n*sizeof(double));cudaMalloc(&dO,n*sizeof(double));
  cudaMemcpy(dX,X,n*sizeof(double),cudaMemcpyHostToDevice);
  const char*nm[3]={"exp","log","sin"};
  printf("=== cu_fp elementary throughput, PB=%d, N=%d ===\n",PB,n);
  mpfr_t a,r;mpfr_inits2(PB,a,r,(mpfr_ptr)0);
  for(int fn=0;fn<3;fn++){
    float g=tk(fn,n,dX,dO); cudaMemcpy(O,dO,n*sizeof(double),cudaMemcpyDeviceToHost);
    struct timespec t0,t1;clock_gettime(CLOCK_MONOTONIC,&t0);
    for(int i=0;i<n;i++){ mpfr_set_d(a,X[i],MPFR_RNDN);
      if(fn==EXP)mpfr_exp(r,a,MPFR_RNDN); else if(fn==LOG){mpfr_abs(a,a,MPFR_RNDN);mpfr_log(r,a,MPFR_RNDN);} else mpfr_sin(r,a,MPFR_RNDN); }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double c=(t1.tv_sec-t0.tv_sec)*1e3+(t1.tv_nsec-t0.tv_nsec)*1e-6;
    printf("%-4s : GPU %7.3f ms   CPU(sysMPFR) %8.3f ms   %6.1fx\n",nm[fn],g,c,c/g);
  }
  return 0;
}
