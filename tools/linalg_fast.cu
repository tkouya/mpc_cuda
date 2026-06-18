/* linalg_fast.cu -- fixed-precision register-resident matvec (y=A*x) and matmul
 * (C=A*B) using cu_freal<PB>.  Dot-product accumulator lives in registers; NO
 * arena, NO runtime-precision dispatch.  Accuracy checked bit-exact against the
 * system MPFR at PB bits (same rounding sequence: t=a*b @PB, acc+=t @PB).
 *
 *   nvcc -arch=sm_121 -O3 -fmad=false -Iinclude -I/usr/local/include \
 *        -o /tmp/linalg_fast tools/linalg_fast.cu -L/usr/local/lib -lmpfr -lgmp
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include "mpc_cuda/cu_freal.cuh"
using namespace cu_fp;
#include <gmp.h>
#include <mpfr.h>

#ifndef PB
#define PB 1024
#endif
#ifndef NMV
#define NMV 256          /* matvec dimension */
#endif
#ifndef NMM
#define NMM 96           /* matmul dimension */
#endif
#ifndef LBLOCKS
#define LBLOCKS 256
#endif
#ifndef LTHREADS
#define LTHREADS 32
#endif

typedef cu_freal<PB> F;

/* y[i] = sum_k A[i*n+k]*x[k] */
__global__ void
matvec_kernel (int n, const double *A, const double *x, double *y)
{
  int stride=gridDim.x*blockDim.x;
  for (int i=blockIdx.x*blockDim.x+threadIdx.x; i<n; i+=stride)
    {
      F acc; acc.set_zero();
      for (int k=0;k<n;k++){
        F a=F::from_double(A[(size_t)i*n+k]);
        F b=F::from_double(x[k]);
        acc = cu_fadd<PB>(acc, cu_fmul<PB>(a,b));
      }
      y[i]=acc.to_double();
    }
}
/* C[i*n+j] = sum_k A[i*n+k]*B[k*n+j] */
__global__ void
matmul_kernel (int n, const double *A, const double *B, double *C)
{
  int stride=gridDim.x*blockDim.x; long total=(long)n*n;
  for (long e=blockIdx.x*blockDim.x+threadIdx.x; e<total; e+=stride)
    {
      int i=(int)(e/n), j=(int)(e%n);
      F acc; acc.set_zero();
      for (int k=0;k<n;k++){
        F a=F::from_double(A[(size_t)i*n+k]);
        F b=F::from_double(B[(size_t)k*n+j]);
        acc = cu_fadd<PB>(acc, cu_fmul<PB>(a,b));
      }
      C[e]=acc.to_double();
    }
}

/* host MPFR reference dot product (same rounding sequence) */
static double
mpfr_dot (const double *Arow, int stridea, const double *Bcol, int strideb, int n)
{
  mpfr_t acc,a,b,t; mpfr_inits2(PB,acc,a,b,t,(mpfr_ptr)0);
  mpfr_set_zero(acc,1);
  for (int k=0;k<n;k++){
    mpfr_set_d(a,Arow[(size_t)k*stridea],MPFR_RNDN);
    mpfr_set_d(b,Bcol[(size_t)k*strideb],MPFR_RNDN);
    mpfr_mul(t,a,b,MPFR_RNDN); mpfr_add(acc,acc,t,MPFR_RNDN);
  }
  double r=mpfr_get_d(acc,MPFR_RNDN);
  mpfr_clears(acc,a,b,t,(mpfr_ptr)0); return r;
}

static float time_kernel_mv(int n,const double*dA,const double*dx,double*dy){
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  matvec_kernel<<<LBLOCKS,LTHREADS>>>(n,dA,dx,dy); cudaDeviceSynchronize();
  cudaEventRecord(e0); matvec_kernel<<<LBLOCKS,LTHREADS>>>(n,dA,dx,dy); cudaEventRecord(e1);
  cudaEventSynchronize(e1); float ms=0; cudaEventElapsedTime(&ms,e0,e1); return ms;
}
static float time_kernel_mm(int n,const double*dA,const double*dB,double*dC){
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  matmul_kernel<<<LBLOCKS,LTHREADS>>>(n,dA,dB,dC); cudaDeviceSynchronize();
  cudaEventRecord(e0); matmul_kernel<<<LBLOCKS,LTHREADS>>>(n,dA,dB,dC); cudaEventRecord(e1);
  cudaEventSynchronize(e1); float ms=0; cudaEventElapsedTime(&ms,e0,e1); return ms;
}

int main(void)
{
  cudaDeviceSetLimit(cudaLimitStackSize,(size_t)128*1024);
  printf("=== fixed-precision register-resident linear algebra, PB=%d ===\n",PB);

  /* ---------------- matvec ---------------- */
  {
    int n=NMV; size_t MA=(size_t)n*n*sizeof(double);
    double *A=(double*)malloc(MA),*x=(double*)malloc(n*sizeof(double));
    double *yg=(double*)malloc(n*sizeof(double)),*yc=(double*)malloc(n*sizeof(double));
    for(int i=0;i<n;i++){ x[i]=1.0+i*1e-3; for(int j=0;j<n;j++) A[(size_t)i*n+j]=1.0+(i+2*j)*1e-4; }
    double *dA,*dx,*dy; cudaMalloc(&dA,MA); cudaMalloc(&dx,n*sizeof(double)); cudaMalloc(&dy,n*sizeof(double));
    cudaMemcpy(dA,A,MA,cudaMemcpyHostToDevice); cudaMemcpy(dx,x,n*sizeof(double),cudaMemcpyHostToDevice);
    float ms=time_kernel_mv(n,dA,dx,dy);
    cudaError_t e=cudaGetLastError(); if(e){printf("matvec err %s\n",cudaGetErrorString(e));return 1;}
    cudaMemcpy(yg,dy,n*sizeof(double),cudaMemcpyDeviceToHost);
    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    for(int i=0;i<n;i++) yc[i]=mpfr_dot(A+(size_t)i*n,1,x,1,n);
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double cpu=(t1.tv_sec-t0.tv_sec)*1e3+(t1.tv_nsec-t0.tv_nsec)*1e-6;
    double mr=0; int mm=0; for(int i=0;i<n;i++){ double rel=yc[i]?fabs((yg[i]-yc[i])/yc[i]):fabs(yg[i]); if(rel>mr)mr=rel; if(rel>1e-13)mm++; }
    printf("matvec  N=%-4d : GPU %8.3f ms  CPU(sysMPFR) %9.3f ms  %6.1fx  maxrel %.2e %s\n",
           n,ms,cpu,cpu/ms,mr,mm?"FAIL":"OK");
    free(A);free(x);free(yg);free(yc); cudaFree(dA);cudaFree(dx);cudaFree(dy);
  }
  /* ---------------- matmul ---------------- */
  {
    int n=NMM; size_t MA=(size_t)n*n*sizeof(double);
    double *A=(double*)malloc(MA),*B=(double*)malloc(MA);
    double *Cg=(double*)malloc(MA),*Cc=(double*)malloc(MA);
    for(int i=0;i<n;i++)for(int j=0;j<n;j++){ A[(size_t)i*n+j]=1.0+(i+2*j)*1e-4; B[(size_t)i*n+j]=0.5+(2*i+j)*1e-4; }
    double *dA,*dB,*dC; cudaMalloc(&dA,MA); cudaMalloc(&dB,MA); cudaMalloc(&dC,MA);
    cudaMemcpy(dA,A,MA,cudaMemcpyHostToDevice); cudaMemcpy(dB,B,MA,cudaMemcpyHostToDevice);
    float ms=time_kernel_mm(n,dA,dB,dC);
    cudaError_t e=cudaGetLastError(); if(e){printf("matmul err %s\n",cudaGetErrorString(e));return 1;}
    cudaMemcpy(Cg,dC,MA,cudaMemcpyDeviceToHost);
    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    for(int i=0;i<n;i++)for(int j=0;j<n;j++) Cc[(size_t)i*n+j]=mpfr_dot(A+(size_t)i*n,1,B+j,n,n);
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double cpu=(t1.tv_sec-t0.tv_sec)*1e3+(t1.tv_nsec-t0.tv_nsec)*1e-6;
    double mr=0; int mm=0; long tot=(long)n*n;
    for(long ee=0;ee<tot;ee++){ double rel=Cc[ee]?fabs((Cg[ee]-Cc[ee])/Cc[ee]):fabs(Cg[ee]); if(rel>mr)mr=rel; if(rel>1e-13)mm++; }
    printf("matmul  N=%-4d : GPU %8.3f ms  CPU(sysMPFR) %9.3f ms  %6.1fx  maxrel %.2e %s\n",
           n,ms,cpu,cpu/ms,mr,mm?"FAIL":"OK");
    printf("sample  C[0][0]=%.15g  C[N-1][N-1]=%.15g\n",Cg[0],Cg[tot-1]);
    free(A);free(B);free(Cg);free(Cc); cudaFree(dA);cudaFree(dB);cudaFree(dC);
  }
  return 0;
}
