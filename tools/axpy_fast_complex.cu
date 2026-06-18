/* axpy_fast_complex.cu -- fixed-precision register-resident COMPLEX AXPY
 * z = a*x + y using cu_fcomplex<PB>, vs system MPC on the CPU.  Compare to the
 * existing cu_mpc-arena GPU path (./build/axpy_mpc, run separately).
 *
 *   nvcc -arch=sm_121 -O3 -fmad=false -Iinclude -I/usr/local/include \
 *        -o /tmp/axfc tools/axpy_fast_complex.cu -L/usr/local/lib -lmpc -lmpfr -lgmp
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include "mpc_cuda/cu_fcomplex.cuh"
using namespace cu_fp;
#include <gmp.h>
#include <mpfr.h>
#include <mpc.h>

#ifndef PB
#define PB 1024
#endif
#ifndef NN
#define NN 262144
#endif
#ifndef LBLOCKS
#define LBLOCKS 256
#endif
#ifndef LTHREADS
#define LTHREADS 32
#endif
typedef cu_fcomplex<PB> C;

__global__ void axpyc (int n, double ar, double ai, const double *xr, const double *xi,
                       const double *yr, const double *yi, double *zr, double *zi)
{
  int s=gridDim.x*blockDim.x; C a(ar,ai);
  for (int i=blockIdx.x*blockDim.x+threadIdx.x; i<n; i+=s){
    C z = a*C(xr[i],xi[i]) + C(yr[i],yi[i]);
    zr[i]=z.real_d(); zi[i]=z.imag_d();
  }
}

int main(void)
{
  cudaDeviceSetLimit(cudaLimitStackSize,(size_t)128*1024);
  double ar=1.5, ai=-0.25;
  int n=NN; size_t B=n*sizeof(double);
  double *xr=(double*)malloc(B),*xi=(double*)malloc(B),*yr=(double*)malloc(B),*yi=(double*)malloc(B);
  double *zr=(double*)malloc(B),*zi=(double*)malloc(B);
  for(int i=0;i<n;i++){ xr[i]=1.0+i*1e-4; xi[i]=0.5-i*3e-5; yr[i]=2.0-i*7e-5; yi[i]=-1.0+i*2e-5; }
  double *dxr,*dxi,*dyr,*dyi,*dzr,*dzi;
  cudaMalloc(&dxr,B);cudaMalloc(&dxi,B);cudaMalloc(&dyr,B);cudaMalloc(&dyi,B);cudaMalloc(&dzr,B);cudaMalloc(&dzi,B);
  cudaMemcpy(dxr,xr,B,cudaMemcpyHostToDevice);cudaMemcpy(dxi,xi,B,cudaMemcpyHostToDevice);
  cudaMemcpy(dyr,yr,B,cudaMemcpyHostToDevice);cudaMemcpy(dyi,yi,B,cudaMemcpyHostToDevice);
  cudaEvent_t e0,e1; cudaEventCreate(&e0);cudaEventCreate(&e1);
  const int IT=20;
  axpyc<<<LBLOCKS,LTHREADS>>>(n,ar,ai,dxr,dxi,dyr,dyi,dzr,dzi); cudaDeviceSynchronize();
  cudaEventRecord(e0);
  for(int it=0;it<IT;it++) axpyc<<<LBLOCKS,LTHREADS>>>(n,ar,ai,dxr,dxi,dyr,dyi,dzr,dzi);
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float tot=0; cudaEventElapsedTime(&tot,e0,e1); float gpu=tot/IT;
  cudaError_t e=cudaGetLastError(); if(e){printf("err %s\n",cudaGetErrorString(e));return 1;}
  cudaMemcpy(zr,dzr,B,cudaMemcpyDeviceToHost); cudaMemcpy(zi,dzi,B,cudaMemcpyDeviceToHost);

  /* CPU reference: system MPC at PB bits */
  mpc_t a,x,y,t; mpc_init2(a,PB);mpc_init2(x,PB);mpc_init2(y,PB);mpc_init2(t,PB);
  mpc_set_d_d(a,ar,ai,MPC_RNDNN);
  struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
  double mr=0; int mm=0;
  for(int i=0;i<n;i++){
    mpc_set_d_d(x,xr[i],xi[i],MPC_RNDNN); mpc_set_d_d(y,yr[i],yi[i],MPC_RNDNN);
    mpc_mul(t,a,x,MPC_RNDNN); mpc_add(t,t,y,MPC_RNDNN);
    double cr=mpfr_get_d(mpc_realref(t),MPFR_RNDN), ci=mpfr_get_d(mpc_imagref(t),MPFR_RNDN);
    double rr=cr?fabs((zr[i]-cr)/cr):fabs(zr[i]); double ri=ci?fabs((zi[i]-ci)/ci):fabs(zi[i]);
    if(rr>mr)mr=rr; if(ri>mr)mr=ri; if(rr>1e-13||ri>1e-13)mm++;
  }
  clock_gettime(CLOCK_MONOTONIC,&t1);
  double cpu=(t1.tv_sec-t0.tv_sec)*1e3+(t1.tv_nsec-t0.tv_nsec)*1e-6;
  printf("=== fixed-precision complex AXPY (z=a*x+y), PB=%d, N=%d ===\n",PB,n);
  printf("GPU time : %8.3f ms   (cu_fcomplex<%d>, register-resident, NO arena)\n",gpu,PB);
  printf("CPU time : %8.3f ms   (system MPC on host)\n",cpu);
  printf("speedup  : %8.2fx (GPU vs system CPU MPC)\n",cpu/gpu);
  printf("accuracy : max rel = %.3e (%d exceed 1e-13)\n",mr,mm);
  printf("sample   : z[0]=(%.12g, %.12g)\n",zr[0],zi[0]);
  return mm?1:0;
}
