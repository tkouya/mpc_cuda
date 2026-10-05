/* bench_gpu_fixed.cu -- GPU throughput of cu_freal<PB> / cu_fcomplex<PB>
 * element-wise operations (mul, add, axpy r=a*x+y, complex mul and axpy) on
 * arrays of full random PB-bit operands in global memory, grid-striding over
 * the whole device, against the same operations on ALL CPU cores with the
 * cu_freal host path and with FLINT nfloat (tools/bench_gpu_fixed_cpu.cpp).
 * Every GPU result sampled is checked bit-exact against the host path, and the
 * warp-staged variant (cu_fwarp.cuh) element by element against the plain one.
 *
 *   make bench-gpu-fixed FLINT_PREFIX=...        (or ./build/bench_gpu_fixed -g: GPU only)
 * Options: -DBG_M=elements (default 2^20), -DBG_TPB=threads/block (128),
 *          -DBG_OPMASK=bits of {mul,add,axpy,cmul,caxpy}, -DBG_PBMAX=bits.
 * Each operation runs twice: plain (each thread reads its element, 8/16 bytes
 * at a 16+8N-byte stride, ~1-1.6 TB/s on an H100) and warp-staged (coalesced
 * tiles through shared memory with cu_warp_load/cu_warp_store).  The warp
 * variant is skipped ("n/a") when its buffer, 32 elements per warp, exceeds the
 * device's shared memory per block (e.g. 4096-bit complex on a GB10, 99 KB). */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "bench_gpu_fixed.h"
#include "mpc_cuda/cu_fwarp.cuh"

#ifndef BG_M
#define BG_M (1L<<20)
#endif
#ifndef BG_TPB
#define BG_TPB 128
#endif
#ifndef BG_OPMASK
#define BG_OPMASK 31
#endif
#ifndef BG_PBMAX
#define BG_PBMAX 4096
#endif
#define MEL BG_M
#define TPB BG_TPB
#define OPMASK BG_OPMASK
#define PBMAX BG_PBMAX
static const char *opn[5]={"mul","add","axpy","cmul","caxpy"};

template<int PB> __global__ void k_mul (long n, const cu_freal<PB>*x, const cu_freal<PB>*y, cu_freal<PB>*r)
{ for (long i=blockIdx.x*(long)blockDim.x+threadIdx.x;i<n;i+=(long)gridDim.x*blockDim.x) r[i]=cu_fmul<PB>(x[i],y[i]); }
template<int PB> __global__ void k_add (long n, const cu_freal<PB>*x, const cu_freal<PB>*y, cu_freal<PB>*r)
{ for (long i=blockIdx.x*(long)blockDim.x+threadIdx.x;i<n;i+=(long)gridDim.x*blockDim.x) r[i]=cu_fadd<PB>(x[i],y[i]); }
template<int PB> __global__ void k_axpy (long n, cu_freal<PB> a, const cu_freal<PB>*x, const cu_freal<PB>*y, cu_freal<PB>*r)
{ for (long i=blockIdx.x*(long)blockDim.x+threadIdx.x;i<n;i+=(long)gridDim.x*blockDim.x) r[i]=cu_fadd<PB>(cu_fmul<PB>(a,x[i]),y[i]); }
template<int PB> __global__ void k_cmul (long n, const cu_fcomplex<PB>*x, const cu_fcomplex<PB>*y, cu_fcomplex<PB>*r)
{ for (long i=blockIdx.x*(long)blockDim.x+threadIdx.x;i<n;i+=(long)gridDim.x*blockDim.x) r[i]=cu_cmul<PB>(x[i],y[i]); }
template<int PB> __global__ void k_caxpy (long n, cu_fcomplex<PB> a, const cu_fcomplex<PB>*x, const cu_fcomplex<PB>*y, cu_fcomplex<PB>*r)
{ for (long i=blockIdx.x*(long)blockDim.x+threadIdx.x;i<n;i+=(long)gridDim.x*blockDim.x) r[i]=cu_cadd<PB>(cu_cmul<PB>(a,x[i]),y[i]); }

/* the same kernels with warp-cooperative (coalesced) loads/stores, cu_fwarp.cuh */
#define WARP_LOOP(T) T *wb = cu_fwarp_dyn<T> (); \
  for (long base = cu_warp_first_tile (); base < n; base += cu_warp_tile_stride ())
template<int PB> __global__ void kw_mul (long n, const cu_freal<PB>*x, const cu_freal<PB>*y, cu_freal<PB>*r)
{ WARP_LOOP(cu_freal<PB>) { cu_freal<PB> u=cu_warp_load(x,base,n,wb), v=cu_warp_load(y,base,n,wb);
    cu_warp_store(r,base,n,cu_fmul<PB>(u,v),wb); } }
template<int PB> __global__ void kw_add (long n, const cu_freal<PB>*x, const cu_freal<PB>*y, cu_freal<PB>*r)
{ WARP_LOOP(cu_freal<PB>) { cu_freal<PB> u=cu_warp_load(x,base,n,wb), v=cu_warp_load(y,base,n,wb);
    cu_warp_store(r,base,n,cu_fadd<PB>(u,v),wb); } }
template<int PB> __global__ void kw_axpy (long n, cu_freal<PB> a, const cu_freal<PB>*x, const cu_freal<PB>*y, cu_freal<PB>*r)
{ WARP_LOOP(cu_freal<PB>) { cu_freal<PB> u=cu_warp_load(x,base,n,wb), v=cu_warp_load(y,base,n,wb);
    cu_warp_store(r,base,n,cu_fadd<PB>(cu_fmul<PB>(a,u),v),wb); } }
template<int PB> __global__ void kw_cmul (long n, const cu_fcomplex<PB>*x, const cu_fcomplex<PB>*y, cu_fcomplex<PB>*r)
{ WARP_LOOP(cu_fcomplex<PB>) { cu_fcomplex<PB> u=cu_warp_load(x,base,n,wb), v=cu_warp_load(y,base,n,wb);
    cu_warp_store(r,base,n,cu_cmul<PB>(u,v),wb); } }
template<int PB> __global__ void kw_caxpy (long n, cu_fcomplex<PB> a, const cu_fcomplex<PB>*x, const cu_fcomplex<PB>*y, cu_fcomplex<PB>*r)
{ WARP_LOOP(cu_fcomplex<PB>) { cu_fcomplex<PB> u=cu_warp_load(x,base,n,wb), v=cu_warp_load(y,base,n,wb);
    cu_warp_store(r,base,n,cu_cadd<PB>(cu_cmul<PB>(a,u),v),wb); } }

template<class L> static double gtime (L launch, long n)
{
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  launch(); cudaDeviceSynchronize();
  float best=1e30f;
  for (int p=0;p<5;p++){ cudaEventRecord(a); launch(); cudaEventRecord(b); cudaEventSynchronize(b);
    float t; cudaEventElapsedTime(&t,a,b); if (t<best) best=t; }
  cudaError_t e=cudaGetLastError(); if (e!=cudaSuccess){ printf("CUDA error %s\n",cudaGetErrorString(e)); exit(1); }
  return best*1e6/n;                                /* ns per element */
}

template<int PB> static bool same (const cu_freal<PB>&a, const cu_freal<PB>&b)
{ if (a.is_zero()||b.is_zero()) return a.is_zero()==b.is_zero();
  return a.sign==b.sign && a.exp==b.exp && !memcmp(a.m,b.m,sizeof a.m); }

template<int PB> static void bench (int nsm, bool docpu)
{
  if constexpr (PB > PBMAX) return; else {
  typedef cu_freal<PB> F; typedef cu_fcomplex<PB> C;
  const long M=MEL; cu_limb seed=0x12345678ULL^PB;
  std::vector<F> x(M), y(M), r(M), rw(M); F a=bg_rand<PB>(&seed);
  for (long i=0;i<M;i++){ x[i]=bg_rand<PB>(&seed); y[i]=bg_rand<PB>(&seed); }
  C ca(a,x[0]);
  F *dx,*dy,*dr; size_t B=M*sizeof(F);
  cudaMalloc(&dx,B); cudaMalloc(&dy,B); cudaMalloc(&dr,B);
  cudaMemcpy(dx,x.data(),B,cudaMemcpyHostToDevice); cudaMemcpy(dy,y.data(),B,cudaMemcpyHostToDevice);
  for (int op=0;op<5;op++){
    if (!((OPMASK>>op)&1)) continue;
    const long n = op>=3 ? M/2 : M;
    int bl=0, blw=0;
    double tg=0, tw=0;
    switch (op){
#define LAUNCH(BL, TG, K, ...) { cudaOccupancyMaxActiveBlocksPerMultiprocessor(&BL, K, TPB, 0); \
      int g=BL*nsm; if ((long)g*TPB > n) g=(n+TPB-1)/TPB; \
      TG=gtime([&]{ K<<<g,TPB>>>(__VA_ARGS__); }, n); }
    case 0: if constexpr ((OPMASK>>0)&1) LAUNCH(bl,tg,k_mul<PB>, n,dx,dy,dr); break;
    case 1: if constexpr ((OPMASK>>1)&1) LAUNCH(bl,tg,k_add<PB>, n,dx,dy,dr); break;
    case 2: if constexpr ((OPMASK>>2)&1) LAUNCH(bl,tg,k_axpy<PB>, n,a,dx,dy,dr); break;
    case 3: if constexpr ((OPMASK>>3)&1) LAUNCH(bl,tg,k_cmul<PB>, n,(C*)dx,(C*)dy,(C*)dr); break;
    case 4: if constexpr ((OPMASK>>4)&1) LAUNCH(bl,tg,k_caxpy<PB>, n,ca,(C*)dx,(C*)dy,(C*)dr); break;
    }
    cudaMemcpy(r.data(),dr,B,cudaMemcpyDeviceToHost);
    cudaMemset(dr,0,B);
    switch (op){
#define LAUNCHW(BL, TG, T, K, ...) { const size_t sm = cu_fwarp_prepare<T> (K, TPB); \
      if (sm == 0) break;      /* buffer exceeds the device's shared memory */ \
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&BL, K, TPB, sm); \
      int g=BL*nsm; if ((long)g*TPB > n) g=(n+TPB-1)/TPB; \
      TG=gtime([&]{ K<<<g,TPB,sm>>>(__VA_ARGS__); }, n); }
    case 0: if constexpr ((OPMASK>>0)&1) LAUNCHW(blw,tw,F,kw_mul<PB>, n,dx,dy,dr); break;
    case 1: if constexpr ((OPMASK>>1)&1) LAUNCHW(blw,tw,F,kw_add<PB>, n,dx,dy,dr); break;
    case 2: if constexpr ((OPMASK>>2)&1) LAUNCHW(blw,tw,F,kw_axpy<PB>, n,a,dx,dy,dr); break;
    case 3: if constexpr ((OPMASK>>3)&1) LAUNCHW(blw,tw,C,kw_cmul<PB>, n,(C*)dx,(C*)dy,(C*)dr); break;
    case 4: if constexpr ((OPMASK>>4)&1) LAUNCHW(blw,tw,C,kw_caxpy<PB>, n,ca,(C*)dx,(C*)dy,(C*)dr); break;
    }
    long badw=0;                       /* warp-staged == plain, every element */
    if (tw > 0){
      cudaMemcpy(rw.data(),dr,B,cudaMemcpyDeviceToHost);
      for (long i=0;i<(op>=3?2*n:n);i++) badw += !same<PB>(r[i],rw[i]);
    }
    long bad=0; const long step=  (op>=3? n/2048 : n/4096) + 1;
    for (long i=0;i<n;i+=step){
      if (op<3){ F h = op==0? cu_fmul<PB>(x[i],y[i]) : op==1? cu_fadd<PB>(x[i],y[i]) : cu_fadd<PB>(cu_fmul<PB>(a,x[i]),y[i]);
        bad += !same<PB>(h,r[i]); }
      else { const C *cx=(const C*)x.data(), *cy=(const C*)y.data(); const C *cr=(const C*)r.data();
        C h = op==3? cu_cmul<PB>(cx[i],cy[i]) : cu_cadd<PB>(cu_cmul<PB>(ca,cx[i]),cy[i]);
        bad += !same<PB>(h.re,cr[i].re) + !same<PB>(h.im,cr[i].im); }
    }
    double tc=0, tn=0; if (docpu) bg_cpu_times(PB, M, op, &tc, &tn);
    const double tb = (tw > 0 && tw < tg) ? tw : tg;
    char ws[40];                       /* warp column; n/a when it did not fit */
    if (tw > 0) snprintf (ws, sizeof ws, "%8.3f ns (%4.2fx)", tw, tg/tw);
    else        snprintf (ws, sizeof ws, "     n/a ns (  -  )");
    if (docpu)
      printf("%5d %-6s GPU %8.3f warp %s | CPU-all cu %8.3f  nfloat %8.3f ns | GPU/nf %6.1fx  cu/nf %5.2fx %s\n",
             PB, opn[op], tg, ws, tc, tn, tn/tb, tn/tc, (bad||badw)? "MISMATCH":"exact");
    else
      printf("%5d %-6s GPU %8.3f warp %s  occ %d/%d  %s\n", PB, opn[op], tg, ws, bl, blw,
             (bad||badw)? "MISMATCH":"exact");
    fflush(stdout);
  }
  cudaFree(dx); cudaFree(dy); cudaFree(dr);
  }
}

int main (int argc, char **argv)
{
  bool docpu = !(argc>1 && !strcmp(argv[1],"-g"));
  cudaDeviceProp pr; cudaGetDeviceProperties(&pr,0);
  printf("=== GPU %s (%d SMs) vs %d CPU threads, M=%ld, ns/element (complex: per complex op) ===\n",
         pr.name, pr.multiProcessorCount, docpu ? bg_cpu_threads() : 0, MEL);
  printf("    GPU: plain r[i]=f(x[i],y[i]);  warp: cu_warp_load/cu_warp_store (cu_fwarp.cuh), (x) = GPU/warp\n");
  printf("    GPU/nf = nfloat time on all CPU threads / faster GPU time;  cu/nf = nfloat / cu_freal host, all threads\n");
  int nsm=pr.multiProcessorCount;
  bench<64>(nsm,docpu); bench<128>(nsm,docpu); bench<256>(nsm,docpu); bench<512>(nsm,docpu);
  bench<1024>(nsm,docpu); bench<2048>(nsm,docpu); bench<4096>(nsm,docpu);
  return 0;
}
