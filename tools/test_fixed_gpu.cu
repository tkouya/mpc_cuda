/* test_fixed_gpu.cu -- the DEVICE path of cu_freal<PB> / cu_fcomplex<PB>
 * (mul, add, sub; complex mul, add) checked bit-exact against the HOST path on
 * the rounding corner cases of tools/fixed_testgen.h (shaped mantissas, exact
 * ties, near-total cancellation, exponent gaps 0..P+70, zero operands).  The
 * host path itself is checked bit-exact against MPFR/MPC by test_freal_host
 * (make check-fixed-host); tools/test_freal.cu / test_fcomplex.cu compare the
 * device with MPFR/MPC directly on random operands.  Also checks the
 * warp-cooperative loads/stores of cu_fwarp.cuh (partial last tiles, 8- and
 * 16-byte words, static and dynamic buffers, nothing written past n).
 * No MPFR needed here.
 *
 *   nvcc -arch=sm_90 -O3 -std=c++17 -Iinclude tools/test_fixed_gpu.cu -o build/test_fixed_gpu
 */
#include <cstdio>
#include <cstring>
#include <vector>
#include "mpc_cuda/cu_freal.cuh"
#include "mpc_cuda/cu_fcomplex.cuh"
#include "mpc_cuda/cu_fwarp.cuh"
using namespace cu_fp;
#include "fixed_testgen.h"

template<int PB> __global__ void kr (int n, const cu_freal<PB>*a, const cu_freal<PB>*b,
                                     cu_freal<PB>*m, cu_freal<PB>*s, cu_freal<PB>*d)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<n){ m[i]=cu_fmul<PB>(a[i],b[i]); s[i]=cu_fadd<PB>(a[i],b[i]); d[i]=cu_fsub<PB>(a[i],b[i]); }
}
template<int PB> __global__ void kc (int n, const cu_fcomplex<PB>*a, const cu_fcomplex<PB>*b,
                                     cu_fcomplex<PB>*m, cu_fcomplex<PB>*s)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<n){ m[i]=cu_cmul<PB>(a[i],b[i]); s[i]=cu_cadd<PB>(a[i],b[i]); }
}

template<int PB> static bool eq (const cu_freal<PB>&x, const cu_freal<PB>&y)
{
  if (x.is_zero()||y.is_zero()) return x.is_zero()&&y.is_zero();
  return x.sign==y.sign && x.exp==y.exp && !memcmp(x.m,y.m,sizeof x.m);
}

template<int PB> static int run (int M)
{
  typedef cu_freal<PB> F; typedef cu_fcomplex<PB> C;
  cu_limb s=0xabcdef12345ULL^PB;
  std::vector<F> A(M), B(M), Rm(M), Rs(M), Rd(M);
  for (int i=0;i<M;i++){
    const int k=i%8;
    if (k<5)      randpair<PB>(&s,A[i],B[i]);
    else if (k<7) tiepair<PB>(&s,A[i],B[i]);
    else { randpair<PB>(&s,A[i],B[i]); if (xs(&s)&1) A[i].set_zero(); else B[i].set_zero(); }
  }
  F *dA,*dB,*dm,*ds,*dd; const size_t z=M*sizeof(F);
  cudaMalloc(&dA,z); cudaMalloc(&dB,z); cudaMalloc(&dm,z); cudaMalloc(&ds,z); cudaMalloc(&dd,z);
  cudaMemcpy(dA,A.data(),z,cudaMemcpyHostToDevice); cudaMemcpy(dB,B.data(),z,cudaMemcpyHostToDevice);
  kr<PB><<<(M+127)/128,128>>>(M,dA,dB,dm,ds,dd);
  cudaMemcpy(Rm.data(),dm,z,cudaMemcpyDeviceToHost);
  cudaMemcpy(Rs.data(),ds,z,cudaMemcpyDeviceToHost);
  cudaMemcpy(Rd.data(),dd,z,cudaMemcpyDeviceToHost);
  long bm=0, bs=0, bd=0;
  for (int i=0;i<M;i++){
    bm += !eq<PB>(Rm[i],cu_fmul<PB>(A[i],B[i]));
    bs += !eq<PB>(Rs[i],cu_fadd<PB>(A[i],B[i]));
    bd += !eq<PB>(Rd[i],cu_fsub<PB>(A[i],B[i]));
  }
  /* complex: consecutive reals paired up, so every generator kind meets every other */
  const int MC=M/2;
  const C *cA=(const C*)A.data(), *cB=(const C*)B.data();
  std::vector<C> Cm(MC), Cs(MC);
  kc<PB><<<(MC+127)/128,128>>>(MC,(C*)dA,(C*)dB,(C*)dm,(C*)ds);
  const cudaError_t e=cudaDeviceSynchronize();
  cudaMemcpy(Cm.data(),dm,MC*sizeof(C),cudaMemcpyDeviceToHost);
  cudaMemcpy(Cs.data(),ds,MC*sizeof(C),cudaMemcpyDeviceToHost);
  long cm=0, cs=0;
  for (int i=0;i<MC;i++){
    const C h=cu_cmul<PB>(cA[i],cB[i]), g=cu_cadd<PB>(cA[i],cB[i]);
    cm += !eq<PB>(Cm[i].re,h.re) + !eq<PB>(Cm[i].im,h.im);
    cs += !eq<PB>(Cs[i].re,g.re) + !eq<PB>(Cs[i].im,g.im);
  }
  const bool ok = !(bm|bs|bd|cm|cs) && e==cudaSuccess;
  printf("PB=%4d  mul %ld add %ld sub %ld  cmul %ld cadd %ld  / %d   %s%s\n",
         PB,bm,bs,bd,cm,cs,M, ok?"OK":"FAIL", e==cudaSuccess?"":cudaGetErrorString(e));
  cudaFree(dA); cudaFree(dB); cudaFree(dm); cudaFree(ds); cudaFree(dd);
  return !ok;
}

/* ---- cu_fwarp: r[i] = op(x[i], y[i]) through cu_warp_load / cu_warp_store ---- */
template<class T> __device__ static T wop (const T &a, const T &b);
template<int PB> __device__ cu_freal<PB> wop (const cu_freal<PB> &a, const cu_freal<PB> &b)
{ return cu_fadd<PB> (cu_fmul<PB> (a, b), a); }
template<int PB> __device__ cu_fcomplex<PB> wop (const cu_fcomplex<PB> &a, const cu_fcomplex<PB> &b)
{ return cu_cadd<PB> (cu_cmul<PB> (a, b), a); }
template<class T> static T hop (const T &a, const T &b);
template<int PB> cu_freal<PB> hop (const cu_freal<PB> &a, const cu_freal<PB> &b)
{ return cu_fadd<PB> (cu_fmul<PB> (a, b), a); }
template<int PB> cu_fcomplex<PB> hop (const cu_fcomplex<PB> &a, const cu_fcomplex<PB> &b)
{ return cu_cadd<PB> (cu_cmul<PB> (a, b), a); }

template<class T, bool DYN> __global__ void kw (long n, const T *x, const T *y, T *r)
{
  T *wb;
  if constexpr (DYN) wb = cu_fwarp_dyn<T> ();
  else { __shared__ cu_fwarp_buf<T, 2> buf; wb = buf.warp (); }          /* 64 threads */
  for (long base = cu_warp_first_tile (); base < n; base += cu_warp_tile_stride ())
    {
      const T a = cu_warp_load (x, base, n, wb), b = cu_warp_load (y, base, n, wb);
      cu_warp_store (r, base, n, wop (a, b), wb);
    }
}

template<class T> static bool eqT (const T &a, const T &b);
template<int PB> bool eqT (const cu_freal<PB> &a, const cu_freal<PB> &b) { return eq<PB> (a, b); }
template<int PB> bool eqT (const cu_fcomplex<PB> &a, const cu_fcomplex<PB> &b)
{ return eq<PB> (a.re, b.re) && eq<PB> (a.im, b.im); }

template<class T, bool DYN, class MK> static int run_warp (const char *name, MK mk)
{
  static const long ns[] = {1, 31, 32, 33, 95, 1000, 100003};
  const long NMAX = 100003, PAD = 40;
  std::vector<T> x(NMAX), y(NMAX), r(NMAX+PAD), sent(PAD);
  cu_limb s=0x5eed^sizeof(T);
  for (long i=0;i<NMAX;i++){ x[i]=mk(&s); y[i]=mk(&s); }
  for (long i=0;i<PAD;i++) sent[i]=mk(&s);
  T *dx,*dy,*dr;
  cudaMalloc(&dx,NMAX*sizeof(T)); cudaMalloc(&dy,NMAX*sizeof(T)); cudaMalloc(&dr,(NMAX+PAD)*sizeof(T));
  cudaMemcpy(dx,x.data(),NMAX*sizeof(T),cudaMemcpyHostToDevice);
  cudaMemcpy(dy,y.data(),NMAX*sizeof(T),cudaMemcpyHostToDevice);
  const size_t sm = DYN ? cu_fwarp_prepare<T> (kw<T,DYN>, 64) : 0;
  long bad=0;
  for (long n : ns){
    for (long i=0;i<PAD;i++) r[n+i]=sent[i];
    cudaMemcpy(dr+n,r.data()+n,PAD*sizeof(T),cudaMemcpyHostToDevice);    /* sentinels after n */
    kw<T,DYN><<<7,64,sm>>>(n,dx,dy,dr);                                    /* small grid: many tiles per warp */
    if (cudaDeviceSynchronize()!=cudaSuccess){ printf("  %s: %s\n",name,cudaGetErrorString(cudaGetLastError())); return 1; }
    cudaMemcpy(r.data(),dr,(n+PAD)*sizeof(T),cudaMemcpyDeviceToHost);
    for (long i=0;i<n;i++) bad += !eqT (r[i], hop (x[i], y[i]));
    for (long i=0;i<PAD;i++) bad += !eqT (r[n+i], sent[i]);                /* untouched */
  }
  printf("cu_fwarp %-22s %s buffer, %2d-byte words: %ld mismatches   %s\n", name, DYN?"dynamic":"static ",
         detail::cu_fwarp_word<T>::W16 ? 16 : 8, bad, bad?"FAIL":"OK");
  cudaFree(dx); cudaFree(dy); cudaFree(dr);
  return bad!=0;
}
template<int PB> static cu_freal<PB> mkr (cu_limb *s){ cu_freal<PB> a,b; randpair<PB>(s,a,b); return a; }
template<int PB> static cu_fcomplex<PB> mkc (cu_limb *s){ cu_freal<PB> a,b; randpair<PB>(s,a,b); return cu_fcomplex<PB>(a,b); }

int main (int argc, char **argv)
{
  const int M = argc>1 ? atoi(argv[1]) : 200000;
  printf("=== device cu_freal/cu_fcomplex<PB> vs host path, corner-case operands ===\n");
  int f=0;
  f|=run<64>(M);  f|=run<96>(M);  f|=run<128>(M); f|=run<160>(M); f|=run<192>(M);
  f|=run<256>(M); f|=run<288>(M); f|=run<384>(M); f|=run<512>(M); f|=run<544>(M);
  f|=run<768>(M); f|=run<1024>(M); f|=run<1056>(M); f|=run<2048>(M); f|=run<4096>(M/4);
  f|=run_warp<cu_freal<64>,false>("cu_freal<64>", mkr<64>);
  f|=run_warp<cu_freal<128>,false>("cu_freal<128>", mkr<128>);
  f|=run_warp<cu_freal<1024>,false>("cu_freal<1024>", mkr<1024>);
  f|=run_warp<cu_freal<1056>,true>("cu_freal<1056>", mkr<1056>);
  f|=run_warp<cu_fcomplex<192>,false>("cu_fcomplex<192>", mkc<192>);
  f|=run_warp<cu_fcomplex<2048>,true>("cu_fcomplex<2048>", mkc<2048>);
  f|=run_warp<cu_freal<4096>,true>("cu_freal<4096>", mkr<4096>);
  printf(f ? "DEVICE/HOST MISMATCHES\n" : "ALL PRECISIONS BIT-EXACT (device == host)\n");
  return f;
}
