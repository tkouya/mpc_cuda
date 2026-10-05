/* bench_gpu_fixed.h -- shared by tools/bench_gpu_fixed.cu (GPU, nvcc) and
 * tools/bench_gpu_fixed_cpu.cpp (CPU: cu_freal host path and FLINT nfloat on
 * all cores, OpenMP; plain C++, so FLINT's headers never meet nvcc). */
#ifndef CU_BENCH_GPU_FIXED_H
#define CU_BENCH_GPU_FIXED_H
#include "mpc_cuda/cu_freal.cuh"
#include "mpc_cuda/cu_fcomplex.cuh"
using namespace cu_fp;

static inline cu_limb bg_xs (cu_limb *s){ cu_limb x=*s; x^=x<<13; x^=x>>7; x^=x<<17; return *s=x; }

/* full random PB-bit mantissa, random sign, exponent in [-2,2] */
template<int PB> static cu_freal<PB> bg_rand (cu_limb *s)
{
  cu_freal<PB> r; const int N=cu_freal<PB>::N, SB=cu_freal<PB>::SB;
  r.sign=(bg_xs(s)&1)?1:-1;
  for (int i=0;i<N;i++) r.m[i]=bg_xs(s);
  r.m[N-1] |= 1ULL<<63;
  if (SB) r.m[0] &= ~(((cu_limb)1<<SB)-1);
  r.exp=(long)(bg_xs(s)%5)-2;
  return r;
}

/* ns per element on all CPU threads for op 0 mul, 1 add, 2 axpy (r = a*x+y),
 * 3 cmul, 4 caxpy (per complex element): t_cu = cu_freal host path,
 * t_nf = FLINT nfloat / nfloat_complex (not correctly rounded) */
extern "C" void bg_cpu_times (int pb, long m, int op, double *t_cu, double *t_nf);
extern "C" int  bg_cpu_threads (void);
#endif
