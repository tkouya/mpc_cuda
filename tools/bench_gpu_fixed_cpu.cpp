/* bench_gpu_fixed_cpu.cpp -- CPU half of tools/bench_gpu_fixed.cu: the same
 * element-wise operations with the cu_freal/cu_fcomplex host path and with
 * FLINT nfloat / nfloat_complex, on all OpenMP threads (best of 3 passes of
 * >= 0.2 s).  Build with -O3 -march=native -fopenmp; needs FLINT >= 3.1.   */
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>
#include "bench_gpu_fixed.h"
#include <gmp.h>
#include <flint/flint.h>
#include <flint/arf.h>
#include <flint/gr.h>
#include <flint/nfloat.h>

template<class Fn> static double tbest (Fn f, long M)
{
  f(); double best=1e300;
  for (int p=0;p<3;p++){ long reps=0; double t0=omp_get_wtime(), el;
    do { f(); reps++; el=omp_get_wtime()-t0; } while (el<0.2);
    double ns=el*1e9/((double)reps*M); if (ns<best) best=ns; }
  return best;
}

static void to_nf (nfloat_ptr out, const cu_limb *m, int N, long exp, int sign, gr_ctx_t ctx)
{
  mpz_t Z; mpz_init(Z); mpz_import(Z,N,-1,8,0,0,m);
  arf_t t; arf_init(t); arf_set_mpz(t,Z); arf_mul_2exp_si(t,t,exp-64L*N); if (sign<0) arf_neg(t,t);
  nfloat_set_arf(out,t,ctx); arf_clear(t); mpz_clear(Z);
}

template<int PB> static void run (long M, int op, double *tc, double *tn)
{
  typedef cu_freal<PB> F; typedef cu_fcomplex<PB> C; const int N=F::N;
  cu_limb seed=0x12345678ULL^PB;
  std::vector<F> x(M), y(M), r(M); F a=bg_rand<PB>(&seed);
  for (long i=0;i<M;i++){ x[i]=bg_rand<PB>(&seed); y[i]=bg_rand<PB>(&seed); }
  gr_ctx_t ctx; nfloat_ctx_init(ctx,PB,0); gr_ctx_t cctx; nfloat_complex_ctx_init(cctx,PB,0);
  const long ns=NFLOAT_CTX_DATA_NLIMBS(ctx);
  const bool cplx = op>=3;
  const long M2 = cplx ? M/2 : M;                      /* complex: pairs of reals */
  std::vector<ulong> nx(M*ns), ny(M*ns), nr(M*ns), na(2*ns);
  for (long i=0;i<M;i++){ to_nf(nx.data()+i*ns,x[i].m,N,x[i].exp,x[i].sign,ctx); to_nf(ny.data()+i*ns,y[i].m,N,y[i].exp,y[i].sign,ctx); }
  to_nf(na.data(),a.m,N,a.exp,a.sign,ctx); to_nf(na.data()+ns,x[0].m,N,x[0].exp,x[0].sign,ctx);
  C ca(a,x[0]); const C *cx=(const C*)x.data(), *cy=(const C*)y.data(); C *cr=(C*)r.data();
  const long cs=2*ns;  /* nfloat complex = re,im consecutive */
  switch (op){
  case 0:
    *tc=tbest([&]{
#pragma omp parallel for schedule(static)
      for(long i=0;i<M;i++) r[i]=cu_fmul<PB>(x[i],y[i]); },M);
    *tn=tbest([&]{
#pragma omp parallel for schedule(static)
      for(long i=0;i<M;i++) nfloat_mul(nr.data()+i*ns,nx.data()+i*ns,ny.data()+i*ns,ctx); },M);
    break;
  case 1:
    *tc=tbest([&]{
#pragma omp parallel for schedule(static)
      for(long i=0;i<M;i++) r[i]=cu_fadd<PB>(x[i],y[i]); },M);
    *tn=tbest([&]{
#pragma omp parallel for schedule(static)
      for(long i=0;i<M;i++) nfloat_add(nr.data()+i*ns,nx.data()+i*ns,ny.data()+i*ns,ctx); },M);
    break;
  case 2:  /* r = a*x + y (out of place, so repeated passes keep the data) */
    *tc=tbest([&]{
#pragma omp parallel for schedule(static)
      for(long i=0;i<M;i++) r[i]=cu_fadd<PB>(cu_fmul<PB>(a,x[i]),y[i]); },M);
    *tn=tbest([&]{
#pragma omp parallel
      { ulong t[160];
#pragma omp for schedule(static)
      for(long i=0;i<M;i++){ nfloat_mul(t,na.data(),nx.data()+i*ns,ctx); nfloat_add(nr.data()+i*ns,t,ny.data()+i*ns,ctx);} } },M);
    break;
  case 3:
    *tc=tbest([&]{
#pragma omp parallel for schedule(static)
      for(long i=0;i<M2;i++) cr[i]=cu_cmul<PB>(cx[i],cy[i]); },M2);
    *tn=tbest([&]{
#pragma omp parallel for schedule(static)
      for(long i=0;i<M2;i++) nfloat_complex_mul(nr.data()+i*cs,nx.data()+i*cs,ny.data()+i*cs,cctx); },M2);
    break;
  case 4:
    *tc=tbest([&]{
#pragma omp parallel for schedule(static)
      for(long i=0;i<M2;i++) cr[i]=cu_cadd<PB>(cu_cmul<PB>(ca,cx[i]),cy[i]); },M2);
    *tn=tbest([&]{
#pragma omp parallel
      { ulong t[320];
#pragma omp for schedule(static)
      for(long i=0;i<M2;i++){ nfloat_complex_mul(t,na.data(),nx.data()+i*cs,cctx); nfloat_complex_add(nr.data()+i*cs,t,ny.data()+i*cs,cctx);} } },M2);
    break;
  }
  gr_ctx_clear(ctx); gr_ctx_clear(cctx);
}

extern "C" int bg_cpu_threads (void) { return omp_get_max_threads(); }

extern "C" void bg_cpu_times (int pb, long M, int op, double *tc, double *tn)
{
  switch (pb){
  case 64: run<64>(M,op,tc,tn); break;   case 128: run<128>(M,op,tc,tn); break;
  case 256: run<256>(M,op,tc,tn); break; case 512: run<512>(M,op,tc,tn); break;
  case 1024: run<1024>(M,op,tc,tn); break; case 2048: run<2048>(M,op,tc,tn); break;
  case 4096: run<4096>(M,op,tc,tn); break;
  default: *tc=*tn=0;
  }
}
