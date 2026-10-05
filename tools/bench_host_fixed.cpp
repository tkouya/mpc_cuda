/* bench_host_fixed.cpp -- single-core HOST throughput of the fixed-precision
 * cu_freal<PB> / cu_fcomplex<PB> against FLINT nfloat / nfloat_complex and
 * system MPFR / MPC, per operation, over PB = 64 .. 4096 bits.
 *
 *   real   : mul, add (random signs), axpy (y += a*x), dot (sum x*y),
 *            faxpy (y = cu_ffma_cr(a,x,y), one rounding), fdot (correctly rounded dot)
 *   complex: mul, axpy, dot, fcdot (correctly rounded complex dot)
 * (cu_ffma itself is cu_fmul + cu_fadd, i.e. the axpy row.)
 * For faxpy/fdot/fcdot the MPFR/MPC column is mpfr_fma / mpfr_dot / mpc_dot
 * (also correctly rounded); nfloat stays _nfloat_vec_addmul_scalar /
 * _nfloat_vec_dot / _nfloat_complex_vec_dot (not correctly rounded).
 *
 * All operands carry full random PB-bit mantissas (exponents in [-2,2]).
 * Times are ns per element (best of 5 timed passes, the three libraries
 * interleaved pass by pass); "x nf" is
 * nfloat_time / cu_time (>1: cu_freal is faster).  cu_freal and MPFR/MPC are
 * correctly rounded per operation; nfloat is not (1-2 ulp), and its dot
 * products accumulate with a single final rounding, so "dot" compares cu's
 * mul+add loop with nfloat's fused _nfloat_vec_dot.
 *
 * Pin to a big core for stable numbers, e.g. on GB10 (X925 = CPUs 5-9,15-19):
 *   taskset -c 5 ./build/bench_host_fixed
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include "mpc_cuda/cu_freal.cuh"
#include "mpc_cuda/cu_fcomplex.cuh"
#include "mpc_cuda/cu_ffused.cuh"
using namespace cu_fp;

#include <gmp.h>
#include <mpfr.h>
#include <mpc.h>
#include <flint/flint.h>
#include <flint/arf.h>
#include <flint/gr.h>
#include <flint/nfloat.h>

#ifndef BENCH_M
#define BENCH_M 65536                    /* elements per vector: large enough that the
                                            branch predictor cannot memorize the data */
#endif
#ifndef BENCH_MAXBITS
#define BENCH_MAXBITS 4096               /* skip precisions above this */
#endif
#ifndef BENCH_SEC
#define BENCH_SEC 0.05                   /* minimum seconds per timed pass */
#endif

static const int M = BENCH_M;

static cu_limb xs (cu_limb *s){ cu_limb x=*s; x^=x<<13; x^=x>>7; x^=x<<17; return *s=x; }

template<int PB>
static cu_freal<PB> randf (cu_limb *s)
{
  cu_freal<PB> r; const int N=cu_freal<PB>::N, SB=cu_freal<PB>::SB;
  r.sign=(xs(s)&1)?1:-1;
  for (int i=0;i<N;i++) r.m[i]=xs(s);
  r.m[N-1] |= 1ULL<<63;
  if (SB) r.m[0] &= ~(((cu_limb)1<<SB)-1);
  r.exp=(long)(xs(s)%5)-2;
  return r;
}

template<int PB>
static void to_mpfr (mpfr_t out, const cu_freal<PB> &x)
{
  const int N=cu_freal<PB>::N;
  if (x.is_zero()){ mpfr_set_zero(out,1); return; }
  mpz_t Z; mpz_init(Z);
  mpz_import (Z, N, -1, sizeof(cu_limb), 0, 0, x.m);
  mpfr_set_z (out, Z, MPFR_RNDN);
  mpfr_mul_2si (out, out, x.exp - N*64, MPFR_RNDN);
  if (x.sign<0) mpfr_neg(out,out,MPFR_RNDN);
  mpz_clear(Z);
}

static void mpfr_to_nfloat (nfloat_ptr out, const mpfr_t x, gr_ctx_t ctx)
{
  arf_t t; arf_init(t); arf_set_mpfr(t,x);
  if (nfloat_set_arf(out,t,ctx)!=GR_SUCCESS){ fprintf(stderr,"nfloat_set_arf failed\n"); exit(1); }
  arf_clear(t);
}

static volatile double g_sink;

/* ns per element of one timed pass (>= BENCH_SEC/2); f() processes M elements */
template<class Fn>
static double pass_ns (Fn &f)
{
  using clk=std::chrono::steady_clock;
  long reps=0; double el=0; auto t0=clk::now();
  do { f(); reps++; el=std::chrono::duration<double>(clk::now()-t0).count(); }
  while (el<BENCH_SEC/2);
  return el*1e9/((double)reps*M);
}
/* the three contenders timed in interleaved rounds, best of 5 each, so clock
 * drift and similar slow effects hit all of them alike */
template<class A, class B, class C>
static void time3 (A fa, B fb, C fc, double &ta, double &tb, double &tc)
{
  fa(); fb(); fc();                                 /* warm-up */
  ta=tb=tc=1e300;
  for (int round=0;round<5;round++){
    double x=pass_ns(fa); if (x<ta) ta=x;
    x=pass_ns(fb); if (x<tb) tb=x;
    x=pass_ns(fc); if (x<tc) tc=x;
  }
}

static void row (int pb, const char *op, double cu, double nf, double mp)
{
  printf("%5d  %-6s %10.1f %10.1f %10.1f   %6.2fx %6.2fx\n",
         pb, op, cu, nf, mp, nf/cu, mp/cu);
}

template<int PB>
static void bench ()
{
  if (PB > BENCH_MAXBITS) return;
  typedef cu_freal<PB> F; typedef cu_fcomplex<PB> C;
  cu_limb seed=0x9e3779b97f4a7c15ULL ^ (cu_limb)PB;

  /* ---------------- data ---------------- */
  F *x=new F[M], *y=new F[M], *r=new F[M]; F a=randf<PB>(&seed);
  C *cx=new C[M], *cy=new C[M], *cr=new C[M]; C ca(randf<PB>(&seed),randf<PB>(&seed));
  for (int i=0;i<M;i++){ x[i]=randf<PB>(&seed); y[i]=randf<PB>(&seed);
    cx[i]=C(randf<PB>(&seed),randf<PB>(&seed)); cy[i]=C(randf<PB>(&seed),randf<PB>(&seed)); }

  mpfr_t *mx=new mpfr_t[M], *my=new mpfr_t[M], *mr=new mpfr_t[M], ma, macc, mt;
  mpc_t *qx=new mpc_t[M], *qy=new mpc_t[M], *qr=new mpc_t[M], qa, qacc, qt;
  for (int i=0;i<M;i++){
    mpfr_init2(mx[i],PB); mpfr_init2(my[i],PB); mpfr_init2(mr[i],PB);
    to_mpfr<PB>(mx[i],x[i]); to_mpfr<PB>(my[i],y[i]);
    mpc_init2(qx[i],PB); mpc_init2(qy[i],PB); mpc_init2(qr[i],PB);
    to_mpfr<PB>(mpc_realref(qx[i]),cx[i].re); to_mpfr<PB>(mpc_imagref(qx[i]),cx[i].im);
    to_mpfr<PB>(mpc_realref(qy[i]),cy[i].re); to_mpfr<PB>(mpc_imagref(qy[i]),cy[i].im);
  }
  mpfr_inits2(PB,ma,macc,mt,(mpfr_ptr)0); to_mpfr<PB>(ma,a);
  mpc_init2(qa,PB); mpc_init2(qacc,PB); mpc_init2(qt,PB);
  to_mpfr<PB>(mpc_realref(qa),ca.re); to_mpfr<PB>(mpc_imagref(qa),ca.im);

  gr_ctx_t nctx, cctx;
  nfloat_ctx_init(nctx,PB,0); nfloat_complex_ctx_init(cctx,PB,0);
  const slong ns=NFLOAT_CTX_DATA_NLIMBS(nctx), cs=NFLOAT_COMPLEX_CTX_DATA_NLIMBS(cctx);
  ulong *nx=new ulong[M*ns], *ny=new ulong[M*ns], *nr=new ulong[M*ns], *na=new ulong[ns], *nacc=new ulong[ns];
  ulong *kx=new ulong[M*cs], *ky=new ulong[M*cs], *kr=new ulong[M*cs], *ka=new ulong[cs], *kacc=new ulong[cs], *kt=new ulong[cs];
  for (int i=0;i<M;i++){
    mpfr_to_nfloat(nx+i*ns,mx[i],nctx); mpfr_to_nfloat(ny+i*ns,my[i],nctx);
    mpfr_to_nfloat(NFLOAT_COMPLEX_RE(kx+i*cs,cctx),mpc_realref(qx[i]),nctx);
    mpfr_to_nfloat(NFLOAT_COMPLEX_IM(kx+i*cs,cctx),mpc_imagref(qx[i]),nctx);
    mpfr_to_nfloat(NFLOAT_COMPLEX_RE(ky+i*cs,cctx),mpc_realref(qy[i]),nctx);
    mpfr_to_nfloat(NFLOAT_COMPLEX_IM(ky+i*cs,cctx),mpc_imagref(qy[i]),nctx);
  }
  mpfr_to_nfloat(na,ma,nctx);
  mpfr_to_nfloat(NFLOAT_COMPLEX_RE(ka,cctx),mpc_realref(qa),nctx);
  mpfr_to_nfloat(NFLOAT_COMPLEX_IM(ka,cctx),mpc_imagref(qa),nctx);

  /* axpy updates y in place (timed after dot); values only drift by
   * O(reps*|a x|), harmless for timing. */
  F *yy=new F[M]; memcpy(yy,y,sizeof(F)*M);
  C *cyy=new C[M]; memcpy(cyy,cy,sizeof(C)*M);

  double t_cu, t_nf, t_mp;

  /* ---- real mul ---- */
  time3 ([&]{ for(int i=0;i<M;i++) r[i]=cu_fmul<PB>(x[i],y[i]); },
        [&]{ for(int i=0;i<M;i++) nfloat_mul(nr+i*ns,nx+i*ns,ny+i*ns,nctx); },
        [&]{ for(int i=0;i<M;i++) mpfr_mul(mr[i],mx[i],my[i],MPFR_RNDN); }, t_cu, t_nf, t_mp);
  row(PB,"mul",t_cu,t_nf,t_mp);
  /* ---- real add (mixed signs -> add and sub) ---- */
  time3 ([&]{ for(int i=0;i<M;i++) r[i]=cu_fadd<PB>(x[i],y[i]); },
        [&]{ for(int i=0;i<M;i++) nfloat_add(nr+i*ns,nx+i*ns,ny+i*ns,nctx); },
        [&]{ for(int i=0;i<M;i++) mpfr_add(mr[i],mx[i],my[i],MPFR_RNDN); }, t_cu, t_nf, t_mp);
  row(PB,"add",t_cu,t_nf,t_mp);
  /* ---- real dot ---- */
  time3 ([&]{ F s; s.set_zero(); for(int i=0;i<M;i++) s=cu_fadd<PB>(s,cu_fmul<PB>(x[i],y[i])); g_sink=s.to_double(); },
        [&]{ _nfloat_vec_dot(nacc,NULL,0,nx,ny,M,nctx); },
        [&]{ mpfr_set_zero(macc,1); for(int i=0;i<M;i++){ mpfr_mul(mt,mx[i],my[i],MPFR_RNDN); mpfr_add(macc,macc,mt,MPFR_RNDN);} }, t_cu, t_nf, t_mp);
  row(PB,"dot",t_cu,t_nf,t_mp);
  /* ---- real axpy y += a*x ---- */
  time3 ([&]{ for(int i=0;i<M;i++) yy[i]=cu_fadd<PB>(yy[i],cu_fmul<PB>(a,x[i])); },
        [&]{ _nfloat_vec_addmul_scalar(ny,nx,M,na,nctx); },
        [&]{ for(int i=0;i<M;i++){ mpfr_mul(mt,ma,mx[i],MPFR_RNDN); mpfr_add(my[i],my[i],mt,MPFR_RNDN);} }, t_cu, t_nf, t_mp);
  row(PB,"axpy",t_cu,t_nf,t_mp);
  /* ---- fused: y = fma(a,x,y) and the correctly rounded dot ---- */
  time3 ([&]{ for(int i=0;i<M;i++) yy[i]=cu_ffma_cr<PB>(a,x[i],yy[i]); },
        [&]{ _nfloat_vec_addmul_scalar(ny,nx,M,na,nctx); },
        [&]{ for(int i=0;i<M;i++) mpfr_fma(my[i],ma,mx[i],my[i],MPFR_RNDN); }, t_cu, t_nf, t_mp);
  row(PB,"faxpy",t_cu,t_nf,t_mp);
  {
    mpfr_ptr *pmx=new mpfr_ptr[M], *pmy=new mpfr_ptr[M];
    for (int i=0;i<M;i++){ pmx[i]=mx[i]; pmy[i]=my[i]; }
    time3 ([&]{ g_sink=cu_fdot<PB>(x,y,M).to_double(); },
        [&]{ _nfloat_vec_dot(nacc,NULL,0,nx,ny,M,nctx); },
        [&]{ mpfr_dot(macc,pmx,pmy,(unsigned long)M,MPFR_RNDN); }, t_cu, t_nf, t_mp);
    row(PB,"fdot",t_cu,t_nf,t_mp);
    delete[] pmx; delete[] pmy;
  }

  /* ---- complex mul ---- */
  time3 ([&]{ for(int i=0;i<M;i++) cr[i]=cu_cmul<PB>(cx[i],cy[i]); },
        [&]{ for(int i=0;i<M;i++) nfloat_complex_mul(kr+i*cs,kx+i*cs,ky+i*cs,cctx); },
        [&]{ for(int i=0;i<M;i++) mpc_mul(qr[i],qx[i],qy[i],MPC_RNDNN); }, t_cu, t_nf, t_mp);
  row(PB,"cmul",t_cu,t_nf,t_mp);
  /* ---- complex dot ---- */
  time3 ([&]{ C s(0.0,0.0); for(int i=0;i<M;i++) s=cu_cadd<PB>(s,cu_cmul<PB>(cx[i],cy[i])); g_sink=s.real_d(); },
        [&]{ _nfloat_complex_vec_dot(kacc,NULL,0,kx,ky,M,cctx); },
        [&]{ mpc_set_ui(qacc,0,MPC_RNDNN); for(int i=0;i<M;i++){ mpc_mul(qt,qx[i],qy[i],MPC_RNDNN); mpc_add(qacc,qacc,qt,MPC_RNDNN);} }, t_cu, t_nf, t_mp);
  row(PB,"cdot",t_cu,t_nf,t_mp);
  /* ---- complex axpy y += a*x ---- */
  time3 ([&]{ for(int i=0;i<M;i++) cyy[i]=cu_cadd<PB>(cyy[i],cu_cmul<PB>(ca,cx[i])); },
        [&]{ for(int i=0;i<M;i++){ nfloat_complex_mul(kt,ka,kx+i*cs,cctx); nfloat_complex_add(ky+i*cs,ky+i*cs,kt,cctx);} },
        [&]{ for(int i=0;i<M;i++){ mpc_mul(qt,qa,qx[i],MPC_RNDNN); mpc_add(qy[i],qy[i],qt,MPC_RNDNN);} }, t_cu, t_nf, t_mp);
  row(PB,"caxpy",t_cu,t_nf,t_mp);
  {
    mpc_ptr *pqx=new mpc_ptr[M], *pqy=new mpc_ptr[M];
    for (int i=0;i<M;i++){ pqx[i]=qx[i]; pqy[i]=qy[i]; }
    time3 ([&]{ g_sink=cu_cdot<PB>(cx,cy,M).real_d(); },
        [&]{ _nfloat_complex_vec_dot(kacc,NULL,0,kx,ky,M,cctx); },
        [&]{ mpc_dot(qacc,pqx,pqy,(unsigned long)M,MPC_RNDNN); }, t_cu, t_nf, t_mp);
    row(PB,"fcdot",t_cu,t_nf,t_mp);
    delete[] pqx; delete[] pqy;
  }
  fflush(stdout);

  /* ---------------- cleanup ---------------- */
  for (int i=0;i<M;i++){ mpfr_clear(mx[i]); mpfr_clear(my[i]); mpfr_clear(mr[i]);
    mpc_clear(qx[i]); mpc_clear(qy[i]); mpc_clear(qr[i]); }
  mpfr_clears(ma,macc,mt,(mpfr_ptr)0); mpc_clear(qa); mpc_clear(qacc); mpc_clear(qt);
  gr_ctx_clear(nctx); gr_ctx_clear(cctx);
  delete[] x; delete[] y; delete[] r; delete[] yy; delete[] cx; delete[] cy; delete[] cr; delete[] cyy;
  delete[] mx; delete[] my; delete[] mr; delete[] qx; delete[] qy; delete[] qr;
  delete[] nx; delete[] ny; delete[] nr; delete[] na; delete[] nacc;
  delete[] kx; delete[] ky; delete[] kr; delete[] ka; delete[] kacc; delete[] kt;
}

int main (void)
{
  { /* bring the core to its sustained clock before the first (64-bit) rows */
    using clk=std::chrono::steady_clock; volatile cu_limb w=1; auto t0=clk::now();
    while (std::chrono::duration<double>(clk::now()-t0).count() < 0.5) for (int i=0;i<100000;i++) w=w*3+1; }
  printf("=== host fixed-precision throughput, 1 core, M=%d, ns/element ===\n", M);
  printf("    FLINT %s, MPFR %s, MPC %s\n", FLINT_VERSION, mpfr_get_version(), mpc_get_version());
  printf("%5s  %-6s %10s %10s %10s   %7s %7s\n","bits","op","cu_freal","nfloat","MPFR/MPC","x nf","x mp");
  bench<64>();   bench<128>();  bench<192>();  bench<256>();
  bench<384>();  bench<512>();  bench<768>();  bench<1024>();
  bench<1536>(); bench<2048>(); bench<3072>(); bench<4096>();
  return 0;
}
