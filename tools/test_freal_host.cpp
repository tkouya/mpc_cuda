/* test_freal_host.cpp -- validate the HOST path of cu_freal<PB> (mul/add/sub)
 * and cu_fcomplex<PB> (mul/add/sub) bit-exact against system MPFR / MPC
 * (RNDN / MPC_RNDNN).  Plain g++, no nvcc.  Besides uniformly random operands
 * it drives the rounding corner cases that random data almost never hits:
 * all-ones / power-of-two / sparse mantissas (ties, carry-out on rounding),
 * exponent gaps around 0, 1, 64 and P, and near-total cancellation.
 *
 * Also checks the fused operations of cu_ffused.cuh: cu_ffma_cr vs mpfr_fma,
 * cu_ffma vs mpfr_mul+mpfr_add, cu_fdot vs mpfr_dot, cu_cfma_cr vs mpc_fma,
 * cu_cfma vs mpc_mul+mpc_add and cu_cdot vs mpc_dot.
 *
 *   g++ -O2 -Iinclude tools/test_freal_host.cpp -o build/test_freal_host \
 *       -lmpc -lmpfr -lgmp
 */
#include <cstdio>
#include <cstdlib>
#include "mpc_cuda/cu_freal.cuh"
#include "mpc_cuda/cu_fcomplex.cuh"
#include "mpc_cuda/cu_ffused.cuh"
using namespace cu_fp;

#include <gmp.h>
#include <mpfr.h>
#include <mpc.h>

#include "fixed_testgen.h"   /* xs, shape_mant, randf, randpair, tiepair */

template<int PB>
static void to_mpfr (mpfr_t out, const cu_freal<PB> &x)
{
  const int N=cu_freal<PB>::N;
  if (x.is_zero()){ mpfr_set_zero(out,1); return; }
  mpz_t M; mpz_init(M);
  mpz_import (M, N, -1, sizeof(cu_limb), 0, 0, x.m);
  mpfr_set_z (out, M, MPFR_RNDN);
  mpfr_mul_2si (out, out, x.exp - N*64, MPFR_RNDN);
  if (x.sign<0) mpfr_neg(out,out,MPFR_RNDN);
  mpz_clear(M);
}

/* representation sanity: normalized, low SB bits clear */
template<int PB>
static bool well_formed (const cu_freal<PB> &x)
{
  const int N=cu_freal<PB>::N, SB=cu_freal<PB>::SB;
  if (x.is_zero()) return true;
  if (!(x.m[N-1]>>63)) return false;
  if (SB && (x.m[0] & (((cu_limb)1<<SB)-1))) return false;
  return x.sign==1 || x.sign==-1;
}

template<int PB>
static int run_real (int M, cu_limb seed)
{
  typedef cu_freal<PB> F;
  mpfr_t ma,mb,ref,mine; mpfr_inits2(PB,ma,mb,ref,mine,(mpfr_ptr)0);
  long bm=0,ba=0,bs=0;
  for (int i=0;i<M;i++){
    F a,b;
    if ((i%8)==7) tiepair<PB>(&seed,a,b); else randpair<PB>(&seed,a,b);
    if (xs(&seed)&1){ F t=a; a=b; b=t; }
    if ((i%64)==21) b.set_zero();
    to_mpfr<PB>(ma,a); to_mpfr<PB>(mb,b);
    F mu=cu_fmul<PB>(a,b), ad=cu_fadd<PB>(a,b), su=cu_fsub<PB>(a,b);
    mpfr_mul(ref,ma,mb,MPFR_RNDN); to_mpfr<PB>(mine,mu); if(!mpfr_equal_p(ref,mine)||!well_formed(mu)) bm++;
    mpfr_add(ref,ma,mb,MPFR_RNDN); to_mpfr<PB>(mine,ad); if(!mpfr_equal_p(ref,mine)||!well_formed(ad)) ba++;
    mpfr_sub(ref,ma,mb,MPFR_RNDN); to_mpfr<PB>(mine,su); if(!mpfr_equal_p(ref,mine)||!well_formed(su)) bs++;
  }
  printf("real    PB=%4d (N=%2d SB=%2d): mul %ld  add %ld  sub %ld  / %d   %s\n",
         PB,F::N,F::SB,bm,ba,bs,M,(bm||ba||bs)?"FAIL":"OK");
  mpfr_clears(ma,mb,ref,mine,(mpfr_ptr)0);
  return (bm||ba||bs)?1:0;
}

template<int PB>
static int run_cplx (int M, cu_limb seed)
{
  typedef cu_freal<PB> F; typedef cu_fcomplex<PB> C;
  mpc_t ca,cb,cr; mpc_init2(ca,PB); mpc_init2(cb,PB); mpc_init2(cr,PB);
  mpfr_t mine; mpfr_init2(mine,PB);
  long bm=0,ba=0,bs=0;
  for (int i=0;i<M;i++){
    F ar,ai,br,bi;
    if ((i%8)==3){ tiepair<PB>(&seed,ar,br); tiepair<PB>(&seed,ai,bi); }
    else { randpair<PB>(&seed,ar,br); randpair<PB>(&seed,ai,bi); }
    if ((i&7)==0){ ai=ar; bi=br; if (xs(&seed)&1) bi.sign=-bi.sign; }   /* re/im cancel */
    if ((i%16)==11) ai.set_zero();                       /* pure real a (tie pair) */
    if ((i%16)==3){ br.set_zero(); }                     /* pure imaginary b (tie pair) */
    if ((i%32)==13){ ai.set_zero(); bi.set_zero(); }     /* both real        */
    C a(ar,ai), b(br,bi);
    to_mpfr<PB>(mpc_realref(ca),ar); to_mpfr<PB>(mpc_imagref(ca),ai);
    to_mpfr<PB>(mpc_realref(cb),br); to_mpfr<PB>(mpc_imagref(cb),bi);
    C mu=cu_cmul<PB>(a,b), ad=cu_cadd<PB>(a,b), su=cu_csub<PB>(a,b);
#define CU_CHK(OP,RES,CNT) \
    OP(cr,ca,cb,MPC_RNDNN); \
    to_mpfr<PB>(mine,RES.re); bool ok=mpfr_equal_p(mpc_realref(cr),mine); \
    to_mpfr<PB>(mine,RES.im); ok=ok&&mpfr_equal_p(mpc_imagref(cr),mine); \
    if(!ok||!well_formed(RES.re)||!well_formed(RES.im)) CNT++;
    { CU_CHK(mpc_mul,mu,bm) }
    { CU_CHK(mpc_add,ad,ba) }
    { CU_CHK(mpc_sub,su,bs) }
#undef CU_CHK
  }
  printf("complex PB=%4d (N=%2d SB=%2d): mul %ld  add %ld  sub %ld  / %d   %s\n",
         PB,F::N,F::SB,bm,ba,bs,M,(bm||ba||bs)?"FAIL":"OK");
  mpc_clear(ca); mpc_clear(cb); mpc_clear(cr); mpfr_clear(mine);
  return (bm||ba||bs)?1:0;
}

/* fused ops: fma and dot (real and complex), incl. cancellation-heavy cases */
template<int PB>
static int run_fused (int M, cu_limb seed)
{
  typedef cu_freal<PB> F; typedef cu_fcomplex<PB> C;
  long bf=0, bd=0, bcf=0, bcd=0;
  mpfr_t ma,mb,mc,ref,mine; mpfr_inits2(PB,ma,mb,mc,ref,mine,(mpfr_ptr)0);
  for (int i=0;i<M;i++){                                   /* fma */
    F a,b,c;
    if ((i%8)==7) tiepair<PB>(&seed,a,b); else randpair<PB>(&seed,a,b);
    long e=a.exp+b.exp; int mode=(int)(xs(&seed)%6);
    c=randf<PB>(&seed, e + (long)(xs(&seed)%9) - 4);
    if (mode==0){ c=cu_fmul<PB>(a,b); c.sign=-c.sign;            /* a*b - RN(a*b): tiny */
                  if ((i&1) && !c.is_zero()) c.m[0]^=(cu_limb)1<<cu_freal<PB>::SB; }
    if (mode==1) c=randf<PB>(&seed, e - (long)(xs(&seed)%(3*PB)));
    if (mode==2) c=randf<PB>(&seed, e + (long)(xs(&seed)%(3*PB)));
    if ((i%32)==9) c.set_zero();
    if ((i%32)==17) b.set_zero();
    to_mpfr<PB>(ma,a); to_mpfr<PB>(mb,b); to_mpfr<PB>(mc,c);
    mpfr_fma(ref,ma,mb,mc,MPFR_RNDN);
    F r=cu_ffma_cr<PB>(a,b,c); to_mpfr<PB>(mine,r);
    if (!mpfr_equal_p(ref,mine) || !well_formed(r)) bf++;
    mpfr_mul(ref,ma,mb,MPFR_RNDN); mpfr_add(ref,ref,mc,MPFR_RNDN);    /* two roundings */
    r=cu_ffma<PB>(a,b,c); to_mpfr<PB>(mine,r);
    if (!mpfr_equal_p(ref,mine) || !well_formed(r)) bf++;
  }
  const int MAXN=24;
  F x[MAXN], y[MAXN]; mpfr_t mx[MAXN], my[MAXN]; mpfr_ptr px[MAXN], py[MAXN];
  for (int k=0;k<MAXN;k++){ mpfr_init2(mx[k],PB); mpfr_init2(my[k],PB); px[k]=mx[k]; py[k]=my[k]; }
  for (int i=0;i<M/4;i++){                                  /* dot */
    int n=1+(int)(xs(&seed)%MAXN), mode=(int)(xs(&seed)%5);
    for (int k=0;k<n;k++){
      randpair<PB>(&seed,x[k],y[k]);
      if (mode==1) { x[k].exp=(long)(xs(&seed)%400)-200; }              /* wide exponent spread */
      if (mode==2 && k>0 && (k&1)){ x[k]=x[k-1]; y[k]=y[k-1]; y[k].sign=-y[k].sign;   /* exact cancellation pairs */
                                    if (xs(&seed)&1) y[k].m[0]^=(cu_limb)1<<(cu_freal<PB>::SB+(xs(&seed)%8)); }
      if (mode==3 && (xs(&seed)%4)==0) x[k].set_zero();
      if (mode==4){ tiepair<PB>(&seed,x[k],y[k]); }
    }
    for (int k=0;k<n;k++){ to_mpfr<PB>(mx[k],x[k]); to_mpfr<PB>(my[k],y[k]); }
    mpfr_dot(ref,px,py,(unsigned long)n,MPFR_RNDN);
    F r=cu_fdot<PB>(x,y,n); to_mpfr<PB>(mine,r);
    if (!mpfr_equal_p(ref,mine) || !well_formed(r)) bd++;
  }
  mpc_t qa,qb,qc,qr; mpc_init2(qa,PB); mpc_init2(qb,PB); mpc_init2(qc,PB); mpc_init2(qr,PB);
  for (int i=0;i<M/4;i++){                                  /* complex fma */
    F ar,ai,br,bi,cr,ci;
    randpair<PB>(&seed,ar,br); randpair<PB>(&seed,ai,bi);
    cr=randf<PB>(&seed, ar.exp+br.exp+(long)(xs(&seed)%5)-2); ci=randf<PB>(&seed, ar.exp+bi.exp+(long)(xs(&seed)%5)-2);
    if ((i%4)==1){ C t=cu_cmul<PB>(C(ar,ai),C(br,bi)); cr=t.re; ci=t.im; cr.sign=-cr.sign; ci.sign=-ci.sign; }
    if ((i%16)==5) ai.set_zero();
    C a(ar,ai), b(br,bi), c(cr,ci);
    to_mpfr<PB>(mpc_realref(qa),ar); to_mpfr<PB>(mpc_imagref(qa),ai);
    to_mpfr<PB>(mpc_realref(qb),br); to_mpfr<PB>(mpc_imagref(qb),bi);
    to_mpfr<PB>(mpc_realref(qc),cr); to_mpfr<PB>(mpc_imagref(qc),ci);
    mpc_fma(qr,qa,qb,qc,MPC_RNDNN);
    C r=cu_cfma_cr<PB>(a,b,c);
    to_mpfr<PB>(mine,r.re); bool ok=mpfr_equal_p(mpc_realref(qr),mine);
    to_mpfr<PB>(mine,r.im); ok=ok&&mpfr_equal_p(mpc_imagref(qr),mine);
    if (!ok) bcf++;
    mpc_mul(qr,qa,qb,MPC_RNDNN); mpc_add(qr,qr,qc,MPC_RNDNN);          /* two roundings */
    r=cu_cfma<PB>(a,b,c);
    to_mpfr<PB>(mine,r.re); ok=mpfr_equal_p(mpc_realref(qr),mine);
    to_mpfr<PB>(mine,r.im); ok=ok&&mpfr_equal_p(mpc_imagref(qr),mine);
    if (!ok) bcf++;
  }
  C cx[MAXN], cy[MAXN]; mpc_t qx[MAXN], qy[MAXN]; mpc_ptr pqx[MAXN], pqy[MAXN];
  for (int k=0;k<MAXN;k++){ mpc_init2(qx[k],PB); mpc_init2(qy[k],PB); pqx[k]=qx[k]; pqy[k]=qy[k]; }
  for (int i=0;i<M/8;i++){                                  /* complex dot */
    int n=1+(int)(xs(&seed)%MAXN);
    for (int k=0;k<n;k++){
      F ar,ai,br,bi; randpair<PB>(&seed,ar,br); randpair<PB>(&seed,ai,bi);
      if ((i%4)==2 && k>0 && (k&1)){ ar=cx[k-1].re; ai=cx[k-1].im; br=cy[k-1].re; bi=cy[k-1].im; br.sign=-br.sign; bi.sign=-bi.sign; }
      cx[k]=C(ar,ai); cy[k]=C(br,bi);
      to_mpfr<PB>(mpc_realref(qx[k]),ar); to_mpfr<PB>(mpc_imagref(qx[k]),ai);
      to_mpfr<PB>(mpc_realref(qy[k]),br); to_mpfr<PB>(mpc_imagref(qy[k]),bi);
    }
    mpc_dot(qr,pqx,pqy,(unsigned long)n,MPC_RNDNN);
    C r=cu_cdot<PB>(cx,cy,n);
    to_mpfr<PB>(mine,r.re); bool ok=mpfr_equal_p(mpc_realref(qr),mine);
    to_mpfr<PB>(mine,r.im); ok=ok&&mpfr_equal_p(mpc_imagref(qr),mine);
    if (!ok) bcd++;
  }
  for (int k=0;k<MAXN;k++){ mpfr_clear(mx[k]); mpfr_clear(my[k]); mpc_clear(qx[k]); mpc_clear(qy[k]); }
  mpc_clear(qa); mpc_clear(qb); mpc_clear(qc); mpc_clear(qr);
  mpfr_clears(ma,mb,mc,ref,mine,(mpfr_ptr)0);
  printf("fused   PB=%4d (N=%2d SB=%2d): fma %ld/%d  dot %ld/%d  cfma %ld/%d  cdot %ld/%d   %s\n",
         PB,F::N,F::SB,bf,2*M,bd,M/4,bcf,M/2,bcd,M/8,(bf||bd||bcf||bcd)?"FAIL":"OK");
  return (bf||bd||bcf||bcd)?1:0;
}

/* the host mulhigh drops columns < K=N-3; check 0 <= P - T*2^(64K) < (K+1)*2^(64(K+1)) */
template<int PB>
static int run_mulhigh_bound (int M, cu_limb seed)
{
  const int N=cu_freal<PB>::N;
  if constexpr (N<4) return 0;
  else {
    const int K=N-3;
    long bad=0;
    mpz_t P,T,D,lim; mpz_inits(P,T,D,lim,(mpz_ptr)0);
    mpz_set_ui(lim,K+1); mpz_mul_2exp(lim,lim,64*(K+1));
    for (int i=0;i<M;i++){
      cu_freal<PB> a,b;
      if (i%4==0){ shape_mant<PB>(a.m,&seed,1); shape_mant<PB>(b.m,&seed,(int)(xs(&seed)%8)); }
      else { shape_mant<PB>(a.m,&seed,(int)(xs(&seed)%8)); shape_mant<PB>(b.m,&seed,(int)(xs(&seed)%8)); }
      cu_limb P2[2*N], TT[N+3];
      cu_mul_cols<N,0>(P2,a.m,b.m);
      cu_mul_cols<N,N-3>(TT,a.m,b.m);
      mpz_import(P,2*N,-1,sizeof(cu_limb),0,0,P2);
      mpz_import(T,N+3,-1,sizeof(cu_limb),0,0,TT); mpz_mul_2exp(T,T,64*K);
      mpz_sub(D,P,T);
      if (mpz_sgn(D)<0 || mpz_cmp(D,lim)>=0) bad++;
      /* the full product itself must match GMP */
      mpz_t A,B; mpz_inits(A,B,(mpz_ptr)0);
      mpz_import(A,N,-1,sizeof(cu_limb),0,0,a.m); mpz_import(B,N,-1,sizeof(cu_limb),0,0,b.m);
      mpz_mul(A,A,B); if (mpz_cmp(A,P)) bad++;
      mpz_clears(A,B,(mpz_ptr)0);
    }
    mpz_clears(P,T,D,lim,(mpz_ptr)0);
    printf("mulhigh PB=%4d (N=%2d K=%2d): bound/product violations %ld / %d   %s\n",
           PB,N,K,bad,M,bad?"FAIL":"OK");
    return bad?1:0;
  }
}

template<int PB>
static int run (int M, cu_limb seed)
{ return run_real<PB>(M,seed) | run_cplx<PB>(M/4,seed^0x5a5aULL)
         | run_mulhigh_bound<PB>(M/10,seed^0xa5a5ULL) | run_fused<PB>(M/5,seed^0x3c3cULL); }

int main (int argc, char **argv)
{
  int M = argc>1 ? atoi(argv[1]) : 100000;
  int bad=0;
  printf("=== host cu_freal/cu_fcomplex<PB> vs system MPFR/MPC (RNDN), %d cases ===\n",M);
  bad|=run<32>  (M,0x1111ull);
  bad|=run<64>  (M,0x2222ull);
  bad|=run<96>  (M,0x3333ull);
  bad|=run<128> (M,0x4444ull);
  bad|=run<160> (M,0x5555ull);
  bad|=run<192> (M,0x5656ull);
  bad|=run<256> (M,0x6666ull);
  bad|=run<288> (M,0x7777ull);
  bad|=run<512> (M,0x8888ull);
  bad|=run<1024>(M,0x9999ull);
  bad|=run<1056>(M,0xaaaaull);
  bad|=run<2048>(M/4,0xbbbbull);
  bad|=run<4096>(M/16,0xccccull);
  printf("%s\n", bad?"*** FAILURES ***":"ALL PRECISIONS BIT-EXACT (host)");
  return bad;
}
