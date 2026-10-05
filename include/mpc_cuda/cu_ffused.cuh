/* cu_ffused.cuh -- fused, correctly rounded operations on the fixed-precision
 * types (host only):
 *
 *   cu_ffma<PB>(a, b, c)              RNDN(RNDN(a*b) + c)      fast: cu_fmul + cu_fadd
 *   cu_ffma_cr<PB>(a, b, c)           RNDN(a*b + c)            (= mpfr_fma)
 *   cu_fdot<PB>(x, y, n)              RNDN(sum x[i]*y[i])      (= mpfr_dot)
 *   cu_cfma<PB>(a, b, c)              a*b + c via cu_cmul + cu_cadd (two roundings)
 *   cu_cfma_cr<PB>(a, b, c)           per-part RNDN(a*b + c)   (= mpc_fma, MPC_RNDNN)
 *   cu_cdot<PB>(x, y, n)              per-part RNDN(sum x*y)   (= mpc_dot, MPC_RNDNN)
 *
 * The _cr variants and the dot products round once per result, unlike a
 * loop of cu_fmul/cu_fadd.  (A single rounding needs a wider intermediate, so
 * cu_ffma_cr is 10-60% slower than cu_ffma; use it when the single rounding
 * matters.)  Fast path:
 * truncated products (cu_mul_cols with the low columns dropped) summed in a
 * fixed-point accumulator, with an explicit bound on the truncation error and
 * a rounding test (cu_round_fixed / cu_sum_round<..,true>).  When the test
 * cannot certify the result (it lies within the error of a rounding boundary,
 * or massive cancellation), the exact products are summed exactly instead.
 * The exact fallback of the dot products uses heap memory proportional to the
 * exponent span of the products.
 */
#ifndef CU_MPC_CUDA_FFUSED_CUH
#define CU_MPC_CUDA_FFUSED_CUH

#include "mpc_cuda/cu_fcomplex.cuh"

#if !defined(__CUDA_ARCH__)
#include <vector>

namespace cu_fp {

/* ---------------------------------------------------------------- fma */

/* exact: the 2N-limb product plus c, rounded once (the only path for N < 4,
 * where the product is short; otherwise a cold fallback) */
template<int PB> static inline cu_freal<PB>
cu_ffma_exact (const cu_freal<PB> &a, const cu_freal<PB> &b, const cu_freal<PB> &c)
{
  const int N=cu_freal<PB>::N, WN=cu_wterm<PB>::WN;
  cu_wterm<PB> t1 = cu_mk_term<PB> (a,b), t2;
  t2.zero=false; t2.sign=c.sign; t2.exp=c.exp;
  for (int i=0;i<N;i++){ t2.M[i]=0; t2.M[WN-N+i]=c.m[i]; }
  return cu_round_sum2<PB> (t1, t2);
}

template<int PB> CU_FP_COLD static cu_freal<PB>
cu_ffma_exact_cold (const cu_freal<PB> &a, const cu_freal<PB> &b, const cu_freal<PB> &c)
{ return cu_ffma_exact<PB> (a, b, c); }

template<int PB> static inline cu_freal<PB>
cu_ffma_cr (const cu_freal<PB> &a, const cu_freal<PB> &b, const cu_freal<PB> &c)
{
  typedef cu_freal<PB> F; const int N=F::N;
  if (a.is_zero() || b.is_zero()) return c;
  if (c.is_zero()) return cu_fmul<PB> (a,b);
  if constexpr (N >= 4)
    {
      const int WT=cu_tterm<PB>::WT;
      cu_tterm<PB> P; cu_mk_tterm<PB> (P, a, b);
      cu_tterm<PB> Q; Q.sign=c.sign; Q.exp=c.exp; Q.exact=true;
      for (int i=0;i<WT-N;i++) Q.M[i]=0;
      for (int i=0;i<N;i++) Q.M[WT-N+i]=c.m[i];
      const cu_limb eu = P.exact ? 0 : (cu_limb)(4*N);
      const bool sub = (P.sign!=Q.sign);
      bool ok = true;
      F r = (cu_tcmp<PB>(P,Q) >= 0)
          ? cu_sum_round<PB,WT,true> (P.M, P.exp, Q.M, Q.exp, sub, P.sign, eu, &ok)
          : cu_sum_round<PB,WT,true> (Q.M, Q.exp, P.M, P.exp, sub, Q.sign, eu, &ok);
      if (__builtin_expect (ok, 1)) return r;
    }
  if constexpr (N >= 4) return cu_ffma_exact_cold<PB> (a, b, c);
  else                  return cu_ffma_exact<PB> (a, b, c);
}

/* ---------------------------------------------------------------- dot */

/* Exact: every product exactly (2N limbs), accumulated in two's complement
 * over the full exponent span of the products, rounded once. */
template<int PB, class GET> CU_FP_COLD static cu_freal<PB>
cu_dot_exact (long n, GET get)
{
  typedef cu_freal<PB> F; const int N=F::N;
  long emax=0, emin=0; bool any=false;
  for (long i=0;i<n;i++){
    const F *a,*b; int s; get (i,a,b,s);
    if (a->is_zero() || b->is_zero()) continue;
    const long te = a->exp + b->exp;
    if (!any){ emax=emin=te; any=true; }
    else { if (te>emax) emax=te; if (te<emin) emin=te; }
  }
  if (!any){ F r; r.set_zero(); return r; }
  /* bit 0 of the accumulator has weight 2^(emin-128N); one limb of headroom
   * for growth and the sign */
  const long span = emax - emin + 128L*N;
  const long len  = span/64 + 3;
  std::vector<cu_limb> acc (len, 0);
  cu_limb P[2*N];
  for (long i=0;i<n;i++){
    const F *a,*b; int s; get (i,a,b,s);
    if (a->is_zero() || b->is_zero()) continue;
    cu_mul_full<PB> (P, a->m, b->m);
    s *= a->sign*b->sign;
    const long off = a->exp + b->exp - emin;          /* bit offset of P */
    const long q = off>>6; const int rb = (int)(off&63);
    cu_limb c = (s<0) ? 1 : 0;                         /* add (+P) or (~P + 1) */
    const cu_limb inv = (s<0) ? ~(cu_limb)0 : 0;
    for (long k=0;k<len-q;k++){
      cu_limb lo = (k<2*N) ? P[k] : 0, hm = (k>=1 && k-1<2*N) ? P[k-1] : 0;
      cu_limb w = rb ? ((lo<<rb) | (hm>>(64-rb))) : lo;
      if (k>=2*N+1) w=0;
      acc[q+k] = cu_addc (acc[q+k], w^inv, c);
    }
  }
  int sgn = 1;
  if (acc[len-1]>>63){                                 /* negative: negate */
    sgn=-1; cu_limb c=1;
    for (long k=0;k<len;k++) acc[k]=cu_addc (~acc[k], 0, c);
  }
  long hl=len-1; while (hl>=0 && !acc[hl]) hl--;
  if (hl<0){ F r; r.set_zero(); return r; }
  const int msb = (int)(hl*64 + 63 - cu_clz64 (acc[hl]));
  return F::finalize (acc.data(), (int)len, msb, (long)msb + emin - 128L*N, 0, 0, sgn);
}

/* Fast: truncated products in a fixed-point two's-complement accumulator.
 * Products keep the columns >= K = N-4 (one more than cu_fmul, so that the
 * truncation error ends a full limb below the guard word G of the result
 * even when the leading product's top bit is clear); for N <= 4, K = 0 and
 * the products are exact.  Layout (limbs): [2 low guard limbs | LT limbs
 * where a top-exponent product lands | 1 headroom/sign limb], LA = LT+3, bit
 * 0 weight 2^(Emax-64(LA-1)).  Each term's truncation error is < (K+1) units
 * of its limb 1, which sits at accumulator limb 3 or lower (2^192 units); the
 * bits shifted out below bit 0 add < 1 unit per term.  So |error| <
 * (n(K+1)+1) * 2^192 units, or < n units when K = 0.                      */
template<int PB, class GET> static cu_freal<PB>
cu_dot_core (long n, GET get)
{
  typedef cu_freal<PB> F; const int N=F::N;
  const int K = (N>=5) ? N-4 : 0, LT = 2*N-K, LA = LT+3;
  long emax=0, nnz=0;
  for (long i=0;i<n;i++){
    const F *a,*b; int s; get (i,a,b,s);
    if (a->is_zero() || b->is_zero()) continue;
    const long te = a->exp + b->exp;
    if (!nnz || te>emax) emax=te;
    nnz++;
  }
  if (!nnz){ F r; r.set_zero(); return r; }
  cu_limb acc[LA]; for (int k=0;k<LA;k++) acc[k]=0;
  cu_limb T[LT], W[LA];
  for (long i=0;i<n;i++){
    const F *a,*b; int s; get (i,a,b,s);
    if (a->is_zero() || b->is_zero()) continue;
    const long sh = emax - (a->exp + b->exp);
    const long q = sh>>6; const int rb = (int)(sh&63);
    if (q >= LT+2) continue;                           /* < 1 unit: only error */
    cu_mul_cols<N,K> (T, a->m, b->m);
    if (__builtin_expect (q == 0, 1))
      {                     /* the usual case: W = T*2^(128-rb), no branch on rb
                               ((x<<1)<<(63-rb) is x<<(64-rb), or 0 for rb = 0) */
        W[0] = 0;
        W[1] = (T[0]<<1)<<(63-rb);
        for (int t=1;t<LT;t++) W[t+1] = (T[t-1]>>rb) | ((T[t]<<1)<<(63-rb));
        W[LT+1] = T[LT-1]>>rb;
        W[LT+2] = 0;
      }
    else
      {
        for (int k=0;k<LA;k++) W[k]=0;
        for (int t=0;t<LT;t++){
          const long j = t + 2 - q;
          if (j>=0) W[j] |= rb ? (T[t]>>rb) : T[t];
          if (rb && j-1>=0) W[j-1] |= T[t]<<(64-rb);
        }
      }
    /* acc += +-W in two's complement, branch-free (random signs would
     * mispredict an add/sub branch half of the time) */
    const cu_limb neg = (cu_limb)0 - (cu_limb)(s*a->sign*b->sign < 0);
    for (int k=0;k<LA;k++) W[k]^=neg;
    cu_add_n<LA> (acc, acc, W, neg&1);
  }
  int sgn = 1;
  if (acc[LA-1]>>63){
    sgn=-1; cu_limb c=1;
    for (int k=0;k<LA;k++) acc[k]=cu_addc (~acc[k], 0, c);
  }
  F r;
  const cu_limb ecount = K ? (cu_limb)nnz*(cu_limb)(K+1) + 1 : (cu_limb)nnz;
  const long    epos   = K ? 192 : 0;
  if (__builtin_expect (cu_round_fixed<PB,LA> (r, acc, sgn, emax - 64L*(LA-1), ecount, epos), 1))
    return r;
  return cu_dot_exact<PB> (n, get);
}

/* RNDN(sum_{i<n} x[i]*y[i]) */
template<int PB> static inline cu_freal<PB>
cu_fdot (const cu_freal<PB> *x, const cu_freal<PB> *y, long n)
{
  return cu_dot_core<PB> (n, [&](long i, const cu_freal<PB> *&a, const cu_freal<PB> *&b, int &s)
                              { a=&x[i]; b=&y[i]; s=1; });
}

/* fast a*b + c with two roundings (cu_fmul then cu_fadd) */
template<int PB> static inline cu_freal<PB>
cu_ffma (const cu_freal<PB> &a, const cu_freal<PB> &b, const cu_freal<PB> &c)
{ return cu_fadd<PB> (cu_fmul<PB> (a,b), c); }

/* ------------------------------------------------------- complex fma / dot */

/* per part: re = RNDN(ar*br - ai*bi + cr), im = RNDN(ar*bi + ai*br + ci) */
template<int PB> static inline cu_fcomplex<PB>
cu_cfma_cr (const cu_fcomplex<PB> &a, const cu_fcomplex<PB> &b, const cu_fcomplex<PB> &c)
{
  typedef cu_freal<PB> F;
  static const F one = F::from_double (1.0);
  const F *re[3][2] = {{&a.re,&b.re},{&a.im,&b.im},{&c.re,&one}};
  const F *im[3][2] = {{&a.re,&b.im},{&a.im,&b.re},{&c.im,&one}};
  cu_fcomplex<PB> r;
  r.re = cu_dot_core<PB> (3, [&](long i, const F *&p, const F *&q, int &s)
                             { p=re[i][0]; q=re[i][1]; s=(i==1)?-1:1; });
  r.im = cu_dot_core<PB> (3, [&](long i, const F *&p, const F *&q, int &s)
                             { p=im[i][0]; q=im[i][1]; s=1; });
  return r;
}

/* fast a*b + c with two roundings per part (cu_cmul then cu_cadd) */
template<int PB> static inline cu_fcomplex<PB>
cu_cfma (const cu_fcomplex<PB> &a, const cu_fcomplex<PB> &b, const cu_fcomplex<PB> &c)
{ return cu_cadd<PB> (cu_cmul<PB> (a,b), c); }

/* per part RNDN of sum x[i]*y[i] */
template<int PB> static inline cu_fcomplex<PB>
cu_cdot (const cu_fcomplex<PB> *x, const cu_fcomplex<PB> *y, long n)
{
  typedef cu_freal<PB> F;
  cu_fcomplex<PB> r;
  r.re = cu_dot_core<PB> (2*n, [&](long i, const F *&p, const F *&q, int &s)
           { const long k=i>>1; if (i&1){ p=&x[k].im; q=&y[k].im; s=-1; }
                                 else   { p=&x[k].re; q=&y[k].re; s= 1; } });
  r.im = cu_dot_core<PB> (2*n, [&](long i, const F *&p, const F *&q, int &s)
           { const long k=i>>1; if (i&1){ p=&x[k].im; q=&y[k].re; }
                                 else   { p=&x[k].re; q=&y[k].im; } s=1; });
  return r;
}

} /* namespace cu_fp */
#endif /* !__CUDA_ARCH__ */
#endif /* CU_MPC_CUDA_FFUSED_CUH */
