/* cu_fcomplex.cuh -- fixed-precision, register-resident complex number for CUDA,
 * built on cu_fp::cu_freal<PB>.  Precision PB (a multiple of 32) is a
 * compile-time template parameter, so both components live in registers.
 * Arithmetic (+, -, *) is bit-exact with MPC's per-component round-to-nearest
 * (MPC_RNDNN).
 *
 *   cu_fp::cu_fcomplex<256> a, b, c;
 *   a = {1.5, -0.5};                 // (re, im) from doubles
 *   c = a * b + a;                   // operators
 *   double re = c.real_d(), im = c.imag_d();
 *
 * Complex multiply uses the correctly-rounded "fmms/fmma" identity
 *   re = round(ar*br - ai*bi),   im = round(ar*bi + ai*br),
 * each computed from the EXACT 2N-limb products (no double rounding), which is
 * exactly MPC's correctly-rounded result for each component.
 */
#ifndef CU_MPC_CUDA_FCOMPLEX_CUH
#define CU_MPC_CUDA_FCOMPLEX_CUH

#include "mpc_cuda/cu_freal.cuh"

namespace cu_fp {

/* ---- normalized wide (2N-limb) product term: value = sign*M*2^(exp-2N*64) ---- */
template<int PB> struct cu_wterm {
  static const int WN = 2*cu_freal<PB>::N;
  int sign; long exp; bool zero; cu_limb M[WN];   /* MSB of M set when !zero */
};

template<int PB> __host__ __device__ static cu_wterm<PB>
cu_mk_term (const cu_freal<PB> &a, const cu_freal<PB> &b)
{
  cu_wterm<PB> t; const int WN = cu_wterm<PB>::WN;
  if (a.is_zero()||b.is_zero()){ t.zero=true; t.sign=1; t.exp=0;
    for(int i=0;i<WN;i++) t.M[i]=0; return t; }
  t.zero=false; t.sign=a.sign*b.sign;
  cu_mul_full<PB> (t.M, a.m, b.m);
  if (t.M[WN-1] & (1ULL<<63)) t.exp = a.exp + b.exp;
  else { for (int i=WN-1;i>0;i--) t.M[i]=(t.M[i]<<1)|(t.M[i-1]>>63);
         t.M[0]<<=1; t.exp = a.exp + b.exp - 1; }
  return t;
}

template<int PB> __host__ __device__ static int
cu_wcmp (const cu_wterm<PB> &a, const cu_wterm<PB> &b)
{
  if (a.exp!=b.exp) return a.exp>b.exp?1:-1;
  for (int i=cu_wterm<PB>::WN-1;i>=0;i--) if (a.M[i]!=b.M[i]) return a.M[i]>b.M[i]?1:-1;
  return 0;
}

/* round(A + B) to PB bits, A,B normalized signed wide terms (exact intermediates) */
template<int PB> __host__ __device__ static cu_freal<PB>
cu_round_sum2 (cu_wterm<PB> A, cu_wterm<PB> B)
{
  typedef cu_freal<PB> F; const int WN=cu_wterm<PB>::WN; const int WB=WN+3; /* 128-bit guard + carry */
  if (A.zero && B.zero){ F r; r.set_zero(); return r; }
  if (A.zero) return F::finalize (B.M, WN, WN*64-1, B.exp-1, 0,0, B.sign);
  if (B.zero) return F::finalize (A.M, WN, WN*64-1, A.exp-1, 0,0, A.sign);
  if (cu_wcmp<PB>(A,B)<0){ cu_wterm<PB> t=A; A=B; B=t; }   /* ensure |A|>=|B| */
  long d = A.exp - B.exp;
  cu_limb AB[WB], BB[WB]; cu_limb sticky=0;
  for (int i=0;i<WB;i++){ AB[i]=0; BB[i]=0; }
  for (int i=0;i<WN;i++) AB[i+2]=A.M[i];                   /* A at limbs[2..WN+1] */
  if (d < (long)WB*64){
    cu_limb T[WB]; for(int i=0;i<WB;i++) T[i]=0;
    for (int i=0;i<WN;i++) T[i+2]=B.M[i];
    long ws=d>>6; int bs=(int)(d&63);
    for (long i=0;i<ws && i<WB;i++) sticky |= (T[i]!=0);
    if (bs && ws<WB) sticky |= (T[ws] & (((cu_limb)1<<bs)-1))!=0;
    for (int i=0;i<WB;i++){ long s=ws+i;
      cu_limb x=(s<WB)?T[s]:0, y=(s+1<WB)?T[s+1]:0; BB[i]= bs?((x>>bs)|(y<<(64-bs))):x; }
  } else sticky=1;
  cu_limb D[WB]; int sgn=A.sign;
  if (A.sign==B.sign){
    cu_limb c=0; for(int i=0;i<WB;i++){ cu_limb s=AB[i]+c; c=(s<c); s+=BB[i]; c+=(s<BB[i]); D[i]=s; }
  } else {
    cu_limb br=0; for(int i=0;i<WB;i++){ cu_limb ai=AB[i],bi=BB[i]; cu_limb s=ai-bi; cu_limb bb=(ai<bi);
      cu_limb s2=s-br; bb|=(s<br); D[i]=s2; br=bb; }
    if (sticky){ cu_limb bb=1; for(int i=0;i<WB && bb;i++){ cu_limb s=D[i]-bb; bb=(D[i]<bb); D[i]=s; } }
  }
  int hl=-1; for(int i=WB-1;i>=0;i--){ if(D[i]){ hl=i; break; } }
  if (hl<0){ F r; r.set_zero(); return r; }
  int top=63; { cu_limb v=D[hl]; while(!((v>>top)&1)) top--; }
  int msb=hl*64+top;
  long Ehi = A.exp - (long)(WN+2)*64 + msb;
  return F::finalize (D, WB, msb, Ehi, 0, sticky, sgn);
}

/* correctly-rounded  a*b - c*d  and  a*b + c*d  (RNDN, PB bits) */
template<int PB> __host__ __device__ static cu_freal<PB>
cu_fmms (const cu_freal<PB>&a,const cu_freal<PB>&b,const cu_freal<PB>&c,const cu_freal<PB>&d)
{ cu_wterm<PB> t1=cu_mk_term<PB>(a,b), t2=cu_mk_term<PB>(c,d); t2.sign=-t2.sign; return cu_round_sum2<PB>(t1,t2); }
template<int PB> __host__ __device__ static cu_freal<PB>
cu_fmma (const cu_freal<PB>&a,const cu_freal<PB>&b,const cu_freal<PB>&c,const cu_freal<PB>&d)
{ cu_wterm<PB> t1=cu_mk_term<PB>(a,b), t2=cu_mk_term<PB>(c,d); return cu_round_sum2<PB>(t1,t2); }

/* ============================ the complex type ============================ */
template<int PB>
struct cu_fcomplex
{
  cu_freal<PB> re, im;

  __host__ __device__ cu_fcomplex () {}
  __host__ __device__ cu_fcomplex (const cu_freal<PB>&r, const cu_freal<PB>&i): re(r), im(i) {}
  __host__ __device__ cu_fcomplex (double r, double i)
    : re(cu_freal<PB>::from_double(r)), im(cu_freal<PB>::from_double(i)) {}

  __host__ __device__ static cu_fcomplex from_doubles (double r, double i)
  { return cu_fcomplex(r,i); }
  __host__ __device__ double real_d () const { return re.to_double(); }
  __host__ __device__ double imag_d () const { return im.to_double(); }
};

template<int PB> __host__ __device__ static inline cu_fcomplex<PB>
cu_cadd (const cu_fcomplex<PB>&a,const cu_fcomplex<PB>&b)
{ return cu_fcomplex<PB>(cu_fadd<PB>(a.re,b.re), cu_fadd<PB>(a.im,b.im)); }
template<int PB> __host__ __device__ static inline cu_fcomplex<PB>
cu_csub (const cu_fcomplex<PB>&a,const cu_fcomplex<PB>&b)
{ return cu_fcomplex<PB>(cu_fsub<PB>(a.re,b.re), cu_fsub<PB>(a.im,b.im)); }
template<int PB> __host__ __device__ static inline cu_fcomplex<PB>
cu_cmul (const cu_fcomplex<PB>&a,const cu_fcomplex<PB>&b)
{ return cu_fcomplex<PB>(cu_fmms<PB>(a.re,b.re,a.im,b.im),     /* ar*br - ai*bi */
                         cu_fmma<PB>(a.re,b.im,a.im,b.re)); }  /* ar*bi + ai*br */

template<int PB> __host__ __device__ static inline cu_fcomplex<PB>
operator+ (const cu_fcomplex<PB>&a,const cu_fcomplex<PB>&b){ return cu_cadd<PB>(a,b); }
template<int PB> __host__ __device__ static inline cu_fcomplex<PB>
operator- (const cu_fcomplex<PB>&a,const cu_fcomplex<PB>&b){ return cu_csub<PB>(a,b); }
template<int PB> __host__ __device__ static inline cu_fcomplex<PB>
operator* (const cu_fcomplex<PB>&a,const cu_fcomplex<PB>&b){ return cu_cmul<PB>(a,b); }

} /* namespace cu_fp */
#endif /* CU_MPC_CUDA_FCOMPLEX_CUH */
