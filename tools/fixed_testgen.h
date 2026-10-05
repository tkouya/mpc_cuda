/* fixed_testgen.h -- operand generators for the cu_freal<PB> / cu_fcomplex<PB>
 * tests (host and GPU): random and "shaped" mantissas (all ones, powers of
 * two, sparse -> ties and carry-out on rounding), exponent gaps around 0, 1,
 * 64 and P, near-total cancellation, and products that are exact RNDN ties.
 * Shared by tools/test_freal_host.cpp and tools/test_fixed_gpu.cu.
 * Include after mpc_cuda/cu_freal.cuh with `using namespace cu_fp;`.        */
#ifndef CU_FIXED_TESTGEN_H
#define CU_FIXED_TESTGEN_H

static cu_limb xs (cu_limb *s){ cu_limb x=*s; x^=x<<13; x^=x>>7; x^=x<<17; return *s=x; }

/* mantissa of a given "shape"; MSB set, low SB bits zero */
template<int PB>
static void shape_mant (cu_limb *m, cu_limb *s, int kind)
{
  const int N=cu_freal<PB>::N, SB=cu_freal<PB>::SB;
  switch (kind)
    {
    case 1: for (int i=0;i<N;i++) m[i]=~0ULL; break;                 /* all ones   */
    case 2: for (int i=0;i<N;i++) m[i]=0; m[N-1]=xs(s); break;        /* top limb   */
    case 3: for (int i=0;i<N;i++) m[i]=0; m[0]=(cu_limb)1<<SB; break; /* 1+ulp      */
    case 5: for (int i=0;i<N;i++) m[i]=0; break;                      /* power of 2 */
    case 4: for (int i=0;i<N;i++) m[i]=0; m[N-1]=xs(s); m[0]=xs(s); break;
    case 6: {                                       /* random, low k bits cleared */
      for (int i=0;i<N;i++) m[i]=xs(s);
      int k=(int)(xs(s)%(N*64));
      for (int b=0;b<k;b++) m[b>>6]&=~((cu_limb)1<<(b&63));
      break; }
    default: for (int i=0;i<N;i++) m[i]=xs(s);
    }
  m[N-1] |= 1ULL<<63;
  if (SB) m[0] &= ~(((cu_limb)1<<SB)-1);
}

template<int PB>
static cu_freal<PB> randf (cu_limb *s, long e)
{
  cu_freal<PB> r;
  r.sign=(xs(s)&1)?1:-1;
  shape_mant<PB> (r.m, s, (int)(xs(s)%8));
  r.exp=e;
  return r;
}

/* operand pair with an engineered exponent gap / cancellation */
template<int PB>
static void randpair (cu_limb *s, cu_freal<PB> &a, cu_freal<PB> &b)
{
  const int N=cu_freal<PB>::N, SB=cu_freal<PB>::SB;
  long ea=(long)(xs(s)%41)-20;
  a=randf<PB>(s,ea);
  int pick=(int)(xs(s)%16);
  if (pick<4){ b=randf<PB>(s,(long)(xs(s)%41)-20); return; }
  if (pick<6){                                       /* near-cancellation */
    b=a; if (xs(s)&1) b.sign=-b.sign;
    int k=(int)(xs(s)%(N*64-SB)); cu_limb add=(cu_limb)1<<((k+SB)&63);
    int w=(k+SB)>>6;
    if (xs(s)&1){ for (int i=w;i<N&&add;i++){ cu_limb t=b.m[i]+add; add=(t<add); b.m[i]=t; } }
    else        { for (int i=w;i<N&&add;i++){ cu_limb t=b.m[i]-add; add=(b.m[i]<add); b.m[i]=t; } }
    if (!(b.m[N-1]>>63)){ b=a; b.exp--; }            /* keep normalized */
    return;
  }
  static const long gaps[]={0,1,2,3,62,63,64,65,66};
  long d;
  if (pick<10) d=gaps[xs(s)%9];
  else if (pick<14){ long base=PB+(long)(xs(s)%6)-3; d=base<0?0:base; }
  else d=(long)(xs(s)%(PB+70));
  b=randf<PB>(s,ea-d);
}

/* odd integer of exactly k bits, left-justified into a PB-bit mantissa */
template<int PB>
static void odd_kbits (cu_freal<PB> &r, cu_limb *s, int k)
{
  const int N=cu_freal<PB>::N;
  cu_limb v[N]; for (int i=0;i<N;i++) v[i]=0;
  for (int b=0;b<k;b++) if ((b==0)||(b==k-1)||(xs(s)&1)) v[b>>6]|=(cu_limb)1<<(b&63);
  int sh=N*64-k;                                  /* left-justify */
  int ws=sh>>6, bs=sh&63;
  for (int i=N-1;i>=0;i--){ cu_limb hi=(i-ws>=0)?v[i-ws]:0, lo=(i-ws-1>=0)?v[i-ws-1]:0;
    r.m[i]= bs? ((hi<<bs)|(lo>>(64-bs))) : hi; }
  r.sign=(xs(s)&1)?1:-1; r.exp=(long)(xs(s)%9)-4;
}

/* a*b is an exact RNDN tie (or one bit shorter) for P-bit odd operands */
template<int PB>
static void tiepair (cu_limb *s, cu_freal<PB> &a, cu_freal<PB> &b)
{
  int ka = 1 + (int)(xs(s)%PB);
  int kb = PB + 1 + (int)(xs(s)&1) - ka;
  if (kb<1) kb=1; if (kb>PB) kb=PB;
  odd_kbits<PB>(a,s,ka); odd_kbits<PB>(b,s,kb);
}

#endif /* CU_FIXED_TESTGEN_H */
