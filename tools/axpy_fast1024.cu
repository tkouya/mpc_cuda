/* axpy_fast1024.cu -- fixed-precision (1024-bit = 16 limb) register-resident
 * real arithmetic on the GPU, as a fast-path alternative to the runtime-precision
 * MPFR port.  Implements mul / add (RNDN, bit-exact with MPFR) with the
 * significand held in registers (compile-time-constant limb count -> ptxas
 * promotes the local arrays to registers, no global-memory arena round-trips).
 *
 * Validates rounding against the SYSTEM MPFR (1024-bit, RNDN) on random
 * full-width operands, then benchmarks y = a*x + y for double operands against
 * the existing MPFR GPU path (run separately as ./build/axpy_mpfr).
 *
 *   nvcc -arch=sm_121 -O3 -fmad=false -I/usr/local/include -o /tmp/axpy_fast1024 \
 *        tools/axpy_fast1024.cu -L/usr/local/lib -lmpfr -lgmp
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cstdint>

typedef unsigned long long limb;
#define NL   16            /* limbs in a 1024-bit significand           */
#define WL   18            /* wide accumulator for add (1024 + 128 guard)*/
#define ZERO_EXP  (-0x7fffffffL)

struct F1024 { int sign; long exp; limb m[NL]; };   /* m[NL-1] MSB set when nonzero */

__host__ __device__ static inline int f_is_zero (const F1024 &a){ return a.exp==ZERO_EXP; }

/* ---------------- double <-> F1024 ---------------- */
__host__ __device__ static F1024
from_double (double d)
{
  F1024 r;
#pragma unroll
  for (int i=0;i<NL;i++) r.m[i]=0;
  if (d==0.0){ r.sign=1; r.exp=ZERO_EXP; return r; }
  r.sign = d<0?-1:1; d = d<0?-d:d;
  int k; double f = frexp (d,&k);                  /* d = f*2^k, f in [0.5,1) */
  limb F = (limb) ldexp (f,53);                    /* 53-bit integer          */
  r.m[NL-1] = F << 11;                             /* MSB -> bit 63           */
  r.exp = k;
  return r;
}
__host__ __device__ static double
to_double (const F1024 &x)
{
  if (f_is_zero(x)) return 0.0;
  limb F = x.m[NL-1] >> 11;                         /* top 53 bits             */
  limb rbit = (x.m[NL-1] >> 10) & 1;
  limb sticky = (x.m[NL-1] & 0x3FFULL)!=0;
#pragma unroll
  for (int i=0;i<NL-1;i++) sticky |= (x.m[i]!=0);
  long e = x.exp;
  if (rbit && (sticky || (F&1))){ F++; if (F==(1ULL<<53)){ F>>=1; e++; } }
  return (double)x.sign * ldexp ((double)F, e-53);
}

/* ---------------- 64x64->128 (same inline PTX as the library) ---------- */
__host__ __device__ static inline void
umul (limb &hi, limb &lo, limb a, limb b)
{
#if defined(__CUDA_ARCH__)
  asm ("mul.lo.u64 %0,%2,%3;\n\tmul.hi.u64 %1,%2,%3;"
       : "=l"(lo),"=l"(hi) : "l"(a),"l"(b));
#else
  __uint128_t p = (__uint128_t)a*b; lo=(limb)p; hi=(limb)(p>>64);
#endif
}

/* ---------------- 16x16 -> 32 schoolbook, register-resident ------------ */
__host__ __device__ static void
mul_16x16 (limb r[2*NL], const limb a[NL], const limb b[NL])
{
#pragma unroll
  for (int i=0;i<2*NL;i++) r[i]=0;
#pragma unroll
  for (int j=0;j<NL;j++)
    {
      limb cl=0;
#pragma unroll
      for (int i=0;i<NL;i++)
        {
          limb hi,lo,rl=r[i+j];
#if defined(__CUDA_ARCH__)
          asm ("mad.lo.cc.u64 %0,%2,%3,%4;\n\t"
               "madc.hi.u64   %1,%2,%3,0;\n\t"
               "add.cc.u64    %0,%0,%5;\n\t"
               "addc.u64      %1,%1,0;"
               : "=&l"(lo),"=&l"(hi) : "l"(a[i]),"l"(b[j]),"l"(cl),"l"(rl));
#else
          umul(hi,lo,a[i],b[j]);
          limb t=lo+cl; hi+=(t<lo); lo=t; t=lo+rl; hi+=(t<lo); lo=t;
#endif
          r[i+j]=lo; cl=hi;
        }
      r[NL+j]=cl;
    }
}

/* ---------------- fix_mul: r = a*b, RNDN ---------------- */
__host__ __device__ static F1024
fix_mul (const F1024 &a, const F1024 &b)
{
  F1024 r;
  if (f_is_zero(a)||f_is_zero(b)){ r.sign=1; r.exp=ZERO_EXP;
#pragma unroll
    for(int i=0;i<NL;i++) r.m[i]=0; return r; }
  limb P[2*NL];
  mul_16x16 (P, a.m, b.m);
  r.sign = a.sign*b.sign;
  long e = a.exp + b.exp;
  limb rbit, sticky;
  if (P[2*NL-1] & (1ULL<<63))           /* product MSB at bit 2047: keep >>1024 */
    {
#pragma unroll
      for (int i=0;i<NL;i++) r.m[i]=P[NL+i];
      rbit = (P[NL-1]>>63)&1;
      sticky = (P[NL-1] & ~(1ULL<<63))!=0;
#pragma unroll
      for (int i=0;i<NL-1;i++) sticky |= (P[i]!=0);
    }
  else                                  /* MSB at 2046: keep >>1023, exp-1 */
    {
      e -= 1;
#pragma unroll
      for (int i=0;i<NL;i++)
        r.m[i] = (P[NL-1+i]>>63) | (P[NL+i]<<1);
      rbit = (P[NL-1]>>62)&1;
      sticky = (P[NL-1] & ((1ULL<<62)-1))!=0;
#pragma unroll
      for (int i=0;i<NL-1;i++) sticky |= (P[i]!=0);
    }
  /* round to nearest even */
  if (rbit && (sticky || (r.m[0]&1)))
    {
      limb c=1;
#pragma unroll
      for (int i=0;i<NL && c;i++){ r.m[i]+=c; c = (r.m[i]==0); }
      if (c){ /* overflow 2^1024 -> 2^1023 */
#pragma unroll
        for (int i=0;i<NL;i++) r.m[i]=0;
        r.m[NL-1]=1ULL<<63; e++;
      }
    }
  r.exp = e;
  return r;
}

/* ---------------- magnitude compare ---------------- */
__host__ __device__ static int
cmpmag (const F1024 &a, const F1024 &b)
{
  if (a.exp!=b.exp) return a.exp>b.exp?1:-1;
#pragma unroll
  for (int i=NL-1;i>=0;i--) if (a.m[i]!=b.m[i]) return a.m[i]>b.m[i]?1:-1;
  return 0;
}

/* round-to-even helper on r.m[NL] given rbit/sticky, may bump exp via *e */
__host__ __device__ static void
round_norm (F1024 &r, limb rbit, limb sticky, long &e)
{
  if (rbit && (sticky || (r.m[0]&1)))
    {
      limb c=1;
#pragma unroll
      for (int i=0;i<NL && c;i++){ r.m[i]+=c; c=(r.m[i]==0); }
      if (c){
#pragma unroll
        for (int i=0;i<NL;i++) r.m[i]=0;
        r.m[NL-1]=1ULL<<63; e++;
      }
    }
  r.exp=e;
}

/* ---- generic bit helpers over a limb array (round/sticky extraction) ---- */
__host__ __device__ static inline limb
wbit (const limb *W, int len, long i)
{ if (i<0||i>=(long)len*64) return 0; return (W[i>>6]>>(i&63))&1; }
__host__ __device__ static inline limb
wsticky_below (const limb *W, int len, long i)        /* OR of bits [0..i-1] */
{
  if (i<=0) return 0;
  long full=i>>6; limb s=0;
  for (long j=0;j<full && j<len;j++) s|=W[j];
  int rem=(int)(i&63); if (rem && full<len) s|= W[full] & (((limb)1<<rem)-1);
  return s!=0;
}
__host__ __device__ static inline void
wshift_right (const limb *W, int len, long lo, limb out[NL])  /* out = W >> lo */
{
  long ws=lo>>6; int bs=(int)(lo&63);
  for (int i=0;i<NL;i++){ long s=ws+i;
    limb a=(s>=0&&s<len)?W[s]:0, b2=(s+1>=0&&s+1<len)?W[s+1]:0;
    out[i]= bs? ((a>>bs)|(b2<<(64-bs))) : a; }
}

/* ---------------- magnitude add: |a|+|b|, given sign ---------------- */
__host__ __device__ static F1024
addmag (F1024 a, F1024 b, int sign)
{
  if (b.exp>a.exp){ F1024 t=a; a=b; b=t; }
  long d = a.exp-b.exp;
  /* shift b.m right by d -> Bsh[NL] + rbit/sticky (round bit = bit d-1 of b) */
  limb Bsh[NL]; limb rbit, sticky;
  if (d>=NL*64){ for(int i=0;i<NL;i++) Bsh[i]=0;
    rbit = wbit(b.m,NL,d-1); sticky = wsticky_below(b.m,NL,d-1); }
  else {
    wshift_right (b.m,NL,d,Bsh);
    rbit = wbit(b.m,NL,d-1); sticky = wsticky_below(b.m,NL,d-1);
  }
  F1024 r; r.sign=sign;
  /* sum = a.m + Bsh */
  limb c=0;
#pragma unroll
  for (int i=0;i<NL;i++){ limb s=a.m[i]+c; c=(s<c); s+=Bsh[i]; c+=(s<Bsh[i]); r.m[i]=s; }
  long e=a.exp;
  if (c){ /* carry out of bit 1023 -> shift right 1, MSB set */
    sticky |= rbit;
    rbit = r.m[0]&1;
#pragma unroll
    for (int i=0;i<NL-1;i++) r.m[i]=(r.m[i]>>1)|(r.m[i+1]<<63);
    r.m[NL-1]=(r.m[NL-1]>>1)|(1ULL<<63);
    e++;
  }
  round_norm (r, rbit, sticky, e);
  return r;
}

/* ---------------- magnitude sub: |a|-|b|, |a|>|b|, given sign --------- *
 * wide 18-limb (1024+128 guard) exact subtraction then normalize+round.  */
__host__ __device__ static F1024
submag (const F1024 &a, const F1024 &b, int sign)
{
  long d = a.exp-b.exp;                 /* >=0 */
  /* build A18 = a.m << 128 (a.m in top 16 of 18 limbs) */
  limb A18[WL], B18[WL]; limb ext_sticky=0;
  A18[0]=A18[1]=0;
#pragma unroll
  for (int i=0;i<NL;i++) A18[i+2]=a.m[i];
  /* B18 = (b.m in top 16 of 18 limbs) >> d, sticky for bits below limb0 */
  for (int i=0;i<WL;i++) B18[i]=0;
  if (d < WL*64){
    int ws=(int)(d>>6), bs=(int)(d&63);
    /* place b.m into a temp 18-limb (top16), then right-shift by d */
    limb T[WL]; T[0]=T[1]=0;
    for (int i=0;i<NL;i++) T[i+2]=b.m[i];
    /* sticky: bits shifted past limb0 = low d bits of (T as 1152-bit) */
    for (int i=0;i<ws && i<WL;i++) ext_sticky |= (T[i]!=0);
    if (bs && ws<WL) ext_sticky |= (T[ws] & (((limb)1<<bs)-1))!=0;
    for (int i=0;i<WL;i++){
      limb lo = (i+ws<WL)? T[i+ws] : 0;
      limb hi = (i+ws+1<WL)? T[i+ws+1] : 0;
      B18[i] = bs ? ((lo>>bs)|(hi<<(64-bs))) : lo;
    }
  } else {
    ext_sticky = 1;                     /* b entirely below guard */
  }
  /* D18 = A18 - B18  (nonneg since |a|>|b|; ext_sticky means true b a hair bigger
   * -> subtract 1 ulp-at-limb0 borrow handled by treating ext as fractional) */
  limb D18[WL]; limb br=0;
  for (int i=0;i<WL;i++){ limb ai=A18[i], bi=B18[i]; limb s=ai-bi; limb b2=(ai<bi);
    limb s2=s-br; b2|=(s<br); D18[i]=s2; br=b2; }
  /* if ext_sticky, the true subtrahend is B18 + eps (0<eps<1 at limb0), so the
   * exact result is D18 - eps; for rounding we treat: lower the value by eps,
   * i.e. round/sticky get an extra borrow into the guard.  Handle by noting the
   * exact remainder below the kept window includes -eps; we fold via sticky and a
   * conditional decrement when the guard is exactly zero. Simpler: subtract 1 from
   * D18 if ext_sticky (eps>0) -- correct because eps in (0,1) at limb0 weight, and
   * then the fractional part becomes (1-eps) recorded as sticky. */
  if (ext_sticky){
    limb bb=1;
    for (int i=0;i<WL && bb;i++){ limb s=D18[i]-bb; bb=(D18[i]<bb); D18[i]=s; }
  }
  /* find MSB position p (0..WL*64-1) */
  int hl=-1;
  for (int i=WL-1;i>=0;i--){ if (D18[i]){ hl=i; break; } }
  F1024 r; r.sign=sign;
  if (hl<0){ /* exact zero */ r.sign=1; r.exp=ZERO_EXP;
#pragma unroll
    for(int i=0;i<NL;i++) r.m[i]=0; return r; }
  int top = 63; { limb v=D18[hl]; while (!((v>>top)&1)) top--; }
  int p = hl*64+top;                    /* MSB bit index in D18 */
  long e = a.exp - (WL*64-1) + p;       /* weight bookkeeping: bit (WL*64-1)->a.exp-1 */
  /* extract top 1024 bits [p-1023 .. p] into r.m, with rbit/sticky below */
  limb rbit=0, sticky=ext_sticky;
  /* we want a right-shift of D18 by (p-1023) (if >=0) so bit(p-1023)->bit0 */
  long sh = (long)p - 1023;
  if (sh<=0){
    /* result < 1024 bits: left shift by -sh, exact */
    int ls=(int)(-sh); int ws=ls>>6, bs=ls&63;
#pragma unroll
    for (int i=NL-1;i>=0;i--){
      int src=i-ws;
      limb lo = (src>=0 && src<WL)? D18[src]:0;
      limb hi = (src-1>=0 && src-1<WL)? D18[src-1]:0;
      r.m[i] = bs? ((lo<<bs)|(hi>>(64-bs))) : lo;
    }
    rbit=0; /* sticky already ext_sticky (should be 0 in this cancellation case) */
  } else {
    rbit = wbit(D18,WL,sh-1);
    sticky |= wsticky_below(D18,WL,sh-1);
    wshift_right (D18,WL,sh,r.m);
  }
  round_norm (r, rbit, sticky, e);
  return r;
}

/* ---------------- fix_add: r = a+b, RNDN ---------------- */
__host__ __device__ static F1024
fix_add (const F1024 &a, const F1024 &b)
{
  if (f_is_zero(a)) return b;
  if (f_is_zero(b)) return a;
  if (a.sign==b.sign) return addmag (a,b,a.sign);
  int c = cmpmag (a,b);
  if (c==0){ F1024 r; r.sign=1; r.exp=ZERO_EXP;
#pragma unroll
    for(int i=0;i<NL;i++) r.m[i]=0; return r; }
  return (c>0)? submag (a,b,a.sign) : submag (b,a,b.sign);
}

/* ============================ GPU kernels ============================ */
#ifndef NN
#define NN 16384
#endif
#ifndef LBLOCKS
#define LBLOCKS 256
#endif
#ifndef LTHREADS
#define LTHREADS 64
#endif

__global__ void
axpy_fast_kernel (int n, double a, const double *x, const double *y, double *out)
{
  int stride = gridDim.x*blockDim.x;
  F1024 fa = from_double (a);
  for (int i=blockIdx.x*blockDim.x+threadIdx.x; i<n; i+=stride)
    {
      F1024 fx = from_double (x[i]);
      F1024 fy = from_double (y[i]);
      F1024 t  = fix_mul (fa, fx);      /* t = a*x   */
      F1024 r  = fix_add (t, fy);       /* r = a*x+y */
      out[i] = to_double (r);
    }
}

/* validation kernel: device fix_mul / fix_add on full-width random operands */
__global__ void
valid_kernel (int n, const F1024 *A, const F1024 *B, F1024 *MUL, F1024 *ADD)
{
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i<n){ MUL[i]=fix_mul(A[i],B[i]); ADD[i]=fix_add(A[i],B[i]); }
}

/* ============================ host: MPFR reference ============================ */
#include <gmp.h>
#include <mpfr.h>

static limb xs (limb *s){ limb x=*s; x^=x<<13; x^=x>>7; x^=x<<17; return *s=x; }

static F1024 rand_f (limb *s, int expspread)
{
  F1024 r; r.sign = (xs(s)&1)?1:-1;
  for (int i=0;i<NL;i++) r.m[i]=xs(s);
  r.m[NL-1] |= 1ULL<<63;                 /* normalize MSB */
  r.exp = (long)((xs(s)% (2*expspread+1)) ) - expspread;
  return r;
}
static void f_to_mpfr (mpfr_t out, const F1024 &x)
{
  if (f_is_zero(x)){ mpfr_set_zero(out,1); return; }
  mpz_t M; mpz_init(M);
  mpz_import (M, NL, -1, sizeof(limb), 0, 0, x.m);   /* little-endian limbs */
  mpfr_set_z (out, M, MPFR_RNDN);                    /* exact: 1024-bit */
  mpfr_mul_2si (out, out, x.exp - 1024, MPFR_RNDN);  /* value = M*2^(exp-1024) */
  if (x.sign<0) mpfr_neg (out,out,MPFR_RNDN);
  mpz_clear(M);
}

int main (void)
{
  cudaDeviceSetLimit (cudaLimitStackSize, (size_t)64*1024);

  /* ---------- (1) ROUNDING VALIDATION vs system MPFR (random full-width) ---------- */
  const int M = 200000;
  F1024 *A=(F1024*)malloc(M*sizeof(F1024)), *B=(F1024*)malloc(M*sizeof(F1024));
  limb seed=0x9e3779b97f4a7c15ULL;
  for (int i=0;i<M;i++){
    int spread = (i&3)? 40 : 1100;        /* mix small + large exponent gaps */
    A[i]=rand_f(&seed,spread); B[i]=rand_f(&seed,spread);
  }
  F1024 *dA,*dB,*dMUL,*dADD;
  cudaMalloc(&dA,M*sizeof(F1024)); cudaMalloc(&dB,M*sizeof(F1024));
  cudaMalloc(&dMUL,M*sizeof(F1024)); cudaMalloc(&dADD,M*sizeof(F1024));
  cudaMemcpy(dA,A,M*sizeof(F1024),cudaMemcpyHostToDevice);
  cudaMemcpy(dB,B,M*sizeof(F1024),cudaMemcpyHostToDevice);
  valid_kernel<<<(M+127)/128,128>>>(M,dA,dB,dMUL,dADD);
  cudaDeviceSynchronize();
  F1024 *MUL=(F1024*)malloc(M*sizeof(F1024)), *ADD=(F1024*)malloc(M*sizeof(F1024));
  cudaMemcpy(MUL,dMUL,M*sizeof(F1024),cudaMemcpyDeviceToHost);
  cudaMemcpy(ADD,dADD,M*sizeof(F1024),cudaMemcpyDeviceToHost);

  mpfr_t ma,mb,ref,mine; mpfr_inits2 (1024,ma,mb,ref,mine,(mpfr_ptr)0);
  long mul_bad=0, add_bad=0;
  for (int i=0;i<M;i++){
    f_to_mpfr(ma,A[i]); f_to_mpfr(mb,B[i]);
    mpfr_mul(ref,ma,mb,MPFR_RNDN); f_to_mpfr(mine,MUL[i]);
    if (!mpfr_equal_p(ref,mine)) mul_bad++;
    mpfr_add(ref,ma,mb,MPFR_RNDN); f_to_mpfr(mine,ADD[i]);
    if (!mpfr_equal_p(ref,mine)) add_bad++;
  }
  printf ("=== rounding validation vs system MPFR (1024-bit RNDN), %d random pairs ===\n",M);
  printf ("fix_mul mismatches : %ld / %d   %s\n", mul_bad,M, mul_bad?"FAIL":"OK");
  printf ("fix_add mismatches : %ld / %d   %s\n", add_bad,M, add_bad?"FAIL":"OK");

  /* ---------- (2) AXPY SPEED (double operands), same config as axpy_mpfr ---------- */
  double a=1.5;
  double *x=(double*)malloc(NN*sizeof(double)), *y=(double*)malloc(NN*sizeof(double));
  double *gpu=(double*)malloc(NN*sizeof(double)), *cpu=(double*)malloc(NN*sizeof(double));
  for (int i=0;i<NN;i++){ x[i]=1.0+i*1e-3; y[i]=2.0-i*7e-4; }
  double *dx,*dy,*dout;
  cudaMalloc(&dx,NN*sizeof(double)); cudaMalloc(&dy,NN*sizeof(double)); cudaMalloc(&dout,NN*sizeof(double));
  cudaMemcpy(dx,x,NN*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(dy,y,NN*sizeof(double),cudaMemcpyHostToDevice);
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  const int ITERS=50;
  axpy_fast_kernel<<<LBLOCKS,LTHREADS>>>(NN,a,dx,dy,dout); cudaDeviceSynchronize();
  cudaEventRecord(e0);
  for (int it=0;it<ITERS;it++) axpy_fast_kernel<<<LBLOCKS,LTHREADS>>>(NN,a,dx,dy,dout);
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float tot=0; cudaEventElapsedTime(&tot,e0,e1); float gpu_ms=tot/ITERS;
  cudaMemcpy(gpu,dout,NN*sizeof(double),cudaMemcpyDeviceToHost);

  /* CPU reference: system MPFR 1024-bit, same a*x+y -> double */
  struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
  for (int i=0;i<NN;i++){
    mpfr_set_d(ma,a,MPFR_RNDN); mpfr_set_d(mb,x[i],MPFR_RNDN);
    mpfr_mul(ref,ma,mb,MPFR_RNDN);
    mpfr_set_d(mb,y[i],MPFR_RNDN); mpfr_add(ref,ref,mb,MPFR_RNDN);
    cpu[i]=mpfr_get_d(ref,MPFR_RNDN);
  }
  clock_gettime(CLOCK_MONOTONIC,&t1);
  double cpu_ms=(t1.tv_sec-t0.tv_sec)*1e3+(t1.tv_nsec-t0.tv_nsec)*1e-6;
  double maxrel=0; int mism=0;
  for (int i=0;i<NN;i++){ double rel=cpu[i]?fabs((gpu[i]-cpu[i])/cpu[i]):fabs(gpu[i]);
    if(rel>maxrel)maxrel=rel; if(rel>1e-13)mism++; }
  printf ("\n=== fast 1024-bit AXPY (y=a*x+y), N=%d, register-resident ===\n",NN);
  printf ("launch   : %d blocks x %d threads = %d resident (NO arena)\n",LBLOCKS,LTHREADS,LBLOCKS*LTHREADS);
  printf ("GPU time : %8.3f ms\n",gpu_ms);
  printf ("CPU time : %8.3f ms   (system MPFR 1024-bit on host)\n",cpu_ms);
  printf ("accuracy : max rel |GPU-CPU| = %.3e (%d exceed 1e-13)\n",maxrel,mism);
  printf ("sample   : y[0]=%.15g y[1]=%.15g y[N-1]=%.15g\n",gpu[0],gpu[1],gpu[NN-1]);
  return (mul_bad||add_bad||mism)?1:0;
}
