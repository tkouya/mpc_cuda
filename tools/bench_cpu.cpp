/* bench_cpu.cpp -- CPU-only comparison of the header-only fixed-precision types
 * cu_fp::cu_freal<P> / cu_fcomplex<P> against the system runtime-precision
 * MPFR / MPC, both on the host (single thread), for AXPY r = a*x + y across
 * mantissa widths 128..8192 bits.  No CUDA: built with the host C++ compiler.
 *
 *   g++ -O3 -march=native -Iinclude -I/usr/local/include tools/bench_cpu.cpp \
 *       -o /tmp/bench_cpu -L/usr/local/lib -lmpc -lmpfr -lgmp
 */
#include "mpc_cuda/cu_freal.cuh"
#include "mpc_cuda/cu_fcomplex.cuh"
#include <mpfr.h>
#include <mpc.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
using cu_fp::cu_freal;
using cu_fp::cu_fcomplex;

#ifndef N
#define N 4096
#endif
#ifndef REPS
#define REPS 8
#endif
static double now_ms(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec*1e-6; }

static double X[N],Y[N], XR[N],XI[N],YR[N],YI[N];
static const double AR=1.5, AI=-0.25;

template<int P> static void bench(){
  /* ---- real: cu_freal vs system MPFR ---- */
  double accf=0;
  double t0=now_ms();
  for(int r=0;r<REPS;r++) for(int i=0;i<N;i++){
    cu_freal<P> v = cu_freal<P>(AR)*cu_freal<P>(X[i]) + cu_freal<P>(Y[i]);
    accf += v.to_double();
  }
  double tf=(now_ms()-t0)/REPS;
  mpfr_t a,x,y,t,rr; mpfr_inits2((mpfr_prec_t)P,a,x,y,t,rr,(mpfr_ptr)0);
  double accm=0; t0=now_ms();
  for(int r=0;r<REPS;r++) for(int i=0;i<N;i++){
    mpfr_set_d(a,AR,MPFR_RNDN);mpfr_set_d(x,X[i],MPFR_RNDN);mpfr_set_d(y,Y[i],MPFR_RNDN);
    mpfr_mul(t,a,x,MPFR_RNDN);mpfr_add(rr,t,y,MPFR_RNDN); accm+=mpfr_get_d(rr,MPFR_RNDN);
  }
  double tm=(now_ms()-t0)/REPS;
  mpfr_clears(a,x,y,t,rr,(mpfr_ptr)0);
  printf("REAL %5d : freal %8.3f ms   MPFR %8.3f ms   freal/MPFR %5.2fx   (chk %.3g/%.3g)\n",
         P,tf,tm, tm/tf, accf, accm);

  /* ---- complex: cu_fcomplex vs system MPC ---- */
  double t1=now_ms();
  double cr=0;
  for(int r=0;r<REPS;r++) for(int i=0;i<N;i++){
    cu_fcomplex<P> v = cu_fcomplex<P>(AR,AI)*cu_fcomplex<P>(XR[i],XI[i]) + cu_fcomplex<P>(YR[i],YI[i]);
    cr += v.real_d();
  }
  double tfc=(now_ms()-t1)/REPS;
  mpc_t ca,cx,cy,ct,crr; mpc_init2(ca,(mpfr_prec_t)P);mpc_init2(cx,(mpfr_prec_t)P);mpc_init2(cy,(mpfr_prec_t)P);
  mpc_init2(ct,(mpfr_prec_t)P);mpc_init2(crr,(mpfr_prec_t)P);
  double cm=0; t1=now_ms();
  for(int r=0;r<REPS;r++) for(int i=0;i<N;i++){
    mpc_set_d_d(ca,AR,AI,MPC_RNDNN);mpc_set_d_d(cx,XR[i],XI[i],MPC_RNDNN);mpc_set_d_d(cy,YR[i],YI[i],MPC_RNDNN);
    mpc_mul(ct,ca,cx,MPC_RNDNN);mpc_add(crr,ct,cy,MPC_RNDNN); cm+=mpfr_get_d(mpc_realref(crr),MPFR_RNDN);
  }
  double tmc=(now_ms()-t1)/REPS;
  mpc_clear(ca);mpc_clear(cx);mpc_clear(cy);mpc_clear(ct);mpc_clear(crr);
  printf("CPLX %5d : fcplx %8.3f ms   MPC  %8.3f ms   fcplx/MPC  %5.2fx   (chk %.3g/%.3g)\n",
         P,tfc,tmc, tmc/tfc, cr, cm);
}

int main(){
  for(int i=0;i<N;i++){ X[i]=1.0+i*1e-4; Y[i]=2.0-i*7e-5;
    XR[i]=1.0+i*1e-4;XI[i]=0.5-i*3e-5;YR[i]=2.0-i*7e-5;YI[i]=-1.0+i*2e-5; }
  printf("=== CPU-only AXPY r=a*x+y, N=%d, %d-rep avg (single thread) ===\n",N,REPS);
  printf("    cu_freal/cu_fcomplex<P> (header, __int128, schoolbook) vs system MPFR/MPC (runtime prec)\n");
  printf("    ratio>1 => the fixed-precision header type is faster\n\n");
  bench<128>(); bench<256>(); bench<512>(); bench<1024>();
  bench<2048>(); bench<4096>(); bench<8192>();
  return 0;
}
