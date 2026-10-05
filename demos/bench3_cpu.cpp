/* bench3_cpu.cpp -- CPU (system MPFR/MPC) side of the 3-way AXPY benchmark.
 * Compiled by the host C++ compiler against the system <mpfr.h>/<mpc.h>; the
 * GPU side (demos/bench_3way.cu, nvcc) calls these through the type-free
 * extern "C" interface so the system headers never meet the cu_ library headers
 * in one translation unit.  Returns wall-clock ms and fills the output arrays.
 *
 * The AXPY loops are OpenMP-parallelized so the CPU contender uses every core --
 * a single-thread CPU number would make the GPU/CPU ratio meaningless.  Each
 * thread owns its own MPFR/MPC temporaries (inited once, reused across the
 * thread's chunk), so there is no sharing inside the timed region.
 */
#include <mpfr.h>
#include <mpc.h>
#include <ctime>

#ifdef _OPENMP
#include <omp.h>
#endif

static double now_ms(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
  return t.tv_sec*1e3 + t.tv_nsec*1e-6; }

/* Pin the team to the largest thread count the machine offers (once). */
static int ensure_threads(){
#ifdef _OPENMP
  static int n = 0;
  //if(!n){ n = omp_get_num_procs(); omp_set_num_threads(n); }
  if(!n){ n = omp_get_max_threads(); omp_set_num_threads(n); }
  return n;
#else
  return 1;
#endif
}
extern "C" int cpu_omp_threads(void){ return ensure_threads(); }

extern "C" double
cpu_axpy_real(long prec,int n,double a,const double*X,const double*Y,double*O)
{
  ensure_threads();
  double t0=now_ms();
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    mpfr_t ma,mx,my,t,r; mpfr_inits2((mpfr_prec_t)prec,ma,mx,my,t,r,(mpfr_ptr)0);
    mpfr_set_d(ma,a,MPFR_RNDN);                 /* the scalar is loop-invariant */
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for(int i=0;i<n;i++){
      mpfr_set_d(mx,X[i],MPFR_RNDN); mpfr_set_d(my,Y[i],MPFR_RNDN);
      mpfr_mul(t,ma,mx,MPFR_RNDN); mpfr_add(r,t,my,MPFR_RNDN);
      O[i]=mpfr_get_d(r,MPFR_RNDN);
    }
    mpfr_clears(ma,mx,my,t,r,(mpfr_ptr)0);
  }
  double ms=now_ms()-t0; return ms;
}

extern "C" double
cpu_axpy_cplx(long prec,int n,double ar,double ai,
              const double*XR,const double*XI,const double*YR,const double*YI,
              double*OR,double*OI)
{
  ensure_threads();
  double t0=now_ms();
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    mpc_t a,x,y,t,r;
    mpc_init2(a,(mpfr_prec_t)prec);mpc_init2(x,(mpfr_prec_t)prec);mpc_init2(y,(mpfr_prec_t)prec);
    mpc_init2(t,(mpfr_prec_t)prec);mpc_init2(r,(mpfr_prec_t)prec);
    mpc_set_d_d(a,ar,ai,MPC_RNDNN);             /* the scalar is loop-invariant */
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for(int i=0;i<n;i++){
      mpc_set_d_d(x,XR[i],XI[i],MPC_RNDNN); mpc_set_d_d(y,YR[i],YI[i],MPC_RNDNN);
      mpc_mul(t,a,x,MPC_RNDNN); mpc_add(r,t,y,MPC_RNDNN);
      OR[i]=mpfr_get_d(mpc_realref(r),MPFR_RNDN); OI[i]=mpfr_get_d(mpc_imagref(r),MPFR_RNDN);
    }
    mpc_clear(a);mpc_clear(x);mpc_clear(y);mpc_clear(t);mpc_clear(r);
  }
  double ms=now_ms()-t0; return ms;
}
