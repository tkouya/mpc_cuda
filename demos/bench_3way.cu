/* bench_3way.cu -- three-way speed & accuracy comparison of multiple-precision
 * AXPY  (r = a*x + y),  real and complex, across mantissa widths
 * 128/256/512/1024/2048/4096/8192 bits:
 *
 *   (1) CPU            : system MPFR / MPC               (demos/bench3_cpu.cpp)
 *   (2) GPU cu_mpfr/mpc: this library's runtime-precision port (arena-backed)
 *   (3) GPU cu_freal/  : the fixed-precision register-resident fast path
 *       cu_fcomplex<P>   (P a COMPILE-TIME template parameter)
 *
 * The point is to see WHERE the fixed-precision fast path stops winning: at low
 * precision its significand lives in registers and it is far faster; as the
 * precision grows the per-thread arrays spill to local memory and the advantage
 * over the runtime cu_mpfr path (and eventually the CPU) erodes.
 *
 * The GPU contenders (2)+(3) are in this nvcc TU; the CPU side (1) lives in a
 * separate host C++ TU (system headers) reached via extern "C", so the system
 * <mpfr.h>/<mpc.h> never meet the cu_ library headers.  Build:  make bench3.
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
typedef long int gmp_randstate_t[1];
#include "mpc_cuda.cuh"
using cu_fp::cu_freal;
using cu_fp::cu_fcomplex;

extern "C" double cpu_axpy_real(long,int,double,const double*,const double*,double*);
extern "C" double cpu_axpy_cplx(long,int,double,double,const double*,const double*,
                                const double*,const double*,double*,double*);
extern "C" int    cpu_omp_threads(void);   /* OpenMP team size on the CPU side */

#ifndef N
#define N 4096
#endif
#define LB 128
#define LT 32
#define ITERS 8
#ifndef SLAB
#define SLAB (1024 * 1024)   /* per-thread arena slab; enlarged so the runtime
                               * cu_mpfr/cu_mpc path fits up to 65536-bit operands */
#endif

/* ---------- (3) fixed-precision register-resident kernels ---------- */
template<int P> __global__ void freal_axpy(int n,double a,const double*X,const double*Y,double*O){
  int s=gridDim.x*blockDim.x;
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=s){
    cu_freal<P> t = cu_freal<P>(a) * cu_freal<P>(X[i]);
    O[i] = (t + cu_freal<P>(Y[i])).to_double();
  }
}
template<int P> __global__ void fcplx_axpy(int n,double ar,double ai,
    const double*XR,const double*XI,const double*YR,const double*YI,double*OR,double*OI){
  int s=gridDim.x*blockDim.x; cu_fcomplex<P> a(ar,ai);
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=s){
    cu_fcomplex<P> t = a * cu_fcomplex<P>(XR[i],XI[i]);
    cu_fcomplex<P> r = t + cu_fcomplex<P>(YR[i],YI[i]);
    OR[i]=r.real_d(); OI[i]=r.imag_d();
  }
}
/* ---------- (2) runtime-precision cu_mpfr / cu_mpc kernels (arena) ---------- */
__global__ void cumpfr_axpy(int n,long prec,double a,const double*X,const double*Y,double*O){
  int s=gridDim.x*blockDim.x;
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=s){
    mpc_cuda_arena_reset();
    cu_mpfr_t ma,mx,my,t,r;
    cu_mpfr_init2(ma,prec);cu_mpfr_init2(mx,prec);cu_mpfr_init2(my,prec);cu_mpfr_init2(t,prec);cu_mpfr_init2(r,prec);
    cu_mpfr_set_d(ma,a,CU_MPFR_RNDN);cu_mpfr_set_d(mx,X[i],CU_MPFR_RNDN);cu_mpfr_set_d(my,Y[i],CU_MPFR_RNDN);
    cu_mpfr_mul(t,ma,mx,CU_MPFR_RNDN); cu_mpfr_add(r,t,my,CU_MPFR_RNDN);
    O[i]=cu_mpfr_get_d(r,CU_MPFR_RNDN);
    cu_mpfr_clear(ma);cu_mpfr_clear(mx);cu_mpfr_clear(my);cu_mpfr_clear(t);cu_mpfr_clear(r);
  }
}
__global__ void cumpc_axpy(int n,long prec,double ar,double ai,
    const double*XR,const double*XI,const double*YR,const double*YI,double*OR,double*OI){
  int s=gridDim.x*blockDim.x;
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=s){
    mpc_cuda_arena_reset();
    cu_mpc_t a,x,y,t,r;
    cu_mpc_init2(a,prec);cu_mpc_init2(x,prec);cu_mpc_init2(y,prec);cu_mpc_init2(t,prec);cu_mpc_init2(r,prec);
    cu_mpc_set_d_d(a,ar,ai,CU_MPC_RNDNN);cu_mpc_set_d_d(x,XR[i],XI[i],CU_MPC_RNDNN);cu_mpc_set_d_d(y,YR[i],YI[i],CU_MPC_RNDNN);
    cu_mpc_mul(t,a,x,CU_MPC_RNDNN); cu_mpc_add(r,t,y,CU_MPC_RNDNN);
    OR[i]=cu_mpfr_get_d(cu_mpc_realref(r),CU_MPFR_RNDN); OI[i]=cu_mpfr_get_d(cu_mpc_imagref(r),CU_MPFR_RNDN);
    cu_mpc_clear(a);cu_mpc_clear(x);cu_mpc_clear(y);cu_mpc_clear(t);cu_mpc_clear(r);
  }
}

/* ---------- shared device data ---------- */
static double *dX,*dY,*dOf,*dOm;
static double *dXR,*dXI,*dYR,*dYI,*dOfR,*dOfI,*dOmR,*dOmI;
static double *hX,*hY,*hXR,*hXI,*hYR,*hYI;
static const double AR=1.5, AI=-0.25;

static double maxrel(const double*p,const double*q,int n){
  double m=0; for(int i=0;i<n;i++){ double r=q[i]!=0?fabs((p[i]-q[i])/q[i]):fabs(p[i]); if(r>m)m=r; } return m;
}
template<typename K> static float timed(K launch,const char*tag){
  cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);
  cudaGetLastError();                                  /* clear any stale flag */
  launch(); cudaError_t e=cudaDeviceSynchronize();
  if(e!=cudaSuccess){ fprintf(stderr,"  [%s err: %s]\n",tag,cudaGetErrorString(e)); cudaGetLastError();
    cudaEventDestroy(a);cudaEventDestroy(b); return -1.f; }
  cudaEventRecord(a); for(int i=0;i<ITERS;i++) launch(); cudaEventRecord(b); cudaEventSynchronize(b);
  float t=0;cudaEventElapsedTime(&t,a,b);
  cudaEventDestroy(a);cudaEventDestroy(b); return t/ITERS;
}
static double sp(double num,double den){ return (den>0)?num/den:0.0; }

template<int P> static void bench(void){
  long prec=P;
  double *Of=(double*)malloc(N*sizeof(double)),*Om=(double*)malloc(N*sizeof(double)),*Oc=(double*)malloc(N*sizeof(double));
  float gf=timed([&]{ freal_axpy<P><<<LB,LT>>>(N,AR,dX,dY,dOf); },"freal");
  if(gf>=0) cudaMemcpy(Of,dOf,N*sizeof(double),cudaMemcpyDeviceToHost);
  float gm=timed([&]{ cumpfr_axpy<<<LB,LT>>>(N,prec,AR,dX,dY,dOm); },"cu_mpfr");
  if(gm>=0) cudaMemcpy(Om,dOm,N*sizeof(double),cudaMemcpyDeviceToHost);
  double cf=cpu_axpy_real(prec,N,AR,hX,hY,Oc);
  printf("REAL  %5d : freal %8.3f  cu_mpfr %9.3f  CPU %9.3f ms | freal/cu_mpfr %6.1fx  freal/CPU %7.1fx | rel(f)=%.1e rel(m)=%.1e\n",
    P,gf,gm,cf, sp(gm,gf), sp(cf,gf), gf>=0?maxrel(Of,Oc,N):-1.0, gm>=0?maxrel(Om,Oc,N):-1.0);

  double *OfR=(double*)malloc(N*sizeof(double)),*OfI=(double*)malloc(N*sizeof(double));
  double *OmR=(double*)malloc(N*sizeof(double)),*OmI=(double*)malloc(N*sizeof(double));
  double *OcR=(double*)malloc(N*sizeof(double)),*OcI=(double*)malloc(N*sizeof(double));
  float gfc=timed([&]{ fcplx_axpy<P><<<LB,LT>>>(N,AR,AI,dXR,dXI,dYR,dYI,dOfR,dOfI); },"fcplx");
  if(gfc>=0){ cudaMemcpy(OfR,dOfR,N*sizeof(double),cudaMemcpyDeviceToHost);cudaMemcpy(OfI,dOfI,N*sizeof(double),cudaMemcpyDeviceToHost); }
  float gmc=timed([&]{ cumpc_axpy<<<LB,LT>>>(N,prec,AR,AI,dXR,dXI,dYR,dYI,dOmR,dOmI); },"cu_mpc");
  if(gmc>=0){ cudaMemcpy(OmR,dOmR,N*sizeof(double),cudaMemcpyDeviceToHost);cudaMemcpy(OmI,dOmI,N*sizeof(double),cudaMemcpyDeviceToHost); }
  double cfc=cpu_axpy_cplx(prec,N,AR,AI,hXR,hXI,hYR,hYI,OcR,OcI);
  double rf=gfc>=0?fmax(maxrel(OfR,OcR,N),maxrel(OfI,OcI,N)):-1.0;
  double rm=gmc>=0?fmax(maxrel(OmR,OcR,N),maxrel(OmI,OcI,N)):-1.0;
  printf("CPLX  %5d : fcplx %8.3f  cu_mpc  %9.3f  CPU %9.3f ms | fcplx/cu_mpc  %6.1fx  fcplx/CPU %7.1fx | rel(f)=%.1e rel(m)=%.1e\n",
    P,gfc,gmc,cfc, sp(gmc,gfc), sp(cfc,gfc), rf, rm);
  free(Of);free(Om);free(Oc);free(OfR);free(OfI);free(OmR);free(OmI);free(OcR);free(OcI);
}

int main(void){
  setvbuf(stdout,NULL,_IONBF,0);
  /* NOTE: cudaLimitStackSize has a device-specific MAXIMUM (on GB10, 512KB is
   * rejected with cudaErrorInvalidValue and the stack silently stays at the 1KB
   * default -> the deep cu_mpc call chain then stack-overflows).  128KB is
   * accepted and is enough for the cu_mpc_mul->cu_mpfr_fmms->... chain.  Always CHECK
   * the return value. */
  cudaError_t es=cudaDeviceSetLimit(cudaLimitStackSize,(size_t)128*1024);
  if(es!=cudaSuccess) fprintf(stderr,"warning: cudaLimitStackSize: %s\n",cudaGetErrorString(es));
  cudaDeviceSetLimit(cudaLimitMallocHeapSize,(size_t)256*1024*1024);
  hX=(double*)malloc(N*sizeof(double));hY=(double*)malloc(N*sizeof(double));
  hXR=(double*)malloc(N*sizeof(double));hXI=(double*)malloc(N*sizeof(double));
  hYR=(double*)malloc(N*sizeof(double));hYI=(double*)malloc(N*sizeof(double));
  for(int i=0;i<N;i++){ hX[i]=1.0+i*1e-4; hY[i]=2.0-i*7e-5;
    hXR[i]=1.0+i*1e-4;hXI[i]=0.5-i*3e-5;hYR[i]=2.0-i*7e-5;hYI[i]=-1.0+i*2e-5; }
  size_t B=N*sizeof(double);
  cudaMalloc(&dX,B);cudaMalloc(&dY,B);cudaMalloc(&dOf,B);cudaMalloc(&dOm,B);
  cudaMalloc(&dXR,B);cudaMalloc(&dXI,B);cudaMalloc(&dYR,B);cudaMalloc(&dYI,B);
  cudaMalloc(&dOfR,B);cudaMalloc(&dOfI,B);cudaMalloc(&dOmR,B);cudaMalloc(&dOmI,B);
  cudaMemcpy(dX,hX,B,cudaMemcpyHostToDevice);cudaMemcpy(dY,hY,B,cudaMemcpyHostToDevice);
  cudaMemcpy(dXR,hXR,B,cudaMemcpyHostToDevice);cudaMemcpy(dXI,hXI,B,cudaMemcpyHostToDevice);
  cudaMemcpy(dYR,hYR,B,cudaMemcpyHostToDevice);cudaMemcpy(dYI,hYI,B,cudaMemcpyHostToDevice);
  size_t ntot=(size_t)LB*LT; char*arena; size_t*top;
  cudaMalloc(&arena,ntot*(size_t)SLAB); cudaMalloc(&top,ntot*sizeof(size_t)); cudaMemset(top,0,ntot*sizeof(size_t));
  mpc_cuda_arena_base=arena; mpc_cuda_arena_slab=SLAB; mpc_cuda_arena_top=top;
  cudaDeviceSynchronize();

  int cpu_threads = cpu_omp_threads();   /* CPU AXPY runs OpenMP on every core */
  printf("=== multiple-precision AXPY (r=a*x+y), N=%d, %d-iter avg ===\n",N,ITERS);
  printf("    CPU side: OpenMP on %d thread%s\n", cpu_threads, cpu_threads==1?"":"s");
  printf("    REAL: cu_freal<P> (GPU fixed) vs cu_mpfr (GPU runtime+arena) vs system MPFR (CPU)\n");
  printf("    CPLX: cu_fcomplex<P> (GPU fixed) vs cu_mpc (GPU runtime+arena) vs system MPC (CPU)\n");
  printf("    times in ms; ratio>1 means the GPU fixed path is faster; rel = max rel.error vs CPU\n\n");
  bench<128>(); bench<256>(); bench<512>(); bench<1024>();
  bench<2048>(); bench<4096>(); bench<8192>();
  bench<16384>(); bench<32768>(); bench<65536>();
  return 0;
}
