/* sample_fixed.cu -- using the public fixed-precision API cu_fp::cu_freal<PB>.
 *
 * One #include "mpc_cuda.cuh" gives the register-resident fixed-precision real
 * type cu_freal<PB>, with PB (the mantissa width in bits) a COMPILE-TIME
 * template parameter that may be any multiple of 32 -- 32, 64, 96, 128, ...,
 * 256, 512, 1024, ...  Arithmetic (+, -, *) is bit-exact with MPFR RNDN.
 *
 * Build (no -D, no extra -I beyond the install root):
 *   nvcc -arch=sm_121 -O3 -fmad=false -Iinclude demos/sample_fixed.cu \
 *        -o sample_fixed
 * Optionally add the system MPFR to cross-check on the CPU:
 *   ... -I/usr/local/include -L/usr/local/lib -lmpfr -lgmp -DWITH_MPFR
 */
#include <cstdio>
#include "mpc_cuda.cuh"
using cu_fp::cu_freal;
using cu_fp::cu_fcomplex;

/* AXPY at three different compile-time precisions in one kernel launch. */
template<int PB>
__global__ void axpy (int n, double a, const double *x, const double *y, double *out)
{
  int s = gridDim.x*blockDim.x;
  cu_freal<PB> fa = a;                         /* double -> fixed (implicit)   */
  for (int i=blockIdx.x*blockDim.x+threadIdx.x; i<n; i+=s)
    {
      cu_freal<PB> r = fa * cu_freal<PB>(x[i]) + cu_freal<PB>(y[i]);   /* operators */
      out[i] = (double) r;                     /* fixed -> double               */
    }
}

template<int PB>
static void run (const char *tag)
{
  const int n = 8;
  double x[n], y[n], out[n];
  for (int i=0;i<n;i++){ x[i]=1.0+i*0.25; y[i]=2.0-i*0.1; }
  double *dx,*dy,*dout;
  cudaMalloc(&dx,sizeof x); cudaMalloc(&dy,sizeof y); cudaMalloc(&dout,sizeof out);
  cudaMemcpy(dx,x,sizeof x,cudaMemcpyHostToDevice);
  cudaMemcpy(dy,y,sizeof y,cudaMemcpyHostToDevice);
  axpy<PB><<<1,n>>>(n,1.5,dx,dy,dout);
  cudaDeviceSynchronize();
  cudaMemcpy(out,dout,sizeof out,cudaMemcpyDeviceToHost);
  printf("%-10s PB=%-4d : out[0]=%.15g  out[7]=%.15g\n", tag, PB, out[0], out[7]);
  cudaFree(dx); cudaFree(dy); cudaFree(dout);
}

int main (void)
{
  cudaDeviceSetLimit (cudaLimitStackSize, (size_t)64*1024);
  printf("fixed-precision register-resident AXPY (y=1.5*x+y), one kernel each:\n");
  run<64>   ("cu_freal");     /* 64-bit  mantissa */
  run<128>  ("cu_freal");     /* 128-bit          */
  run<256>  ("cu_freal");     /* 256-bit          */
  run<512>  ("cu_freal");     /* 512-bit          */
  run<1024> ("cu_freal");     /* 1024-bit         */

  /* host-side use of the SAME type (no nvcc execution-space needed at runtime) */
  cu_freal<256> a=1.5, b=1.0/3, c = a*b + a;
  printf("host    PB=256  : 1.5*(1/3)+1.5 = %.15g\n", (double)c);

  /* fixed-precision complex: cu_fcomplex<PB> (bit-exact with MPC) */
  cu_fcomplex<256> za(1.5,-0.25), zx(1.0,0.5), zy(2.0,-1.0);
  cu_fcomplex<256> zz = za*zx + zy;                    /* (1.625,0.5)+(2,-1) */
  printf("host    PB=256  : complex (1.5-0.25i)*(1+0.5i)+(2-1i) = (%.12g, %.12g)\n",
         zz.real_d(), zz.imag_d());

  /* fixed-precision elementary functions (cu_fmath / cu_fcmath) */
  printf("host    PB=256  : exp(1)=%.15g  log(2)=%.15g  sin(1)=%.15g\n",
         cu_fp::cu_exp<256>(cu_freal<256>(1.0)).to_double(),
         cu_fp::cu_log<256>(cu_freal<256>(2.0)).to_double(),
         cu_fp::cu_sin<256>(cu_freal<256>(1.0)).to_double());
  cu_fcomplex<256> ce = cu_fp::cu_cexp<256>(cu_fcomplex<256>(0.0, 3.14159265358979312));
  printf("host    PB=256  : cexp(i*pi) = (%.12g, %.12g)   (~ -1 + 0i)\n",
         ce.real_d(), ce.imag_d());
  return 0;
}
