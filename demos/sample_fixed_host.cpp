/* sample_fixed_host.cpp -- the fixed-precision API on the CPU, without CUDA.
 *
 * One #include "mpc_cuda_host.h" gives cu_fp::cu_freal<PB> / cu_fcomplex<PB>
 * (PB = mantissa bits, any multiple of 32, fixed at compile time), the
 * correctly rounded fused operations and the elementary functions.  + - * are
 * bit-exact with MPFR / MPC round-to-nearest.
 *
 * Build (header-only: no libraries):
 *   g++ -std=c++17 -O2 -march=native -Iinclude demos/sample_fixed_host.cpp \
 *       -o sample_fixed_host
 */
#include <cstdio>
#include "mpc_cuda_host.h"
using cu_fp::cu_freal;
using cu_fp::cu_fcomplex;

template<int PB>
static void axpy (const char *tag)
{
  const int n = 8;
  double out[n];
  cu_freal<PB> a = 1.5;                                  /* double -> fixed */
  for (int i=0;i<n;i++)
    {
      cu_freal<PB> x = 1.0+i*0.25, y = 2.0-i*0.1;
      out[i] = (double)(a*x + y);                        /* operators, fixed -> double */
    }
  printf("%-10s PB=%-4d : out[0]=%.15g  out[7]=%.15g\n", tag, PB, out[0], out[7]);
}

int main (void)
{
  printf("fixed-precision AXPY (y=1.5*x+y) on the host:\n");
  axpy<64>   ("cu_freal");
  axpy<128>  ("cu_freal");
  axpy<256>  ("cu_freal");
  axpy<512>  ("cu_freal");
  axpy<1024> ("cu_freal");

  /* complex, bit-exact with MPC */
  cu_fcomplex<256> za(1.5,-0.25), zx(1.0,0.5), zy(2.0,-1.0);
  cu_fcomplex<256> zz = za*zx + zy;                       /* (1.625,0.5)+(2,-1) */
  printf("PB=256 : complex (1.5-0.25i)*(1+0.5i)+(2-1i) = (%.12g, %.12g)\n",
         zz.real_d(), zz.imag_d());

  /* correctly rounded dot product and fma (one rounding each) */
  cu_freal<256> x[4] = {1.0, 1e-30, -1.0, 3.0}, y[4] = {1.0, 1.0, 1.0, 0.5};
  printf("PB=256 : dot = %.15g   (1 + 1e-30 - 1 + 1.5, rounded once)\n",
         (double) cu_fp::cu_fdot<256>(x, y, 4));
  /* (1+e)(1-e) - 1 = -e^2 with e = 2^-200: the product rounds to 1 at 256 bits,
   * so mul+add (cu_ffma) gives 0 while one rounding (cu_ffma_cr) keeps -e^2 */
  cu_freal<256> one = 1.0, e = cu_fp::cu_scale2<256>(one, -200);
  cu_freal<256> p = one + e, q = one - e, m1 = -1.0;
  printf("PB=256 : (1+e)(1-e)-1, e=2^-200:  cu_ffma = %g   cu_ffma_cr = %g  (= -2^-400)\n",
         (double) cu_fp::cu_ffma<256>(p, q, m1), (double) cu_fp::cu_ffma_cr<256>(p, q, m1));

  /* elementary functions */
  printf("PB=256 : exp(1)=%.15g  log(2)=%.15g  sin(1)=%.15g\n",
         cu_fp::cu_exp<256>(cu_freal<256>(1.0)).to_double(),
         cu_fp::cu_log<256>(cu_freal<256>(2.0)).to_double(),
         cu_fp::cu_sin<256>(cu_freal<256>(1.0)).to_double());
  cu_fcomplex<256> ce = cu_fp::cu_cexp<256>(cu_fcomplex<256>(0.0, 3.14159265358979312));
  printf("PB=256 : cexp(i*pi) = (%.12g, %.12g)   (~ -1 + 0i)\n", ce.real_d(), ce.imag_d());
  return 0;
}
