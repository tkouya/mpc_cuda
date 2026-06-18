/* mpc_cuda.cuh -- single-include umbrella for the mpc_cuda GPU library.
 *
 *   #include "mpc_cuda.cuh"
 *
 * gives you the WHOLE multiple-precision stack (mini-gmp, MPFR and MPC) running
 * inside CUDA kernels, with every public name living in the cu_ / CU_ namespace:
 *
 *     types      cu_mpz_t   cu_mpfr_t   cu_mpc_t   cu_mpfr_rnd_t  ...
 *     functions  cu_mpz_*   cu_mpfr_*   cu_mpc_*                  ...
 *     constants  CU_MPFR_RNDN  CU_MPFR_PREC_MAX  CU_MPC_RNDNN     ...
 *     init macro CU_MPFR_DECL_INIT(x, prec)
 *
 * Because nothing here is spelled mpfr_t / MPFR_RNDN / __MPFR_H, this header can
 * be included in the SAME .cu file as the system <gmp.h>/<mpfr.h>/<mpc.h>
 * (CPU GMP/MPFR/MPC) without any clash -- in either include order.  Use the
 * plain names for the CPU and the cu_ names for the GPU.  See demos/sample_easy.cu.
 *
 * No -D flags and no extra -I beyond the install's include root are required;
 * the headers carry the build constants the library was compiled with.
 *
 * Link with libmpc_cuda (e.g. via the installed `mpc_cuda-link` helper, or by
 * linking the prebuilt objects -- see the manual / Makefile `easy` target).
 */
#ifndef MPC_CUDA_UMBRELLA_CUH
#define MPC_CUDA_UMBRELLA_CUH

/* mini-gmp (cu_mpz_*, cu_mpn_*) + the per-thread bump-arena allocator the
 * device fast path uses (mpc_cuda_arena_* / mpc_cuda_arena_reset). */
#include "mpc_cuda/cu_gmp.h"

/* real MPFR-4.2.2, device-adapted   (cu_mpfr_*) */
#include "mpc_cuda/cu_mpfr.h"

/* real MPC-1.4.1, device-adapted    (cu_mpc_*)  */
#include "mpc_cuda/cu_mpc.h"

/* fixed-precision, register-resident fast path: cu_fp::cu_freal<PB> (PB a
 * multiple of 32).  Compile-time precision keeps the significand in registers
 * (no arena, no runtime dispatch) -- much faster than the runtime cu_mpfr path
 * when the precision is known at compile time, and bit-exact with MPFR RNDN.
 * Header-only C++ template; usable on GPU and CPU.  See demos/sample_fixed.cu. */
#include "mpc_cuda/cu_freal.cuh"

/* fixed-precision complex on top of cu_freal: cu_fp::cu_fcomplex<PB>.  Complex
 * multiply is correctly rounded per component (bit-exact with MPC's MPC_RNDNN)
 * via the exact-intermediate fmms/fmma identity. */
#include "mpc_cuda/cu_fcomplex.cuh"

/* fixed-precision elementary functions (high-accuracy, ~<=1 ULP vs MPFR/MPC,
 * NOT correctly-rounded): real cu_fp::cu_{exp,log,expm1,log1p,sin,cos,tan,atan,
 * sinh,cosh,asin,acos,atanh,pow,sqrt,cbrt,...}  and complex cu_fp::cu_{cexp,clog,
 * csqrt,csin,ccos,ctan,csinh,ccosh}.  Header-only templates; fastest at low/
 * medium precision.  (Real/complex special functions are future work.) */
#include "mpc_cuda/cu_fmath.cuh"
#include "mpc_cuda/cu_fcmath.cuh"

#endif /* MPC_CUDA_UMBRELLA_CUH */
