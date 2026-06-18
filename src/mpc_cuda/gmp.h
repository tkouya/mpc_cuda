/* shim: MPC's #include "gmp.h" -> CUDA mini-gmp + RODATA helper */
#include "mpc_cuda/cuda_minigmp.h"
#ifndef MPFR_RODATA
# ifdef __CUDA_ARCH__
#  define MPFR_RODATA __device__
# else
#  define MPFR_RODATA
# endif
#endif

/* mini-gmp has no random functions: MPFR/MPC need gmp_randstate_t declared. */
#ifndef gmp_randstate_t
typedef long int __gmp_randstate_struct;
typedef __gmp_randstate_struct gmp_randstate_t[1];
#endif

/* mini-gmp provides neither mpf_t (GMP float) nor mpq_t.  MPC declares a few
   conversion prototypes (cu_mpc_set_f/_q, ...) that we do not build for the
   device; provide minimal type stand-ins so the public header parses. */
#ifndef __MPF_STRUCT_DEFINED
#define __MPF_STRUCT_DEFINED
typedef struct { int _mp_prec; int _mp_size; long _mp_exp; mp_limb_t *_mp_d; } __mpf_struct;
typedef __mpf_struct mpf_t[1];
typedef __mpf_struct *mpf_ptr;
typedef const __mpf_struct *mpf_srcptr;
typedef struct { __mpz_struct _mp_num; __mpz_struct _mp_den; } __mpq_struct;
typedef __mpq_struct mpq_t[1];
typedef __mpq_struct *mpq_ptr;
typedef const __mpq_struct *mpq_srcptr;
#endif
