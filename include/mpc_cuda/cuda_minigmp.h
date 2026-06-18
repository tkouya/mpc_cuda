/* Auto-generated from mini-gmp.h by tools/cudafy_minigmp.py.  Do not edit by hand. */
/* mini-gmp, a minimalistic implementation of a GNU GMP subset.

Copyright 2011-2015, 2017, 2019-2021 Free Software Foundation, Inc.

This file is part of the GNU MP Library.

The GNU MP Library is free software; you can redistribute it and/or modify
it under the terms of either:

  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.

or

  * the GNU General Public License as published by the Free Software
    Foundation; either version 2 of the License, or (at your option) any
    later version.

or both in parallel, as here.

The GNU MP Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received copies of the GNU General Public License and the
GNU Lesser General Public License along with the GNU MP Library.  If not,
see https://www.gnu.org/licenses/.  */

/* About mini-gmp: This is a minimal implementation of a subset of the
   GMP interface. It is intended for inclusion into applications which
   have modest bignums needs, as a fallback when the real GMP library
   is not installed.

   This file defines the public interface. */

#ifndef __MINI_GMP_H__
#define __MINI_GMP_H__

/* For size_t */
#include <stddef.h>

#if defined (__cplusplus)
extern "C" {
#endif

void mp_set_memory_functions (void *(*) (size_t),
			      void *(*) (void *, size_t, size_t),
			      void (*) (void *, size_t));

void mp_get_memory_functions (void *(**) (size_t),
			      void *(**) (void *, size_t, size_t),
			      void (**) (void *, size_t));

#ifndef MINI_GMP_LIMB_TYPE
#define MINI_GMP_LIMB_TYPE long
#endif

typedef unsigned MINI_GMP_LIMB_TYPE mp_limb_t;
typedef long mp_size_t;
typedef unsigned long mp_bitcnt_t;

typedef mp_limb_t *mp_ptr;
typedef const mp_limb_t *mp_srcptr;

typedef struct
{
  int _mp_alloc;		/* Number of *limbs* allocated and pointed
				   to by the _mp_d field.  */
  int _mp_size;			/* abs(_mp_size) is the number of limbs the
				   last field points to.  If _mp_size is
				   negative this is a negative number.  */
  mp_limb_t *_mp_d;		/* Pointer to the limbs.  */
} __mpz_struct;

typedef __mpz_struct mpz_t[1];

typedef __mpz_struct *mpz_ptr;
typedef const __mpz_struct *mpz_srcptr;

extern __device__ __managed__ int cu_mp_bits_per_limb;

__host__ __device__ void cu_mpn_copyi (mp_ptr, mp_srcptr, mp_size_t);
__host__ __device__ void cu_mpn_copyd (mp_ptr, mp_srcptr, mp_size_t);
__host__ __device__ void cu_mpn_zero (mp_ptr, mp_size_t);

__host__ __device__ int cu_mpn_cmp (mp_srcptr, mp_srcptr, mp_size_t);
__host__ __device__ int cu_mpn_zero_p (mp_srcptr, mp_size_t);

__host__ __device__ mp_limb_t cu_mpn_add_1 (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);
__host__ __device__ mp_limb_t cu_mpn_add_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
__host__ __device__ mp_limb_t cu_mpn_add (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);

__host__ __device__ mp_limb_t cu_mpn_sub_1 (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);
__host__ __device__ mp_limb_t cu_mpn_sub_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
__host__ __device__ mp_limb_t cu_mpn_sub (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);

__host__ __device__ mp_limb_t cu_mpn_mul_1 (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);
__host__ __device__ mp_limb_t cu_mpn_addmul_1 (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);
__host__ __device__ mp_limb_t cu_mpn_submul_1 (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);

__host__ __device__ mp_limb_t cu_mpn_mul (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);
__host__ __device__ void cu_mpn_mul_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
__host__ __device__ void cu_mpn_sqr (mp_ptr, mp_srcptr, mp_size_t);
__host__ __device__ int cu_mpn_perfect_square_p (mp_srcptr, mp_size_t);
__host__ __device__ mp_size_t cu_mpn_sqrtrem (mp_ptr, mp_ptr, mp_srcptr, mp_size_t);

__host__ __device__ mp_limb_t cu_mpn_lshift (mp_ptr, mp_srcptr, mp_size_t, unsigned int);
__host__ __device__ mp_limb_t cu_mpn_rshift (mp_ptr, mp_srcptr, mp_size_t, unsigned int);

__host__ __device__ mp_bitcnt_t cu_mpn_scan0 (mp_srcptr, mp_bitcnt_t);
__host__ __device__ mp_bitcnt_t cu_mpn_scan1 (mp_srcptr, mp_bitcnt_t);

__host__ __device__ void cu_mpn_com (mp_ptr, mp_srcptr, mp_size_t);
__host__ __device__ mp_limb_t cu_mpn_neg (mp_ptr, mp_srcptr, mp_size_t);

__host__ __device__ mp_bitcnt_t cu_mpn_popcount (mp_srcptr, mp_size_t);

__host__ __device__ mp_limb_t cu_mpn_invert_3by2 (mp_limb_t, mp_limb_t);
#define mpn_invert_limb(x) cu_mpn_invert_3by2 ((x), 0)

__host__ __device__ size_t cu_mpn_get_str (unsigned char *, int, mp_ptr, mp_size_t);
__host__ __device__ mp_size_t cu_mpn_set_str (mp_ptr, const unsigned char *, size_t, int);

__host__ __device__ void cu_mpz_init (mpz_t);
__host__ __device__ void cu_mpz_init2 (mpz_t, mp_bitcnt_t);
__host__ __device__ void cu_mpz_clear (mpz_t);

#define mpz_odd_p(z)   (((z)->_mp_size != 0) & (int) (z)->_mp_d[0])
#define mpz_even_p(z)  (! mpz_odd_p (z))

__host__ __device__ int cu_mpz_sgn (const mpz_t);
__host__ __device__ int cu_mpz_cmp_si (const mpz_t, long);
__host__ __device__ int cu_mpz_cmp_ui (const mpz_t, unsigned long);
__host__ __device__ int cu_mpz_cmp (const mpz_t, const mpz_t);
__host__ __device__ int cu_mpz_cmpabs_ui (const mpz_t, unsigned long);
__host__ __device__ int cu_mpz_cmpabs (const mpz_t, const mpz_t);
__host__ __device__ int cu_mpz_cmp_d (const mpz_t, double);
__host__ __device__ int cu_mpz_cmpabs_d (const mpz_t, double);

__host__ __device__ void cu_mpz_abs (mpz_t, const mpz_t);
__host__ __device__ void cu_mpz_neg (mpz_t, const mpz_t);
__host__ __device__ void cu_mpz_swap (mpz_t, mpz_t);

__host__ __device__ void cu_mpz_add_ui (mpz_t, const mpz_t, unsigned long);
__host__ __device__ void cu_mpz_add (mpz_t, const mpz_t, const mpz_t);
__host__ __device__ void cu_mpz_sub_ui (mpz_t, const mpz_t, unsigned long);
__host__ __device__ void cu_mpz_ui_sub (mpz_t, unsigned long, const mpz_t);
__host__ __device__ void cu_mpz_sub (mpz_t, const mpz_t, const mpz_t);

__host__ __device__ void cu_mpz_mul_si (mpz_t, const mpz_t, long int);
__host__ __device__ void cu_mpz_mul_ui (mpz_t, const mpz_t, unsigned long int);
__host__ __device__ void cu_mpz_mul (mpz_t, const mpz_t, const mpz_t);
__host__ __device__ void cu_mpz_mul_2exp (mpz_t, const mpz_t, mp_bitcnt_t);
__host__ __device__ void cu_mpz_addmul_ui (mpz_t, const mpz_t, unsigned long int);
__host__ __device__ void cu_mpz_addmul (mpz_t, const mpz_t, const mpz_t);
__host__ __device__ void cu_mpz_submul_ui (mpz_t, const mpz_t, unsigned long int);
__host__ __device__ void cu_mpz_submul (mpz_t, const mpz_t, const mpz_t);

__host__ __device__ void cu_mpz_cdiv_qr (mpz_t, mpz_t, const mpz_t, const mpz_t);
__host__ __device__ void cu_mpz_fdiv_qr (mpz_t, mpz_t, const mpz_t, const mpz_t);
__host__ __device__ void cu_mpz_tdiv_qr (mpz_t, mpz_t, const mpz_t, const mpz_t);
__host__ __device__ void cu_mpz_cdiv_q (mpz_t, const mpz_t, const mpz_t);
__host__ __device__ void cu_mpz_fdiv_q (mpz_t, const mpz_t, const mpz_t);
__host__ __device__ void cu_mpz_tdiv_q (mpz_t, const mpz_t, const mpz_t);
__host__ __device__ void cu_mpz_cdiv_r (mpz_t, const mpz_t, const mpz_t);
__host__ __device__ void cu_mpz_fdiv_r (mpz_t, const mpz_t, const mpz_t);
__host__ __device__ void cu_mpz_tdiv_r (mpz_t, const mpz_t, const mpz_t);

__host__ __device__ void cu_mpz_cdiv_q_2exp (mpz_t, const mpz_t, mp_bitcnt_t);
__host__ __device__ void cu_mpz_fdiv_q_2exp (mpz_t, const mpz_t, mp_bitcnt_t);
__host__ __device__ void cu_mpz_tdiv_q_2exp (mpz_t, const mpz_t, mp_bitcnt_t);
__host__ __device__ void cu_mpz_cdiv_r_2exp (mpz_t, const mpz_t, mp_bitcnt_t);
__host__ __device__ void cu_mpz_fdiv_r_2exp (mpz_t, const mpz_t, mp_bitcnt_t);
__host__ __device__ void cu_mpz_tdiv_r_2exp (mpz_t, const mpz_t, mp_bitcnt_t);

__host__ __device__ void cu_mpz_mod (mpz_t, const mpz_t, const mpz_t);

__host__ __device__ void cu_mpz_divexact (mpz_t, const mpz_t, const mpz_t);

__host__ __device__ int cu_mpz_divisible_p (const mpz_t, const mpz_t);
__host__ __device__ int cu_mpz_congruent_p (const mpz_t, const mpz_t, const mpz_t);

__host__ __device__ unsigned long cu_mpz_cdiv_qr_ui (mpz_t, mpz_t, const mpz_t, unsigned long);
__host__ __device__ unsigned long cu_mpz_fdiv_qr_ui (mpz_t, mpz_t, const mpz_t, unsigned long);
__host__ __device__ unsigned long cu_mpz_tdiv_qr_ui (mpz_t, mpz_t, const mpz_t, unsigned long);
__host__ __device__ unsigned long cu_mpz_cdiv_q_ui (mpz_t, const mpz_t, unsigned long);
__host__ __device__ unsigned long cu_mpz_fdiv_q_ui (mpz_t, const mpz_t, unsigned long);
__host__ __device__ unsigned long cu_mpz_tdiv_q_ui (mpz_t, const mpz_t, unsigned long);
__host__ __device__ unsigned long cu_mpz_cdiv_r_ui (mpz_t, const mpz_t, unsigned long);
__host__ __device__ unsigned long cu_mpz_fdiv_r_ui (mpz_t, const mpz_t, unsigned long);
__host__ __device__ unsigned long cu_mpz_tdiv_r_ui (mpz_t, const mpz_t, unsigned long);
__host__ __device__ unsigned long cu_mpz_cdiv_ui (const mpz_t, unsigned long);
__host__ __device__ unsigned long cu_mpz_fdiv_ui (const mpz_t, unsigned long);
__host__ __device__ unsigned long cu_mpz_tdiv_ui (const mpz_t, unsigned long);

__host__ __device__ unsigned long cu_mpz_mod_ui (mpz_t, const mpz_t, unsigned long);

__host__ __device__ void cu_mpz_divexact_ui (mpz_t, const mpz_t, unsigned long);

__host__ __device__ int cu_mpz_divisible_ui_p (const mpz_t, unsigned long);

__host__ __device__ unsigned long cu_mpz_gcd_ui (mpz_t, const mpz_t, unsigned long);
__host__ __device__ void cu_mpz_gcd (mpz_t, const mpz_t, const mpz_t);
__host__ __device__ void cu_mpz_gcdext (mpz_t, mpz_t, mpz_t, const mpz_t, const mpz_t);
__host__ __device__ void cu_mpz_lcm_ui (mpz_t, const mpz_t, unsigned long);
__host__ __device__ void cu_mpz_lcm (mpz_t, const mpz_t, const mpz_t);
__host__ __device__ int cu_mpz_invert (mpz_t, const mpz_t, const mpz_t);

__host__ __device__ void cu_mpz_sqrtrem (mpz_t, mpz_t, const mpz_t);
__host__ __device__ void cu_mpz_sqrt (mpz_t, const mpz_t);
__host__ __device__ int cu_mpz_perfect_square_p (const mpz_t);

__host__ __device__ void cu_mpz_pow_ui (mpz_t, const mpz_t, unsigned long);
__host__ __device__ void cu_mpz_ui_pow_ui (mpz_t, unsigned long, unsigned long);
__host__ __device__ void cu_mpz_powm (mpz_t, const mpz_t, const mpz_t, const mpz_t);
__host__ __device__ void cu_mpz_powm_ui (mpz_t, const mpz_t, unsigned long, const mpz_t);

__host__ __device__ void cu_mpz_rootrem (mpz_t, mpz_t, const mpz_t, unsigned long);
__host__ __device__ int cu_mpz_root (mpz_t, const mpz_t, unsigned long);

__host__ __device__ void cu_mpz_fac_ui (mpz_t, unsigned long);
__host__ __device__ void cu_mpz_2fac_ui (mpz_t, unsigned long);
__host__ __device__ void cu_mpz_mfac_uiui (mpz_t, unsigned long, unsigned long);
__host__ __device__ void cu_mpz_bin_uiui (mpz_t, unsigned long, unsigned long);

__host__ __device__ int cu_mpz_probab_prime_p (const mpz_t, int);

__host__ __device__ int cu_mpz_tstbit (const mpz_t, mp_bitcnt_t);
__host__ __device__ void cu_mpz_setbit (mpz_t, mp_bitcnt_t);
__host__ __device__ void cu_mpz_clrbit (mpz_t, mp_bitcnt_t);
__host__ __device__ void cu_mpz_combit (mpz_t, mp_bitcnt_t);

__host__ __device__ void cu_mpz_com (mpz_t, const mpz_t);
__host__ __device__ void cu_mpz_and (mpz_t, const mpz_t, const mpz_t);
__host__ __device__ void cu_mpz_ior (mpz_t, const mpz_t, const mpz_t);
__host__ __device__ void cu_mpz_xor (mpz_t, const mpz_t, const mpz_t);

__host__ __device__ mp_bitcnt_t cu_mpz_popcount (const mpz_t);
__host__ __device__ mp_bitcnt_t cu_mpz_hamdist (const mpz_t, const mpz_t);
__host__ __device__ mp_bitcnt_t cu_mpz_scan0 (const mpz_t, mp_bitcnt_t);
__host__ __device__ mp_bitcnt_t cu_mpz_scan1 (const mpz_t, mp_bitcnt_t);

__host__ __device__ int cu_mpz_fits_slong_p (const mpz_t);
__host__ __device__ int cu_mpz_fits_ulong_p (const mpz_t);
__host__ __device__ int cu_mpz_fits_sint_p (const mpz_t);
__host__ __device__ int cu_mpz_fits_uint_p (const mpz_t);
__host__ __device__ int cu_mpz_fits_sshort_p (const mpz_t);
__host__ __device__ int cu_mpz_fits_ushort_p (const mpz_t);
__host__ __device__ long int cu_mpz_get_si (const mpz_t);
__host__ __device__ unsigned long int cu_mpz_get_ui (const mpz_t);
__host__ __device__ double cu_mpz_get_d (const mpz_t);
__host__ __device__ size_t cu_mpz_size (const mpz_t);
__host__ __device__ mp_limb_t cu_mpz_getlimbn (const mpz_t, mp_size_t);

__host__ __device__ void cu_mpz_realloc2 (mpz_t, mp_bitcnt_t);
__host__ __device__ mp_srcptr cu_mpz_limbs_read (mpz_srcptr);
__host__ __device__ mp_ptr cu_mpz_limbs_modify (mpz_t, mp_size_t);
__host__ __device__ mp_ptr cu_mpz_limbs_write (mpz_t, mp_size_t);
__host__ __device__ void cu_mpz_limbs_finish (mpz_t, mp_size_t);
__host__ __device__ mpz_srcptr cu_mpz_roinit_n (mpz_t, mp_srcptr, mp_size_t);

#define MPZ_ROINIT_N(xp, xs) {{0, (xs),(xp) }}

__host__ __device__ void cu_mpz_set_si (mpz_t, signed long int);
__host__ __device__ void cu_mpz_set_ui (mpz_t, unsigned long int);
__host__ __device__ void cu_mpz_set (mpz_t, const mpz_t);
__host__ __device__ void cu_mpz_set_d (mpz_t, double);

__host__ __device__ void cu_mpz_init_set_si (mpz_t, signed long int);
__host__ __device__ void cu_mpz_init_set_ui (mpz_t, unsigned long int);
__host__ __device__ void cu_mpz_init_set (mpz_t, const mpz_t);
__host__ __device__ void cu_mpz_init_set_d (mpz_t, double);

__host__ __device__ size_t cu_mpz_sizeinbase (const mpz_t, int);
__host__ __device__ char *cu_mpz_get_str (char *, int, const mpz_t);
int cu_mpz_set_str (mpz_t, const char *, int);
int cu_mpz_init_set_str (mpz_t, const char *, int);

/* This long list taken from gmp.h. */
/* For reference, "defined(EOF)" cannot be used here.  In g++ 2.95.4,
   <iostream> defines EOF but not FILE.  */
#if defined (FILE)                                              \
  || defined (H_STDIO)                                          \
  || defined (_H_STDIO)               /* AIX */                 \
  || defined (_STDIO_H)               /* glibc, Sun, SCO */     \
  || defined (_STDIO_H_)              /* BSD, OSF */            \
  || defined (__STDIO_H)              /* Borland */             \
  || defined (__STDIO_H__)            /* IRIX */                \
  || defined (_STDIO_INCLUDED)        /* HPUX */                \
  || defined (__dj_include_stdio_h_)  /* DJGPP */               \
  || defined (_FILE_DEFINED)          /* Microsoft */           \
  || defined (__STDIO__)              /* Apple MPW MrC */       \
  || defined (_MSL_STDIO_H)           /* Metrowerks */          \
  || defined (_STDIO_H_INCLUDED)      /* QNX4 */		\
  || defined (_ISO_STDIO_ISO_H)       /* Sun C++ */		\
  || defined (__STDIO_LOADED)         /* VMS */			\
  || defined (_STDIO)                 /* HPE NonStop */         \
  || defined (__DEFINED_FILE)         /* musl */
size_t cu_mpz_out_str (FILE *, int, const mpz_t);
#endif

__host__ __device__ void cu_mpz_import (mpz_t, size_t, int, size_t, int, size_t, const void *);
__host__ __device__ void *cu_mpz_export (void *, size_t *, int, size_t, int, size_t, const mpz_t);

#if defined (__cplusplus)
}
#endif

/* ---- per-thread bump arena (see tools/cudafy_minigmp.py preamble) ---- */
extern __managed__ char   *mpc_cuda_arena_base;
extern __managed__ size_t  mpc_cuda_arena_slab;
extern __managed__ size_t *mpc_cuda_arena_top;
__host__ __device__ void *mpc_cuda_dev_alloc (size_t);
__host__ __device__ void  mpc_cuda_dev_free (void *);
__host__ __device__ void *mpc_cuda_dev_realloc (void *, size_t, size_t);
__device__ void mpc_cuda_arena_reset (void);

#endif /* __MINI_GMP_H__ */
