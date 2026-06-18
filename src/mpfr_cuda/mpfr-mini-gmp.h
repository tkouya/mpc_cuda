/* CUDA-adapted from MPFR by tools/cudafy_mpfr.py. */
/* mpfr-mini-gmp.h -- Interface header for mini-gmp.

Copyright 2014-2025 Free Software Foundation, Inc.
Contributed by the Pascaline and Caramba projects, INRIA.

This file is part of the GNU MPFR Library.

The GNU MPFR Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The GNU MPFR Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MPFR Library; see the file COPYING.LESSER.
If not, see <https://www.gnu.org/licenses/>. */

#ifndef mp_limb_signed_t
typedef long mp_limb_signed_t;
#endif

#ifndef __gmp_allocate_func
#define __gmp_allocate_func gmp_default_alloc
#define __gmp_reallocate_func gmp_default_realloc
#define __gmp_free_func gmp_default_free
#endif

#ifndef gmp_randstate_t
typedef long int __gmp_randstate_struct;
typedef __gmp_randstate_struct gmp_randstate_t[1];
#endif

#ifndef gmp_randinit_default
#define WANT_gmp_randinit_default
__host__ __device__ void gmp_randinit_default (gmp_randstate_t);
#endif

#ifndef gmp_randseed_ui
#define WANT_gmp_randseed_ui
__host__ __device__ void gmp_randseed_ui (gmp_randstate_t, unsigned long int);
#endif

#ifndef gmp_randclear
#define WANT_gmp_randclear
__host__ __device__ void gmp_randclear (gmp_randstate_t);
#endif

#ifndef gmp_randinit_set
#define WANT_gmp_randinit_set
__host__ __device__ void gmp_randinit_set (gmp_randstate_t, gmp_randstate_t);
#endif

#ifndef cu_mpn_scan1
#define WANT_mpn_scan1
__host__ __device__ mp_bitcnt_t cu_mpn_scan1 (const mp_limb_t *, mp_bitcnt_t);
#endif

#ifndef cu_mpz_perfect_square_p
#define WANT_mpz_perfect_square_p
__host__ __device__ int cu_mpz_perfect_square_p (const mpz_t);
#endif

#ifndef cu_mpz_addmul_ui
#define WANT_mpz_addmul_ui
__host__ __device__ void cu_mpz_addmul_ui (mpz_t, const mpz_t, unsigned long);
#endif

#ifndef mpn_divrem_1
#define WANT_mpn_divrem_1
__host__ __device__ mp_limb_t mpn_divrem_1 (mp_limb_t*, mp_size_t, mp_limb_t*, mp_size_t,
                        mp_limb_t);
#endif

#ifndef mpz_urandomb
#define WANT_mpz_urandomb
__host__ __device__ void mpz_urandomb (mpz_t, gmp_randstate_t, mp_bitcnt_t);
#endif

#ifndef cu_mpn_zero
#define WANT_mpn_zero
__host__ __device__ void cu_mpn_zero (mp_limb_t *, mp_size_t);
#endif

#ifndef cu_mpn_popcount
#define WANT_mpn_popcount
__host__ __device__ mp_bitcnt_t cu_mpn_popcount (const mp_limb_t *, mp_size_t);
#endif

#ifndef gmp_urandomm_ui
#define WANT_gmp_urandomm_ui
__host__ __device__ unsigned long gmp_urandomm_ui (gmp_randstate_t state, unsigned long n);
#endif

#ifndef gmp_urandomb_ui
#define WANT_gmp_urandomb_ui
__host__ __device__ unsigned long gmp_urandomb_ui (gmp_randstate_t state, unsigned long n);
#endif

#ifndef mpn_divrem
#define WANT_mpn_divrem
__host__ __device__ mp_limb_t mpn_divrem (mp_limb_t *, mp_size_t, mp_limb_t *,
                      mp_size_t, const mp_limb_t *, mp_size_t);
#endif

#ifndef cu_mpz_submul
#define WANT_mpz_submul
__host__ __device__ void cu_mpz_submul (mpz_t, const mpz_t, const mpz_t);
#endif

#ifndef cu_mpz_addmul
#define WANT_mpz_addmul
__host__ __device__ void cu_mpz_addmul (mpz_t, const mpz_t, const mpz_t);
#endif

#ifndef mpn_tdiv_qr
#define WANT_mpn_tdiv_qr
__host__ __device__ void mpn_tdiv_qr (mp_limb_t *, mp_limb_t *, mp_size_t,
                  const mp_limb_t *, mp_size_t, const mp_limb_t *, mp_size_t);
#endif

#ifndef mpz_rrandomb
#define mpz_rrandomb mpz_urandomb
#endif

#ifndef mpz_dump
__host__ __device__ void mpz_dump (mpz_t);
#endif
