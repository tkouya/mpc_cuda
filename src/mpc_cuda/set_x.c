/* CUDA-adapted from MPC by tools/cudafy_mpc.py. */
/* mpc_set_x -- Set the real part of a complex number
   (imaginary part equals +0 regardless of rounding mode).

Copyright (C) 2008, 2009, 2010, 2011, 2022, 2023, 2024 INRIA

This file is part of GNU MPC.

GNU MPC is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the
Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

GNU MPC is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
more details.

You should have received a copy of the GNU Lesser General Public License
along with this program. If not, see http://www.gnu.org/licenses/ .
*/

#include "config.h"

#include "mpc-impl.h"
#ifndef MPC_CUDA_MPFR_SET_STUBS
#define MPC_CUDA_MPFR_SET_STUBS
__host__ __device__ static inline int cu_mpfr_set_f (mpfr_ptr r, mpf_srcptr v, mpfr_rnd_t rnd) { (void) r; (void) v; (void) rnd; __builtin_trap (); return 0; }
__host__ __device__ static inline int cu_mpfr_set_q (mpfr_ptr r, mpq_srcptr v, mpfr_rnd_t rnd) { (void) r; (void) v; (void) rnd; __builtin_trap (); return 0; }
__host__ __device__ static inline int cu_mpfr_set_fr (mpfr_ptr r, mpfr_srcptr v, mpfr_rnd_t rnd) { return cu_mpfr_set (r, v, rnd); }
__host__ __device__ static inline int cu_mpfr_set_uj (mpfr_ptr r, uintmax_t v, mpfr_rnd_t rnd) { (void) r; (void) v; (void) rnd; __builtin_trap (); return 0; }
__host__ __device__ static inline int cu_mpfr_set_sj (mpfr_ptr r, intmax_t v, mpfr_rnd_t rnd) { (void) r; (void) v; (void) rnd; __builtin_trap (); return 0; }
#endif


#define MPC_SET_X(real_t, z, real_value, rnd)     \
  {                                                                     \
    int _inex_re, _inex_im;                                             \
    _inex_re = (cu_mpfr_set_ ## real_t) (mpc_realref (z), (real_value), MPC_RND_RE (rnd)); \
    _inex_im = cu_mpfr_set_ui (mpc_imagref (z), 0, MPC_RND_IM (rnd)); \
    return MPC_INEX (_inex_re, _inex_im);                               \
  }

__host__ __device__
int
cu_mpc_set_fr (mpc_ptr a, mpfr_srcptr b, mpc_rnd_t rnd)
   MPC_SET_X (fr, a, b, rnd)

__host__ __device__
int
cu_mpc_set_d (mpc_ptr a, double b, mpc_rnd_t rnd)
   MPC_SET_X (d, a, b, rnd)

__host__ __device__
int
cu_mpc_set_ld (mpc_ptr a, long double b, mpc_rnd_t rnd)
   MPC_SET_X (ld, a, b, rnd)

__host__ __device__
int
cu_mpc_set_ui (mpc_ptr a, unsigned long int b, mpc_rnd_t rnd)
   MPC_SET_X (ui, a, b, rnd)

__host__ __device__
int
cu_mpc_set_si (mpc_ptr a, long int b, mpc_rnd_t rnd)
   MPC_SET_X (si, a, b, rnd)

__host__ __device__
int
cu_mpc_set_z (mpc_ptr a, mpz_srcptr b, mpc_rnd_t rnd)
   MPC_SET_X (z, a, b, rnd)

__host__ __device__
int
cu_mpc_set_q (mpc_ptr a, mpq_srcptr b, mpc_rnd_t rnd)
   MPC_SET_X (q, a, b, rnd)

__host__ __device__
int
cu_mpc_set_f (mpc_ptr a, mpf_srcptr b, mpc_rnd_t rnd)
   MPC_SET_X (f, a, b, rnd)

#ifdef _MPC_H_HAVE_INTMAX_T
__host__ __device__
int
cu_mpc_set_uj (mpc_ptr a, uintmax_t b, mpc_rnd_t rnd)
   MPC_SET_X (uj, a, b, rnd)

__host__ __device__
int
cu_mpc_set_sj (mpc_ptr a, intmax_t b, mpc_rnd_t rnd)
   MPC_SET_X (sj, a, b, rnd)
#endif

#ifdef _MPC_HAVE_COMPLEX_H
__host__ __device__
int
mpc_set_dc (mpc_ptr a, DOUBLE_COMPLEX b, mpc_rnd_t rnd) {
   return cu_mpc_set_d_d (a, creal (b), cimag (b), rnd);
}

__host__ __device__
int
mpc_set_ldc (mpc_ptr a, LONG_DOUBLE_COMPLEX b, mpc_rnd_t rnd) {
   return cu_mpc_set_ld_ld (a, creall (b), cimagl (b), rnd);
}
#endif

__host__ __device__
void
cu_mpc_set_nan (mpc_ptr a) {
   cu_mpfr_set_nan (mpc_realref (a));
   cu_mpfr_set_nan (mpc_imagref (a));
}
