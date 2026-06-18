/* CUDA-adapted from MPFR by tools/cudafy_mpfr.py. */
/* custom interface -- initialize a floating-point number with given
   allocation area

Copyright 2005-2025 Free Software Foundation, Inc.
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

#include "mpfr-impl.h"

#undef cu_mpfr_custom_get_size
__host__ __device__
size_t
cu_mpfr_custom_get_size (mpfr_prec_t prec)
{
  return MPFR_PREC2LIMBS (prec) * MPFR_BYTES_PER_MP_LIMB;
}

#undef cu_mpfr_custom_init
__host__ __device__
void
cu_mpfr_custom_init (void *mantissa, mpfr_prec_t prec)
{
  return ;
}

#undef cu_mpfr_custom_get_significand
__host__ __device__
void *
cu_mpfr_custom_get_significand (mpfr_srcptr x)
{
  return (void*) MPFR_MANT (x);
}

#undef cu_mpfr_custom_get_exp
__host__ __device__
mpfr_exp_t
cu_mpfr_custom_get_exp (mpfr_srcptr x)
{
  return MPFR_EXP (x);
}

#undef cu_mpfr_custom_move
__host__ __device__
void
cu_mpfr_custom_move (mpfr_ptr x, void *new_position)
{
  MPFR_MANT (x) = (mp_limb_t *) new_position;
}

#undef cu_mpfr_custom_init_set
__host__ __device__
void
cu_mpfr_custom_init_set (mpfr_ptr x, int kind, mpfr_exp_t exp,
                     mpfr_prec_t prec, void *mantissa)
{
  mpfr_kind_t t;
  int s;
  mpfr_exp_t e;

  if (kind >= 0)
    {
      t = (mpfr_kind_t) kind;
      s = MPFR_SIGN_POS;
    }
  else
    {
      t = (mpfr_kind_t) -kind;
      s = MPFR_SIGN_NEG;
    }
  MPFR_ASSERTD (t <= MPFR_REGULAR_KIND);
  e = MPFR_LIKELY (t == MPFR_REGULAR_KIND) ? exp :
    MPFR_UNLIKELY (t == MPFR_NAN_KIND) ? MPFR_EXP_NAN :
    MPFR_UNLIKELY (t == MPFR_INF_KIND) ? MPFR_EXP_INF : MPFR_EXP_ZERO;

  MPFR_PREC (x) = prec;
  MPFR_SET_SIGN (x, s);
  MPFR_EXP (x) = e;
  MPFR_MANT (x) = (mp_limb_t*) mantissa;
  return;
}

#undef cu_mpfr_custom_get_kind
__host__ __device__
int
cu_mpfr_custom_get_kind (mpfr_srcptr x)
{
  if (MPFR_LIKELY (!MPFR_IS_SINGULAR (x)))
    return (int) MPFR_REGULAR_KIND * MPFR_INT_SIGN (x);
  if (MPFR_IS_INF (x))
    return (int) MPFR_INF_KIND * MPFR_INT_SIGN (x);
  if (MPFR_IS_NAN (x))
    return (int) MPFR_NAN_KIND;
  MPFR_ASSERTD (MPFR_IS_ZERO (x));
  return (int) MPFR_ZERO_KIND * MPFR_INT_SIGN (x);
}

