/* CUDA-adapted from MPFR by tools/cudafy_mpfr.py. */
/* cu_mpfr_log10 -- logarithm in base 10.

Copyright 2001-2025 Free Software Foundation, Inc.
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

#define MPFR_NEED_LONGLONG_H
#include "mpfr-impl.h"

 /* The computation of r=log10(a)

    r=log10(a)=log(a)/log(10)
 */

__host__ __device__
int
cu_mpfr_log10 (mpfr_ptr r, mpfr_srcptr a, mpfr_rnd_t rnd_mode)
{
  int inexact;
  MPFR_SAVE_EXPO_DECL (expo);

  MPFR_LOG_FUNC
    (("a[%Pd]=%.*Rg rnd=%d", cu_mpfr_get_prec (a), mpfr_log_prec, a, rnd_mode),
     ("r[%Pd]=%.*Rg inexact=%d",
      cu_mpfr_get_prec (r), mpfr_log_prec, r, inexact));

  /* If a is NaN, the result is NaN */
  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (a)))
    {
      if (MPFR_IS_NAN (a))
        {
          MPFR_SET_NAN (r);
          MPFR_RET_NAN;
        }
      /* check for infinity before zero */
      else if (MPFR_IS_INF (a))
        {
          if (MPFR_IS_NEG (a))
            /* log10(-Inf) = NaN */
            {
              MPFR_SET_NAN (r);
              MPFR_RET_NAN;
            }
          else /* log10(+Inf) = +Inf */
            {
              MPFR_SET_INF (r);
              MPFR_SET_POS (r);
              MPFR_RET (0); /* exact */
            }
        }
      else /* a = 0 */
        {
          MPFR_ASSERTD (MPFR_IS_ZERO (a));
          MPFR_SET_INF (r);
          MPFR_SET_NEG (r);
          MPFR_SET_DIVBY0 ();
          MPFR_RET (0); /* log10(0) is an exact -infinity */
        }
    }

  /* If a is negative, the result is NaN */
  if (MPFR_UNLIKELY (MPFR_IS_NEG (a)))
    {
      MPFR_SET_NAN (r);
      MPFR_RET_NAN;
    }

  /* If a is 1, the result is 0 */
  if (cu_mpfr_cmp_ui (a, 1) == 0)
    {
      MPFR_SET_ZERO (r);
      MPFR_SET_POS (r);
      MPFR_RET (0); /* result is exact */
    }

  MPFR_SAVE_EXPO_MARK (expo);

  /* General case */
  {
    /* Declaration of the intermediary variable */
    mpfr_t t, tt;
    MPFR_ZIV_DECL (loop);
    /* Declaration of the size variable */
    mpfr_prec_t Ny = MPFR_PREC(r);   /* Precision of output variable */
    mpfr_prec_t Nt;        /* Precision of the intermediary variable */
    mpfr_exp_t  err;                           /* Precision of error */

    /* compute the precision of intermediary variable */
    /* the optimal number of bits : see algorithms.tex */
    Nt = Ny + 4 + MPFR_INT_CEIL_LOG2 (Ny);

    /* initialize of intermediary variables */
    cu_mpfr_init2 (t, Nt);
    cu_mpfr_init2 (tt, Nt);

    /* First computation of log10 */
    MPFR_ZIV_INIT (loop, Nt);
    for (;;)
      {
        /* compute log10 */
        cu_mpfr_set_ui (t, 10, MPFR_RNDN);   /* 10 */
        cu_mpfr_log (t, t, MPFR_RNDD);       /* log(10) */
        cu_mpfr_log (tt, a, MPFR_RNDN);      /* log(a) */
        cu_mpfr_div (t, tt, t, MPFR_RNDN);   /* log(a)/log(10) */

        /* estimation of the error */
        err = Nt - 4;
        if (MPFR_LIKELY (MPFR_CAN_ROUND (t, err, Ny, rnd_mode)))
          break;

        /* log10(10^n) is exact:
           FIXME: Can we have 10^n exactly representable as a mpfr_t
           but n can't fit an unsigned long? */
        if (MPFR_IS_POS (t)
            && cu_mpfr_integer_p (t) && cu_mpfr_fits_ulong_p (t, MPFR_RNDN)
            && !cu_mpfr_ui_pow_ui (tt, 10, cu_mpfr_get_ui (t, MPFR_RNDN), MPFR_RNDN)
            && cu_mpfr_equal_p (a, tt))
          break;

        /* actualization of the precision */
        MPFR_ZIV_NEXT (loop, Nt);
        cu_mpfr_set_prec (t, Nt);
        cu_mpfr_set_prec (tt, Nt);
      }
    MPFR_ZIV_FREE (loop);

    inexact = cu_mpfr_set (r, t, rnd_mode);

    cu_mpfr_clear (t);
    cu_mpfr_clear (tt);
  }

  MPFR_SAVE_EXPO_FREE (expo);
  return cu_mpfr_check_range (r, inexact, rnd_mode);
}
