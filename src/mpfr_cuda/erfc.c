/* CUDA-adapted from MPFR by tools/cudafy_mpfr.py. */
/* cu_mpfr_erfc -- The Complementary Error Function of a floating-point number

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

#define MPFR_NEED_LONGLONG_H
#include "mpfr-impl.h"

/* erfc(x) = 1 - erf(x) */

/* Put in y an approximation of erfc(x) for large x, using formulae 7.1.23 and
   7.1.24 from Abramowitz and Stegun.
   Returns e such that the error is bounded by 2^e ulp(y),
   or returns 0 in case of underflow.
*/
__host__ __device__
static mpfr_exp_t
mpfr_erfc_asympt (mpfr_ptr y, mpfr_srcptr x)
{
  mpfr_t t, xx, err;
  unsigned long k;
  mpfr_prec_t prec = MPFR_PREC(y);
  mpfr_exp_t exp_err;

  cu_mpfr_init2 (t, prec);
  cu_mpfr_init2 (xx, prec);
  cu_mpfr_init2 (err, 31);
  /* let u = 2^(1-p), and let us represent the error as (1+u)^err
     with a bound for err */
  cu_mpfr_sqr (xx, x, MPFR_RNDD); /* err <= 1 */
  cu_mpfr_ui_div (xx, 1, xx, MPFR_RNDU); /* upper bound for 1/(2x^2), err <= 2 */
  cu_mpfr_div_2ui (xx, xx, 1, MPFR_RNDU); /* exact */
  cu_mpfr_set_ui (t, 1, MPFR_RNDN); /* current term, exact */
  cu_mpfr_set (y, t, MPFR_RNDN);    /* current sum  */
  cu_mpfr_set_ui (err, 0, MPFR_RNDN);
  for (k = 1; ; k++)
    {
      cu_mpfr_mul_ui (t, t, 2 * k - 1, MPFR_RNDU); /* err <= 4k-3 */
      cu_mpfr_mul (t, t, xx, MPFR_RNDU);           /* err <= 4k */
      /* for -1 < x < 1, and |nx| < 1, we have |(1+x)^n| <= 1+7/4|nx|.
         Indeed, for x>=0: log((1+x)^n) = n*log(1+x) <= n*x. Let y=n*x < 1,
         then exp(y) <= 1+7/4*y.
         For x<=0, let x=-x, we can prove by induction that (1-x)^n >= 1-n*x.*/
      cu_mpfr_mul_2si (err, err, MPFR_GET_EXP (y) - MPFR_GET_EXP (t), MPFR_RNDU);
      cu_mpfr_add_ui (err, err, 14 * k, MPFR_RNDU); /* 2^(1-p) * t <= 2 ulp(t) */
      cu_mpfr_div_2si (err, err, MPFR_GET_EXP (y) - MPFR_GET_EXP (t), MPFR_RNDU);
      if (MPFR_GET_EXP (t) + (mpfr_exp_t) prec <= MPFR_GET_EXP (y))
        {
          /* the truncation error is bounded by |t| < ulp(y) */
          cu_mpfr_add_ui (err, err, 1, MPFR_RNDU);
          break;
        }
      if (k & 1)
        cu_mpfr_sub (y, y, t, MPFR_RNDN);
      else
        cu_mpfr_add (y, y, t, MPFR_RNDN);
    }
  /* the error on y is bounded by err*ulp(y) */
  cu_mpfr_sqr (t, x, MPFR_RNDU);             /* rel. err <= 2^(1-p) */
  cu_mpfr_div_2ui (err, err, 3, MPFR_RNDU);  /* err/8 */
  cu_mpfr_add (err, err, t, MPFR_RNDU);      /* err/8 + xx */
  cu_mpfr_mul_2ui (err, err, 3, MPFR_RNDU);  /* err + 8*xx */
  cu_mpfr_exp (t, t, MPFR_RNDU); /* err <= 1/2*ulp(t) + err(x*x)*t
                                <= 1/2*ulp(t)+2*|x*x|*ulp(t)
                                <= (2*|x*x|+1/2)*ulp(t) */
  cu_mpfr_mul (t, t, x, MPFR_RNDN); /* err <= 1/2*ulp(t) + (4*|x*x|+1)*ulp(t)
                                   <= (4*|x*x|+3/2)*ulp(t) */
  cu_mpfr_const_pi (xx, MPFR_RNDZ); /* err <= ulp(Pi) */
  cu_mpfr_sqrt (xx, xx, MPFR_RNDN); /* err <= 1/2*ulp(xx) + ulp(Pi)/2/sqrt(Pi)
                                   <= 3/2*ulp(xx) */
  cu_mpfr_mul (t, t, xx, MPFR_RNDN); /* err <= (8 |xx| + 13/2) * ulp(t) */
  cu_mpfr_div (y, y, t, MPFR_RNDN); /* the relative error on input y is bounded
                                   by (1+u)^err with u = 2^(1-p), that on
                                   t is bounded by (1+u)^(8 |xx| + 13/2),
                                   thus that on output y is bounded by
                                   8 |xx| + 7 + err. */

  if (MPFR_IS_ZERO(y))
    {
      /* If y is zero, most probably we have underflow. We check it directly
         using the fact that erfc(x) <= exp(-x^2)/sqrt(Pi)/x for x >= 0.
         We compute an upper approximation of exp(-x^2)/sqrt(Pi)/x.
      */
      cu_mpfr_sqr (t, x, MPFR_RNDD);    /* t <= x^2 */
      cu_mpfr_neg (t, t, MPFR_RNDU);    /* -x^2 <= t */
      cu_mpfr_exp (t, t, MPFR_RNDU);    /* exp(-x^2) <= t */
      cu_mpfr_const_pi (xx, MPFR_RNDD); /* xx <= sqrt(Pi), cached */
      cu_mpfr_mul (xx, xx, x, MPFR_RNDD); /* xx <= sqrt(Pi)*x */
      cu_mpfr_div (y, t, xx, MPFR_RNDN); /* if y is zero, this means that the upper
                                        approximation of exp(-x^2)/sqrt(Pi)/x
                                        is nearer from 0 than from 2^(-emin-1),
                                        thus we have underflow. */
      exp_err = 0;
    }
  else
    {
      cu_mpfr_add_ui (err, err, 7, MPFR_RNDU);
      exp_err = MPFR_GET_EXP (err);
    }

  cu_mpfr_clear (t);
  cu_mpfr_clear (xx);
  cu_mpfr_clear (err);
  return exp_err;
}

__host__ __device__
int
cu_mpfr_erfc (mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd)
{
  int inex;
  mpfr_t tmp;
  mpfr_exp_t te, err;
  mpfr_prec_t prec;
  mpfr_exp_t emin = cu_mpfr_get_emin ();
  MPFR_SAVE_EXPO_DECL (expo);
  MPFR_ZIV_DECL (loop);

  MPFR_LOG_FUNC
    (("x[%Pd]=%.*Rg rnd=%d", cu_mpfr_get_prec (x), mpfr_log_prec, x, rnd),
     ("y[%Pd]=%.*Rg inexact=%d", cu_mpfr_get_prec (y), mpfr_log_prec, y, inex));

  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (x)))
    {
      if (MPFR_IS_NAN (x))
        {
          MPFR_SET_NAN (y);
          MPFR_RET_NAN;
        }
      /* erfc(+inf) = 0+, erfc(-inf) = 2 erfc (0) = 1 */
      else if (MPFR_IS_INF (x))
        return cu_mpfr_set_ui (y, MPFR_IS_POS (x) ? 0 : 2, rnd);
      else
        return cu_mpfr_set_ui (y, 1, rnd);
    }

  if (MPFR_IS_POS (x))
    {
      /* by default, emin = 1-2^30, thus the smallest representable
         number is 1/2*2^emin = 2^(-2^30):
         for x >= 27282, erfc(x) < 2^(-2^30-1), and
         for x >= 1787897414, erfc(x) < 2^(-2^62-1).
      */
      if ((emin >= -1073741823 && cu_mpfr_cmp_ui (x, 27282) >= 0) ||
          cu_mpfr_cmp_ui (x, 1787897414) >= 0)
        {
          /* May be incorrect if MPFR_EMAX_MAX >= 2^62. */
          MPFR_STAT_STATIC_ASSERT ((MPFR_EMAX_MAX >> 31) >> 31 == 0);
          return cu_mpfr_underflow (y, (rnd == MPFR_RNDN) ? MPFR_RNDZ : rnd, 1);
        }
    }

  /* Init stuff */
  MPFR_SAVE_EXPO_MARK (expo);

  if (MPFR_IS_NEG (x))
    {
      mpfr_exp_t e = MPFR_EXP(x);
      /* For x < 0 going to -infinity, erfc(x) tends to 2 by below.
         More precisely, we have 2 + 1/sqrt(Pi)/x/exp(x^2) < erfc(x) < 2.
         Thus log2 |2 - erfc(x)| <= -log2|x| - x^2 / log(2).
         If |2 - erfc(x)| < 2^(-PREC(y)) then the result is either 2 or
         nextbelow(2).
         For x <= -27282, -log2|x| - x^2 / log(2) <= -2^30.
      */
      if ((MPFR_PREC(y) <= 7 && e >= 2) ||  /* x <= -2 */
          (MPFR_PREC(y) <= 25 && e >= 3) || /* x <= -4 */
          (MPFR_PREC(y) <= 120 && cu_mpfr_cmp_si (x, -9) <= 0) ||
          cu_mpfr_cmp_si (x, -27282) <= 0)
        {
        near_two:
          cu_mpfr_set_ui (y, 2, MPFR_RNDN);
          MPFR_SET_INEXFLAG ();
          if (rnd == MPFR_RNDZ || rnd == MPFR_RNDD)
            {
              cu_mpfr_nextbelow (y);
              inex = -1;
            }
          else
            inex = 1;
          goto end;
        }
      else if (e >= 3) /* more accurate test */
        {
          mpfr_t t, u;
          int near_2;
          cu_mpfr_init2 (t, 32);
          cu_mpfr_init2 (u, 32);
          /* the following is 1/log(2) rounded to zero on 32 bits */
          cu_mpfr_set_str_binary (t, "1.0111000101010100011101100101001");
          cu_mpfr_sqr (u, x, MPFR_RNDZ);
          cu_mpfr_mul (t, t, u, MPFR_RNDZ); /* t <= x^2/log(2) */
          cu_mpfr_neg (u, x, MPFR_RNDZ); /* 0 <= u <= |x| */
          cu_mpfr_log2 (u, u, MPFR_RNDZ); /* u <= log2(|x|) */
          cu_mpfr_add (t, t, u, MPFR_RNDZ); /* t <= log2|x| + x^2 / log(2) */
          /* Taking into account that mpfr_exp_t >= mpfr_prec_t */
          mpfr_set_exp_t (u, MPFR_PREC (y), MPFR_RNDU);
          near_2 = cu_mpfr_cmp (t, u) >= 0;  /* 1 if PREC(y) <= u <= t <= ... */
          cu_mpfr_clear (t);
          cu_mpfr_clear (u);
          if (near_2)
            goto near_two;
        }
    }

  /* erfc(x) ~ 1, with error < 2^(EXP(x)+1) */
  MPFR_FAST_COMPUTE_IF_SMALL_INPUT (y, cu___gmpfr_one, - MPFR_GET_EXP (x) - 1,
                                    0, MPFR_IS_NEG (x),
                                    rnd, inex = _inexact; goto end);

  prec = MPFR_PREC (y) + MPFR_INT_CEIL_LOG2 (MPFR_PREC (y)) + 3;
  if (MPFR_GET_EXP (x) > 0)
    prec += 2 * MPFR_GET_EXP(x);

  cu_mpfr_init2 (tmp, prec);

  MPFR_ZIV_INIT (loop, prec);            /* Initialize the ZivLoop controller */
  for (;;)                               /* Infinite loop */
    {
      /* use asymptotic formula only whenever x^2 >= p*log(2),
         otherwise it will not converge */
      if (MPFR_IS_POS (x) &&
          2 * MPFR_GET_EXP (x) - 2 >= MPFR_INT_CEIL_LOG2 (prec))
        /* we have x^2 >= p in that case */
        {
          err = mpfr_erfc_asympt (tmp, x);
          if (err == 0) /* underflow case */
            {
              cu_mpfr_clear (tmp);
              MPFR_SAVE_EXPO_FREE (expo);
              return cu_mpfr_underflow (y, (rnd == MPFR_RNDN) ? MPFR_RNDZ : rnd, 1);
            }
        }
      else
        {
          cu_mpfr_erf (tmp, x, MPFR_RNDN);
          MPFR_ASSERTD (!MPFR_IS_SINGULAR (tmp)); /* FIXME: 0 only for x=0 ? */
          te = MPFR_GET_EXP (tmp);
          cu_mpfr_ui_sub (tmp, 1, tmp, MPFR_RNDN);
          /* See error analysis in algorithms.tex for details */
          if (MPFR_IS_ZERO (tmp))
            {
              prec *= 2;
              err = prec; /* ensures MPFR_CAN_ROUND fails */
            }
          else
            err = MAX (te - MPFR_GET_EXP (tmp), 0) + 1;
        }
      if (MPFR_LIKELY (MPFR_CAN_ROUND (tmp, prec - err, MPFR_PREC (y), rnd)))
        break;
      MPFR_ZIV_NEXT (loop, prec);        /* Increase used precision */
      cu_mpfr_set_prec (tmp, prec);
    }
  MPFR_ZIV_FREE (loop);                  /* Free the ZivLoop Controller */

  inex = cu_mpfr_set (y, tmp, rnd);    /* Set y to the computed value */
  cu_mpfr_clear (tmp);

 end:
  MPFR_SAVE_EXPO_FREE (expo);
  return cu_mpfr_check_range (y, inex, rnd);
}
