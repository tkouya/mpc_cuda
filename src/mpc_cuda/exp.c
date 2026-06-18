/* CUDA-adapted from MPC by tools/cudafy_mpc.py. */
/* cu_mpc_exp -- exponential of a complex number.

Copyright (C) 2002, 2009, 2010, 2011, 2012, 2020, 2024 INRIA

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

#include "mpc-impl.h"

__host__ __device__
int
cu_mpc_exp (mpc_ptr rop, mpc_srcptr op, mpc_rnd_t rnd)
{
  mpfr_t x, y, z;
  mpfr_prec_t prec;
  int ok = 0;
  int inex_re, inex_im;
  int saved_underflow, saved_overflow;
  mpfr_exp_t saved_emin, saved_emax;

  /* special values */
  if (cu_mpfr_nan_p (mpc_realref (op)) || cu_mpfr_nan_p (mpc_imagref (op)))
    /* NaNs
       exp(nan +i*y) = nan -i*0   if y = -0,
                       nan +i*0   if y = +0,
                       nan +i*nan otherwise
       exp(x+i*nan) =   +/-0 +/-i*0 if x=-inf,
                      +/-inf +i*nan if x=+inf,
                         nan +i*nan otherwise */
    {
      if (cu_mpfr_zero_p (mpc_imagref (op)))
        return cu_mpc_set (rop, op, MPC_RNDNN);

      if (cu_mpfr_inf_p (mpc_realref (op)))
        {
          if (cu_mpfr_signbit (mpc_realref (op)))
            return cu_mpc_set_ui_ui (rop, 0, 0, MPC_RNDNN);
          else
            {
              cu_mpfr_set_inf (mpc_realref (rop), +1);
              cu_mpfr_set_nan (mpc_imagref (rop));
              return MPC_INEX(0, 0); /* Inf/NaN are exact */
            }
        }
      cu_mpfr_set_nan (mpc_realref (rop));
      cu_mpfr_set_nan (mpc_imagref (rop));
      return MPC_INEX(0, 0); /* NaN is exact */
    }

  if (cu_mpfr_zero_p (mpc_imagref(op)))
    /* special case when the input is real
       exp(x-i*0) = exp(x) -i*0, even if x is NaN
       exp(x+i*0) = exp(x) +i*0, even if x is NaN */
    {
      inex_re = cu_mpfr_exp (mpc_realref(rop), mpc_realref(op), MPC_RND_RE(rnd));
      inex_im = cu_mpfr_set (mpc_imagref(rop), mpc_imagref(op), MPC_RND_IM(rnd));
      return MPC_INEX(inex_re, inex_im);
    }

  if (cu_mpfr_zero_p (mpc_realref (op)))
    /* special case when the input is imaginary  */
    {
      inex_re = cu_mpfr_cos (mpc_realref (rop), mpc_imagref (op), MPC_RND_RE(rnd));
      inex_im = cu_mpfr_sin (mpc_imagref (rop), mpc_imagref (op), MPC_RND_IM(rnd));
      return MPC_INEX(inex_re, inex_im);
    }

  if (cu_mpfr_inf_p (mpc_realref (op)))
    /* real part is an infinity,
       exp(-inf +i*y) = 0*(cos y +i*sin y)
       exp(+inf +i*y) = +/-inf +i*nan         if y = +/-inf
                        +inf*(cos y +i*sin y) if 0 < |y| < inf */
    {
      mpfr_t n;

      cu_mpfr_init2 (n, 2);
      if (cu_mpfr_signbit (mpc_realref (op)))
        cu_mpfr_set_ui (n, 0, MPFR_RNDN);
      else
        cu_mpfr_set_inf (n, +1);

      if (cu_mpfr_inf_p (mpc_imagref (op)))
        {
          int real_sign = cu_mpfr_signbit (mpc_realref (op));
          inex_re = cu_mpfr_set (mpc_realref (rop), n, MPFR_RNDN);
          if (real_sign)
            inex_im = cu_mpfr_set (mpc_imagref (rop), n, MPFR_RNDN);
          else
            {
              cu_mpfr_set_nan (mpc_imagref (rop));
              inex_im = 0; /* NaN is exact */
            }
        }
      else
        {
          mpfr_t c, s;
          cu_mpfr_init2 (c, 2);
          cu_mpfr_init2 (s, 2);

          cu_mpfr_sin_cos (s, c, mpc_imagref (op), MPFR_RNDN);
          inex_re = cu_mpfr_copysign (mpc_realref (rop), n, c, MPFR_RNDN);
          inex_im = cu_mpfr_copysign (mpc_imagref (rop), n, s, MPFR_RNDN);

          cu_mpfr_clear (s);
          cu_mpfr_clear (c);
        }

      cu_mpfr_clear (n);
      return MPC_INEX(inex_re, inex_im);
    }

  if (cu_mpfr_inf_p (mpc_imagref (op)))
    /* real part is finite non-zero number, imaginary part is an infinity */
    {
      cu_mpfr_set_nan (mpc_realref (rop));
      cu_mpfr_set_nan (mpc_imagref (rop));
      return MPC_INEX(0, 0); /* NaN is exact */
    }

  saved_emin = cu_mpfr_get_emin ();
  saved_emax = cu_mpfr_get_emax ();
  cu_mpfr_set_emin (cu_mpfr_get_emin_min ());
  cu_mpfr_set_emax (cu_mpfr_get_emax_max ());

  /* from now on, both parts of op are regular numbers */

  prec = MPC_MAX_PREC(rop)
         + MPC_MAX (MPC_MAX (-cu_mpfr_get_exp (mpc_realref (op)), 0),
                   -cu_mpfr_get_exp (mpc_imagref (op)));
    /* When op is close to 0, then exp is close to 1+Re(op), while
       cos is close to 1-Im(op); to decide on the ternary value of exp*cos,
       we need a high enough precision so that none of exp or cos is
       computed as 1. */
  cu_mpfr_init2 (x, 2);
  cu_mpfr_init2 (y, 2);
  cu_mpfr_init2 (z, 2);

  /* save the underflow or overflow flags from MPFR */
  saved_underflow = cu_mpfr_underflow_p ();
  saved_overflow = cu_mpfr_overflow_p ();

  do
    {
      prec += prec / 2 + cu_mpc_ceil_log2 (prec) + 5;

      cu_mpfr_set_prec (x, prec);
      cu_mpfr_set_prec (y, prec);
      cu_mpfr_set_prec (z, prec);

      /* FIXME: x may overflow so x.y does overflow too, while Re(exp(op))
         could be represented in the precision of rop. */
      cu_mpfr_clear_overflow ();
      cu_mpfr_clear_underflow ();
      cu_mpfr_exp (x, mpc_realref(op), MPFR_RNDN); /* error <= 0.5ulp */
      cu_mpfr_sin_cos (z, y, mpc_imagref(op), MPFR_RNDN); /* errors <= 0.5ulp */
      cu_mpfr_mul (y, y, x, MPFR_RNDN); /* error <= 2ulp */
      ok = cu_mpfr_overflow_p () || cu_mpfr_zero_p (x)
        || cu_mpfr_can_round (y, prec - 2, MPFR_RNDN, MPFR_RNDZ,
                       MPC_PREC_RE(rop) + (MPC_RND_RE(rnd) == MPFR_RNDN));
      if (ok) /* compute imaginary part */
        {
          cu_mpfr_mul (z, z, x, MPFR_RNDN);
          ok = cu_mpfr_overflow_p () || cu_mpfr_zero_p (x)
            || cu_mpfr_can_round (z, prec - 2, MPFR_RNDN, MPFR_RNDZ,
                       MPC_PREC_IM(rop) + (MPC_RND_IM(rnd) == MPFR_RNDN));
        }
    }
  while (ok == 0);

  inex_re = cu_mpfr_set (mpc_realref(rop), y, MPC_RND_RE(rnd));
  inex_im = cu_mpfr_set (mpc_imagref(rop), z, MPC_RND_IM(rnd));
  if (cu_mpfr_overflow_p ())
    {
      inex_re = cu_mpc_fix_inf (mpc_realref(rop), MPC_RND_RE(rnd));
      inex_im = cu_mpc_fix_inf (mpc_imagref(rop), MPC_RND_IM(rnd));
    }
  else if (cu_mpfr_underflow_p ())
    {
      inex_re = cu_mpc_fix_zero (mpc_realref(rop), MPC_RND_RE(rnd));
      inex_im = cu_mpc_fix_zero (mpc_imagref(rop), MPC_RND_IM(rnd));
    }

  cu_mpfr_clear (x);
  cu_mpfr_clear (y);
  cu_mpfr_clear (z);

  /* restore underflow and overflow flags from MPFR */
  if (saved_underflow)
    cu_mpfr_set_underflow ();
  if (saved_overflow)
    cu_mpfr_set_overflow ();

  /* restore the exponent range, and check the range of results */
  cu_mpfr_set_emin (saved_emin);
  cu_mpfr_set_emax (saved_emax);
  inex_re = cu_mpfr_check_range (mpc_realref (rop), inex_re, MPC_RND_RE (rnd));
  inex_im = cu_mpfr_check_range (mpc_imagref (rop), inex_im, MPC_RND_IM (rnd));

  return MPC_INEX(inex_re, inex_im);
}
