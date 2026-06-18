/* CUDA-adapted from MPFR by tools/cudafy_mpfr.py. */
/* cu_mpfr_grandom (rop1, rop2, state, rnd_mode) -- Generate up to two
   pseudorandom real numbers according to a standard normal Gaussian
   distribution and round it to the precision of rop1, rop2 according
   to the given rounding mode.

Copyright 2011-2025 Free Software Foundation, Inc.
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


/* #define MPFR_NEED_LONGLONG_H */
#include "mpfr-impl.h"


__host__ __device__
int
cu_mpfr_grandom (mpfr_ptr rop1, mpfr_ptr rop2, gmp_randstate_t rstate,
              mpfr_rnd_t rnd)
{
  int inex1, inex2, s1, s2;
  mpz_t x, y, xp, yp, t, a, b, s;
  mpfr_t sfr, l, r1, r2;
  mpfr_prec_t tprec, tprec0;

  inex2 = inex1 = 0;

  if (rop2 == NULL) /* only one output requested. */
    tprec0 = MPFR_PREC (rop1);
  else
    tprec0 = MAX (MPFR_PREC (rop1), MPFR_PREC (rop2));

  tprec0 += 11;

  /* We use "Marsaglia polar method" here (cf.
     George Marsaglia, Normal (Gaussian) random variables for supercomputers
     The Journal of Supercomputing, Volume 5, Number 1, 49–55
     DOI: 10.1007/BF00155857).

     First we draw uniform x and y in [0,1] using mpz_urandomb (in
     fixed precision), and scale them to [-1, 1].
  */

  cu_mpz_init (xp);
  cu_mpz_init (yp);
  cu_mpz_init (x);
  cu_mpz_init (y);
  cu_mpz_init (t);
  cu_mpz_init (s);
  cu_mpz_init (a);
  cu_mpz_init (b);
  cu_mpfr_init2 (sfr, MPFR_PREC_MIN);
  cu_mpfr_init2 (l, MPFR_PREC_MIN);
  cu_mpfr_init2 (r1, MPFR_PREC_MIN);
  if (rop2 != NULL)
    cu_mpfr_init2 (r2, MPFR_PREC_MIN);

  cu_mpz_set_ui (xp, 0);
  cu_mpz_set_ui (yp, 0);

  for (;;)
    {
      tprec = tprec0;
      do
        {
          mpz_urandomb (xp, rstate, tprec);
          mpz_urandomb (yp, rstate, tprec);
          cu_mpz_mul (a, xp, xp);
          cu_mpz_mul (b, yp, yp);
          cu_mpz_add (s, a, b);
        }
      while (cu_mpz_sizeinbase (s, 2) > tprec * 2);

      /* now s = x^2 + y^2 < 2^{2tprec} */

      for (;;)
        {
          /* we compute (xp+1)^2 + (yp+1)^2 as s + 2xp + 2yp + 2 */
          cu_mpz_addmul_ui (s, xp, 2);
          cu_mpz_addmul_ui (s, yp, 2);
          cu_mpz_add_ui (s, s, 2);
          /* The case s = 2^(2*tprec) is not possible:
           (a) if xp and yp have different parities, s is odd
           (b) if xp and yp are even, (xp+1)^2 and (yp+1)^2 are 1 mod 4,
               thus s = 2 mod 4 (and tprec >= 1);
           (c) if xp and yp are odd, if we note x = xp+1, y = yp+1 and
               p = tprec, we would have x^2 + y^2 = 2^(2p) with x and y even
               0 < x, y <= 2^p, thus if x' = x/2, y' = y/2 and p'=p-1,
               we would have x'^2 + y'^2 = 2^(2p') with
               0 < x', y' <= 2^p', and we conclude by induction. */
          if (cu_mpz_sizeinbase (s, 2) <= 2 * tprec)
            goto yeepee;
          /* Extend by 32 bits: for tprec=12, the probability we get here
             is 8191/13180825, i.e., about 0.000621 */
          cu_mpz_mul_2exp (xp, xp, 32);
          cu_mpz_mul_2exp (yp, yp, 32);
          mpz_urandomb (x, rstate, 32);
          mpz_urandomb (y, rstate, 32);
          cu_mpz_add (xp, xp, x);
          cu_mpz_add (yp, yp, y);
          tprec += 32;

          cu_mpz_mul (a, xp, xp);
          cu_mpz_mul (b, yp, yp);
          cu_mpz_add (s, a, b);
          if (cu_mpz_sizeinbase (s, 2) > tprec * 2)
            break;
        }
    }
 yeepee:

  /* FIXME: compute s with s -= 2x + 2y + 2 */
  cu_mpz_mul (a, xp, xp);
  cu_mpz_mul (b, yp, yp);
  cu_mpz_add (s, a, b);
  /* Compute the signs of the output */
  mpz_urandomb (x, rstate, 2);
  s1 = cu_mpz_tstbit (x, 0);
  s2 = cu_mpz_tstbit (x, 1);
  for (;;)
    {
      /* s = xp^2 + yp^2 (loop invariant) */
      cu_mpfr_set_prec (sfr, 2 * tprec);
      cu_mpfr_set_prec (l, tprec);
      cu_mpfr_set_z (sfr, s, MPFR_RNDN); /* exact */
      cu_mpfr_mul_2si (sfr, sfr, -2 * tprec, MPFR_RNDN); /* exact */
      cu_mpfr_log (l, sfr, MPFR_RNDN);
      cu_mpfr_neg (l, l, MPFR_RNDN);
      cu_mpfr_mul_2si (l, l, 1, MPFR_RNDN);
      cu_mpfr_div (l, l, sfr, MPFR_RNDN);
      cu_mpfr_sqrt (l, l, MPFR_RNDN);

      cu_mpfr_set_prec (r1, tprec);
      cu_mpfr_mul_z (r1, l, xp, MPFR_RNDN);
      cu_mpfr_div_2ui (r1, r1, tprec, MPFR_RNDN); /* exact */
      if (s1)
        cu_mpfr_neg (r1, r1, MPFR_RNDN);
      if (MPFR_CAN_ROUND (r1, tprec - 2, MPFR_PREC (rop1), rnd))
        {
          if (rop2 != NULL)
            {
              cu_mpfr_set_prec (r2, tprec);
              cu_mpfr_mul_z (r2, l, yp, MPFR_RNDN);
              cu_mpfr_div_2ui (r2, r2, tprec, MPFR_RNDN); /* exact */
              if (s2)
                cu_mpfr_neg (r2, r2, MPFR_RNDN);
              if (MPFR_CAN_ROUND (r2, tprec - 2, MPFR_PREC (rop2), rnd))
                break;
            }
          else
            break;
        }
      /* Extend by 32 bits */
      cu_mpz_mul_2exp (xp, xp, 32);
      cu_mpz_mul_2exp (yp, yp, 32);
      mpz_urandomb (x, rstate, 32);
      mpz_urandomb (y, rstate, 32);
      cu_mpz_add (xp, xp, x);
      cu_mpz_add (yp, yp, y);
      tprec += 32;
      cu_mpz_mul (a, xp, xp);
      cu_mpz_mul (b, yp, yp);
      cu_mpz_add (s, a, b);
    }
  inex1 = cu_mpfr_set (rop1, r1, rnd);
  if (rop2 != NULL)
    {
      inex2 = cu_mpfr_set (rop2, r2, rnd);
      inex2 = cu_mpfr_check_range (rop2, inex2, rnd);
    }
  inex1 = cu_mpfr_check_range (rop1, inex1, rnd);

  if (rop2 != NULL)
    cu_mpfr_clear (r2);
  cu_mpfr_clear (r1);
  cu_mpfr_clear (l);
  cu_mpfr_clear (sfr);
  cu_mpz_clear (b);
  cu_mpz_clear (a);
  cu_mpz_clear (s);
  cu_mpz_clear (t);
  cu_mpz_clear (y);
  cu_mpz_clear (x);
  cu_mpz_clear (yp);
  cu_mpz_clear (xp);

  return INEX (inex1, inex2);
}
