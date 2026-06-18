/* CUDA-adapted from MPFR by tools/cudafy_mpfr.py. */
/* bernoulli -- internal function to compute Bernoulli numbers.

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

/* assume p >= 5 and is odd */
__host__ __device__
static int
is_prime (unsigned long p)
{
  unsigned long q;

  MPFR_ASSERTD (p >= 5 && (p & 1) != 0);
  for (q = 3; q * q <= p; q += 2)
    if ((p % q) == 0)
      return 0;
  return 1;
}

/* Computes and stores B[2n]*(2n+1)! in b[n]
   using Von Staudt–Clausen theorem, which says that the denominator of B[n]
   divides the product of all primes p such that p-1 divides n.
   Since B[n] = zeta(n) * 2*n!/(2pi)^n, we compute an approximation of
   (2n+1)! * zeta(n) * 2*n!/(2pi)^n and round it to the nearest integer. */
__host__ __device__
static void
mpfr_bernoulli_internal (mpz_t *b, unsigned long n)
{
  unsigned long p, err, zn;
  mpz_t s, t, u, den;
  mpz_ptr num;
  mpfr_t y, z;
  int ok;
  /* Prec[n/2] is minimal precision so that result is correct for B[n] */
  mpfr_prec_t prec;
  mpfr_prec_t Prec[] = {0, 5, 5, 6, 6, 9, 16, 10, 19, 23, 25, 27, 35, 31,
                        42, 51, 51, 50, 73, 60, 76, 79, 83, 87, 101, 97,
                        108, 113, 119, 125, 149, 133, 146};

  cu_mpz_init (b[n]);

  if (n == 0)
    {
      cu_mpz_set_ui (b[0], 1);
      return;
    }

  /* compute denominator */
  num = b[n];
  n = 2 * n;
  cu_mpz_init_set_ui (den, 6);
  for (p = 5; p <= n+1; p += 2)
    {
      if ((n % (p-1)) == 0 && is_prime (p))
        cu_mpz_mul_ui (den, den, p);
    }
  if (n <= 64)
    prec = Prec[n >> 1];
  else
    {
      /* evaluate the needed precision: zeta(n)*2*den*n!/(2*pi)^n <=
         3.3*den*(n/e/2/pi)^n*sqrt(2*pi*n) */
      prec = cu___gmpfr_ceil_log2 (7.0 * (double) n); /* bound 2*pi by 7 */
      prec = (prec + 1) >> 1; /* sqrt(2*pi*n) <= 2^prec */
      cu_mpfr_init2 (z, 53);
      cu_mpfr_set_ui_2exp (z, 251469612, -32, MPFR_RNDU); /* 1/e/2/pi <= z */
      cu_mpfr_mul_ui (z, z, n, MPFR_RNDU);
      cu_mpfr_log2 (z, z, MPFR_RNDU);
      cu_mpfr_mul_ui (z, z, n, MPFR_RNDU);
      p = cu_mpfr_get_ui (z, MPFR_RNDU); /* (n/e/2/pi)^n <= 2^p */
      cu_mpfr_clear (z);
      MPFR_INC_PREC (prec, p + cu_mpz_sizeinbase (den, 2));
      /* the +2 term ensures no rounding failure up to n=10000 */
      MPFR_INC_PREC (prec, cu___gmpfr_ceil_log2 (prec) + 2);
    }

 try_again:
  cu_mpz_init (s);
  cu_mpz_init (t);
  cu_mpz_init (u);
  cu_mpz_set_ui (u, 1);
  cu_mpz_mul_2exp (u, u, prec); /* u = 2^prec */
  cu_mpz_ui_pow_ui (t, 3, n);
  cu_mpz_fdiv_q (s, u, t); /* multiply all terms by 2^prec */
  /* we compute a lower bound of the series, thus the final result cannot
     be too large */
  for (p = 4; cu_mpz_cmp_ui (t, 0) > 0; p++)
    {
      cu_mpz_ui_pow_ui (t, p, n);
      cu_mpz_fdiv_q (t, u, t);
      /* 2^prec/p^n-1 < t <= 2^prec/p^n */
      cu_mpz_add (s, s, t);
    }
  /* sum(2^prec/q^n-1, q=3..p) < t <= sum(2^prec/q^n, q=3..p)
     thus the error on the truncated series is at most p-2.
     The neglected part of the series is R = sum(1/x^n, x=p+1..infinity)
     with int(1/x^n, x=p+1..infinity) <= R <= int(1/x^n, x=p..infinity)
     thus 1/(n-1)/(p+1)^(n-1) <= R <= 1/(n-1)/p^(n-1). The difference between
     the lower and upper bound is bounded by p^(-n), which is bounded by
     2^(-prec) since t=0 in the above loop */
  cu_mpz_ui_pow_ui (t, p, n - 1);
  cu_mpz_mul_ui (t, t, n - 1);
  cu_mpz_cdiv_q (t, u, t);
  cu_mpz_add (s, s, t);
  /* now 2^prec * (zeta(n)-1-1/2^n) - p < s <= 2^prec * (zeta(n)-1-1/2^n) */
  /* add 1 which is 2^prec */
  cu_mpz_add (s, s, u);
  /* add 1/2^n which is 2^(prec-n) */
  cu_mpz_cdiv_q_2exp (u, u, n);
  cu_mpz_add (s, s, u);
  /* now 2^prec * zeta(n) - p < s <= 2^prec * zeta(n) */
  /* multiply by n! */
  cu_mpz_fac_ui (t, n);
  cu_mpz_mul (s, s, t);
  /* multiply by 2*den */
  cu_mpz_mul (s, s, den);
  cu_mpz_mul_2exp (s, s, 1);
  /* now convert to mpfr */
  cu_mpfr_init2 (z, prec);
  cu_mpfr_set_z (z, s, MPFR_RNDZ);
  /* now (2^prec * zeta(n) - p) * 2*den*n! - ulp(z) < z <=
     2^prec * zeta(n) * 2*den*n!.
     Since z <= 2^prec * zeta(n) * 2*den*n!,
     ulp(z) <= 2*zeta(n) * 2*den*n!, thus
     (2^prec * zeta(n)-(p+1)) * 2*den*n! < z <= 2^prec * zeta(n) * 2*den*n! */
  cu_mpfr_div_2ui (z, z, prec, MPFR_RNDZ);
  /* now (zeta(n) - (p+1)/2^prec) * 2*den*n! < z <= zeta(n) * 2*den*n! */
  /* divide by (2pi)^n */
  cu_mpfr_init2 (y, prec);
  cu_mpfr_const_pi (y, MPFR_RNDU);
  /* pi <= y <= pi * (1 + 2^(1-prec)) */
  cu_mpfr_mul_2ui (y, y, 1, MPFR_RNDU);
  /* 2pi <= y <= 2pi * (1 + 2^(1-prec)) */
  cu_mpfr_pow_ui (y, y, n, MPFR_RNDU);
  /* (2pi)^n <= y <= (2pi)^n * (1 + 2^(1-prec))^(n+1) */
  cu_mpfr_div (z, z, y, MPFR_RNDZ);
  /* now (zeta(n) - (p+1)/2^prec) * 2*den*n! / (2pi)^n / (1+2^(1-prec))^(n+1)
     <= z <= zeta(n) * 2*den*n! / (2pi)^n, and since zeta(n) >= 1:
     den * B[n] * (1 - (p+1)/2^prec) / (1+2^(1-prec))^(n+1)
     <= z <= den * B[n]
     Since 1 / (1+2^(1-prec))^(n+1) >= (1 - 2^(1-prec))^(n+1) >=
     1 - (n+1) * 2^(1-prec):
     den * B[n] / (2pi)^n * (1 - (p+1)/2^prec) * (1-(n+1)*2^(1-prec))
     <= z <= den * B[n] thus
     den * B[n] * (1 - (2n+p+3)/2^prec) <= z <= den * B[n] */

  /* the error is bounded by 2^(EXP(z)-prec) * (2n+p+3) */
  for (err = 0, p = 2 * n + p + 3; p > 1; err++, p = (p + 1) >> 1);
  zn = MPFR_LIMB_SIZE(z) * GMP_NUMB_BITS; /* total number of bits of z */
  if (err >= prec)
    ok = 0;
  else
    {
      err = prec - err;
      /* now the absolute error is bounded by 2^(EXP(z) - err):
         den * B[n] - 2^(EXP(z) - err) <= z <= den * B[n]
         thus if subtracting 2^(EXP(z) - err) does not change the rounding
         (up) we are ok */
      err = cu_mpn_scan1 (MPFR_MANT(z), zn - err);
      /* weight of this 1 bit is 2^(EXP(z) - zn + err) */
      ok = MPFR_EXP(z) < zn - err;
    }
  cu_mpfr_get_z (num, z, MPFR_RNDU);
  if ((n & 2) == 0)
    cu_mpz_neg (num, num);

  /* multiply by (n+1)! */
  cu_mpz_mul_ui (t, t, n + 1);
  cu_mpz_divexact (t, t, den); /* t was still n! */
  cu_mpz_mul (num, num, t);

  cu_mpfr_clear (y);
  cu_mpfr_clear (z);
  cu_mpz_clear (s);
  cu_mpz_clear (t);
  cu_mpz_clear (u);

  if (!ok)
    {
      MPFR_INC_PREC (prec, prec / 10);
      goto try_again;
    }

  cu_mpz_clear (den);
}

MPFR_RODATA static MPFR_THREAD_ATTR mpz_t *bernoulli_table = NULL;
MPFR_RODATA static MPFR_THREAD_ATTR unsigned long bernoulli_size = 0;
MPFR_RODATA static MPFR_THREAD_ATTR unsigned long bernoulli_alloc = 0;

__host__ __device__
mpz_srcptr
cu_mpfr_bernoulli_cache (unsigned long n)
{
  unsigned long i;

  if (n >= bernoulli_size)
    {
      if (bernoulli_alloc == 0)
        {
          bernoulli_alloc = MAX(16, n + n/4);
          bernoulli_table = (mpz_t *)
            cu_mpfr_allocate_func (bernoulli_alloc * sizeof (mpz_t));
          bernoulli_size  = 0;
        }
      else if (n >= bernoulli_alloc)
        {
          bernoulli_table = (mpz_t *) cu_mpfr_reallocate_func
            (bernoulli_table, bernoulli_alloc * sizeof (mpz_t),
             (n + n/4) * sizeof (mpz_t));
          bernoulli_alloc = n + n/4;
        }
      MPFR_ASSERTD (bernoulli_alloc > n);
      MPFR_ASSERTD (bernoulli_size >= 0);
      for (i = bernoulli_size; i <= n; i++)
        mpfr_bernoulli_internal (bernoulli_table, i);
      bernoulli_size = n+1;
    }
  MPFR_ASSERTD (bernoulli_size > n);
  return bernoulli_table[n];
}

__host__ __device__
void
cu_mpfr_bernoulli_freecache (void)
{
  unsigned long i;

  if (bernoulli_table != NULL)
    {
      for (i = 0; i < bernoulli_size; i++)
        {
          cu_mpz_clear (bernoulli_table[i]);
        }
      cu_mpfr_free_func (bernoulli_table, bernoulli_alloc * sizeof (mpz_t));
      bernoulli_table = NULL;
      bernoulli_alloc = 0;
      bernoulli_size = 0;
    }
}
