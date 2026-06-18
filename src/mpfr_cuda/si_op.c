/* CUDA-adapted from MPFR by tools/cudafy_mpfr.py. */
/* cu_mpfr_add_si -- add a floating-point number with a machine integer
   cu_mpfr_sub_si -- sub a floating-point number with a machine integer
   cu_mpfr_si_sub -- sub a machine number with a floating-point number
   cu_mpfr_mul_si -- multiply a floating-point number by a machine integer
   cu_mpfr_div_si -- divide a floating-point number by a machine integer
   cu_mpfr_si_div -- divide a machine number by a floating-point number

Copyright 2004-2025 Free Software Foundation, Inc.
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

__host__ __device__
int
cu_mpfr_add_si (mpfr_ptr y, mpfr_srcptr x, long int u, mpfr_rnd_t rnd_mode)
{
  int res;

  MPFR_LOG_FUNC
    (("x[%Pd]=%.*Rg u=%ld rnd=%d",
      cu_mpfr_get_prec(x), mpfr_log_prec, x, u, rnd_mode),
     ("y[%Pd]=%.*Rg inexact=%d",
      cu_mpfr_get_prec(y), mpfr_log_prec, y, res));

  if (u >= 0)
    res = cu_mpfr_add_ui (y, x, u, rnd_mode);
  else
    res = cu_mpfr_sub_ui (y, x, - (unsigned long) u, rnd_mode);

  return res;
}

__host__ __device__
int
cu_mpfr_sub_si (mpfr_ptr y, mpfr_srcptr x, long int u, mpfr_rnd_t rnd_mode)
{
  int res;

  MPFR_LOG_FUNC
    (("x[%Pd]=%.*Rg u=%ld rnd=%d",
      cu_mpfr_get_prec(x), mpfr_log_prec, x, u, rnd_mode),
     ("y[%Pd]=%.*Rg inexact=%d",
      cu_mpfr_get_prec(y), mpfr_log_prec, y, res));

  if (u >= 0)
    res = cu_mpfr_sub_ui (y, x, u, rnd_mode);
  else
    res = cu_mpfr_add_ui (y, x, - (unsigned long) u, rnd_mode);

  return res;
}

__host__ __device__
int
cu_mpfr_si_sub (mpfr_ptr y, long int u, mpfr_srcptr x, mpfr_rnd_t rnd_mode)
{
  int res;

  MPFR_LOG_FUNC
    (("x[%Pd]=%.*Rg u=%ld rnd=%d",
      cu_mpfr_get_prec(x), mpfr_log_prec, x, u, rnd_mode),
     ("y[%Pd]=%.*Rg inexact=%d",
      cu_mpfr_get_prec(y), mpfr_log_prec, y, res));

  if (u >= 0)
    res = cu_mpfr_ui_sub (y, u, x, rnd_mode);
  else
    {
      res = - cu_mpfr_add_ui (y, x, - (unsigned long) u,
                           MPFR_INVERT_RND (rnd_mode));
      MPFR_CHANGE_SIGN (y);
    }

  return res;
}

#undef cu_mpfr_mul_si
__host__ __device__
int
cu_mpfr_mul_si (mpfr_ptr y, mpfr_srcptr x, long int u, mpfr_rnd_t rnd_mode)
{
  int res;

  MPFR_LOG_FUNC
    (("x[%Pd]=%.*Rg u=%ld rnd=%d",
      cu_mpfr_get_prec(x), mpfr_log_prec, x, u, rnd_mode),
     ("y[%Pd]=%.*Rg inexact=%d",
      cu_mpfr_get_prec(y), mpfr_log_prec, y, res));

  if (u >= 0)
    res = cu_mpfr_mul_ui (y, x, u, rnd_mode);
  else
    {
      res = - cu_mpfr_mul_ui (y, x, - (unsigned long) u,
                           MPFR_INVERT_RND (rnd_mode));
      MPFR_CHANGE_SIGN (y);
    }

  return res;
}

#undef cu_mpfr_div_si
__host__ __device__
int
cu_mpfr_div_si (mpfr_ptr y, mpfr_srcptr x, long int u, mpfr_rnd_t rnd_mode)
{
  int res;

  MPFR_LOG_FUNC
    (("x[%Pd]=%.*Rg u=%ld rnd=%d",
      cu_mpfr_get_prec(x), mpfr_log_prec, x, u, rnd_mode),
     ("y[%Pd]=%.*Rg inexact=%d",
      cu_mpfr_get_prec(y), mpfr_log_prec, y, res));

  if (u >= 0)
    res = cu_mpfr_div_ui (y, x, u, rnd_mode);
  else
    {
      res = - cu_mpfr_div_ui (y, x, - (unsigned long) u,
                           MPFR_INVERT_RND (rnd_mode));
      MPFR_CHANGE_SIGN (y);
    }

  return res;
}

__host__ __device__
int
cu_mpfr_si_div (mpfr_ptr y, long int u, mpfr_srcptr x, mpfr_rnd_t rnd_mode)
{
  int res;

  MPFR_LOG_FUNC
    (("x[%Pd]=%.*Rg u=%ld rnd=%d",
      cu_mpfr_get_prec(x), mpfr_log_prec, x, u, rnd_mode),
     ("y[%Pd]=%.*Rg inexact=%d",
      cu_mpfr_get_prec(y), mpfr_log_prec, y, res));

  if (u >= 0)
    res = cu_mpfr_ui_div (y, u, x, rnd_mode);
  else
    {
      res = - cu_mpfr_ui_div (y, - (unsigned long) u, x,
                           MPFR_INVERT_RND(rnd_mode));
      MPFR_CHANGE_SIGN (y);
    }

  return res;
}
