/* CUDA-adapted from MPFR by tools/cudafy_mpfr.py. */
/* cu_mpfr_mul_2exp -- multiply a floating-point number by a power of two

Copyright 1999, 2001, 2004, 2006-2025 Free Software Foundation, Inc.
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

/* Obsolete function, use cu_mpfr_mul_2ui or cu_mpfr_mul_2si instead. */

#undef cu_mpfr_mul_2exp

__host__ __device__
int
cu_mpfr_mul_2exp (mpfr_ptr y, mpfr_srcptr x, unsigned long int n, mpfr_rnd_t rnd_mode)
{
  return cu_mpfr_mul_2ui (y, x, n, rnd_mode);
}
