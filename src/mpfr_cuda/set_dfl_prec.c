/* CUDA-adapted from MPFR by tools/cudafy_mpfr.py. */
/* cu_mpfr_set_default_prec, cu_mpfr_get_default_prec -- set/get default precision

Copyright 1999-2001, 2004-2025 Free Software Foundation, Inc.
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

/* default is IEEE double precision, i.e. 53 bits */
MPFR_THREAD_VAR (mpfr_prec_t, cu___gmpfr_default_fp_bit_precision,
                 IEEE_DBL_MANT_DIG)

__host__ __device__
void
cu_mpfr_set_default_prec (mpfr_prec_t prec)
{
  MPFR_ASSERTN (MPFR_PREC_COND (prec));
  cu___gmpfr_default_fp_bit_precision = prec;
}

#undef cu_mpfr_get_default_prec
__host__ __device__
mpfr_prec_t
cu_mpfr_get_default_prec (void)
{
  return cu___gmpfr_default_fp_bit_precision;
}
