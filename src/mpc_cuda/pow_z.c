/* CUDA-adapted from MPC by tools/cudafy_mpc.py. */
/* cu_mpc_pow_z -- Raise a complex number to an integer power.

Copyright (C) 2009, 2010 INRIA

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
cu_mpc_pow_z (mpc_ptr z, mpc_srcptr x, mpz_srcptr y, mpc_rnd_t rnd)
{
   mpc_t yy;
   int inex;
   mpfr_prec_t n = (mpfr_prec_t) cu_mpz_sizeinbase (y, 2);

   /* if y fits in an unsigned long or long, call the corresponding functions,
      which are supposed to be more efficient */
   if (cu_mpz_cmp_ui (y, 0ul) >= 0) {
      if (cu_mpz_fits_ulong_p (y))
         return cu_mpc_pow_usi (z, x, cu_mpz_get_ui (y), 1, rnd);
   }
   else {
      if (cu_mpz_fits_slong_p (y))
         return cu_mpc_pow_usi (z, x, (unsigned long) (-cu_mpz_get_si (y)), -1, rnd);
   }

   cu_mpc_init3 (yy, (n < MPFR_PREC_MIN) ? MPFR_PREC_MIN : n, MPFR_PREC_MIN);
   cu_mpc_set_z (yy, y, MPC_RNDNN);   /* exact */
   inex = mpc_pow (z, x, yy, rnd);
   cu_mpc_clear (yy);
   return inex;
}

