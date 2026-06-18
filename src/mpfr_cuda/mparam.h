/* CUDA-adapted from MPFR by tools/cudafy_mpfr.py. */
/* Various Thresholds of MPFR, not exported.  -*- mode: C -*-

Copyright 2005-2025 Free Software Foundation, Inc.

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

#ifndef __MPFR_IMPL_H__
# error "MPFR Internal not included"
#endif

/* CUDA: per-arch tuning subdirs (x86_64/, x86/, arm/, ...) are not part
   of the transformed device tree -- only generic/ is.  Force the
   generic/default case so the host-arch dispatch never #includes a
   missing per-arch mparam.h.  generic/mparam.h below fills in defaults. */
#define MPFR_TUNE_CASE "default"

/****************************************************************
 * Default values of Threshold.                                 *
 * Must be included in any case: it checks, for every constant, *
 * if it has been defined, and it sets it to a default value if *
 * it was not previously defined.                               *
 ****************************************************************/
#include "generic/mparam.h"
