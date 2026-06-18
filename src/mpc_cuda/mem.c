/* CUDA-adapted from MPC by tools/cudafy_mpc.py. */
/* wrapper functions to allocate, reallocate and free memory

Copyright (C) 2009 INRIA

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

#include <string.h>   /* for strlen */
#include "mpc-impl.h"

__host__ __device__
char *
cu_mpc_alloc_str (size_t len)
{
  void * (*allocfunc) (size_t);
  mp_get_memory_functions (&allocfunc, NULL, NULL);
  return (char *) ((*allocfunc) (len));
}

__host__ __device__
char *
cu_mpc_realloc_str (char * str, size_t oldlen, size_t newlen)
{
  void * (*reallocfunc) (void *, size_t, size_t);
  mp_get_memory_functions (NULL, &reallocfunc, NULL);
  return (char *) ((*reallocfunc) (str, oldlen, newlen));
}

__host__ __device__
void
cu_mpc_free_str (char *str)
{
  void (*freefunc) (void *, size_t);
  mp_get_memory_functions (NULL, NULL, &freefunc);
  (*freefunc) (str, strlen (str) + 1);
}
