#                                               -*- Autoconf -*-
# configure.ac for mpc_cuda -- the GNU multiple-precision stack inside CUDA.
#
# This package is built with nvcc and re-runnable Python transform scripts, not
# a normal C toolchain, so it uses autoconf with a hand-written Makefile.in
# (no automake).  ./configure substitutes the discovered tools and paths.

AC_PREREQ([2.69])
AC_INIT([mpc_cuda], [0.0.2], [tkouya@gmail.com])
AC_CONFIG_SRCDIR([tools/cudafy_mpfr.py])

# --- CUDA (optional): without it, a CPU-only build ---------------------------
# The fixed-precision types (cu_freal / cu_fcomplex / cu_ffused / elementary
# functions) are header-only and work with any C++17 host compiler.  When nvcc
# is not found (or --without-cuda is given) the package configures in CPU-only
# mode: 'make' / 'make check' / 'make install' then cover just that host part,
# and the GPU library, demos and GPU tests are skipped.
AC_ARG_WITH([cuda],
  [AS_HELP_STRING([--without-cuda],
     [CPU-only build: only the header-only fixed-precision host part (default: use CUDA if nvcc is found)])],
  [], [with_cuda=check])
AC_ARG_VAR([NVCC], [NVIDIA CUDA compiler command (default: nvcc)])
HAVE_CUDA=no
AS_IF([test "x$with_cuda" != xno],
  [AC_PATH_PROG([NVCC], [nvcc], [no],
                [$PATH$PATH_SEPARATOR/usr/local/cuda/bin$PATH_SEPARATOR/opt/cuda/bin])
   AS_IF([test "x$NVCC" != xno], [HAVE_CUDA=yes],
         [test "x$with_cuda" = xyes],
           [AC_MSG_ERROR([nvcc not found. Install the CUDA Toolkit or set NVCC=/path/to/nvcc.])],
         [AC_MSG_WARN([nvcc not found: configuring a CPU-only build (fixed-precision host part only)])])],
  [NVCC=no])
AC_SUBST([HAVE_CUDA])

AC_ARG_VAR([PYTHON], [Python 3 interpreter (default: python3)])
AC_PATH_PROG([PYTHON], [python3], [no])
AS_IF([test "x$PYTHON" = xno && test "x$HAVE_CUDA" = xyes],
      [AC_MSG_ERROR([python3 not found; it is required to run the transform scripts.])])

AC_PROG_CXX
AC_CHECK_TOOL([NM], [nm], [nm])

# --- target GPU architecture ------------------------------------------------
AC_ARG_WITH([cuda-arch],
  [AS_HELP_STRING([--with-cuda-arch=ARCH],
                  [target GPU compute architecture, e.g. sm_90, sm_121 (default: sm_90)])],
  [CUDA_ARCH=$withval], [CUDA_ARCH=sm_90])
AC_SUBST([CUDA_ARCH])

# --- upstream multiple-precision sources (OPTIONAL: only for 'make regen') ---
# The package ships the CUDA-adapted sources under src/ and builds entirely
# from them.  The pristine upstream GMP/MPFR/MPC trees are needed ONLY for the
# opt-in 'make regen', which re-derives src/ from upstream via the transform
# scripts.  A normal build (and the dist tarball, which omits upstream) does
# NOT require them, so a missing upstream tree must never fail ./configure.
AC_ARG_WITH([gmp-src],
  [AS_HELP_STRING([--with-gmp-src=DIR], [mini-gmp source dir (default: gmp-6.3.0/mini-gmp)])],
  [GMP_SRC=$withval], [GMP_SRC=gmp-6.3.0/mini-gmp])
AC_ARG_WITH([mpfr-src],
  [AS_HELP_STRING([--with-mpfr-src=DIR], [MPFR src dir (default: mpfr-4.2.2/src)])],
  [MPFR_SRC=$withval], [MPFR_SRC=mpfr-4.2.2/src])
AC_ARG_WITH([mpc-src],
  [AS_HELP_STRING([--with-mpc-src=DIR], [MPC src dir (default: mpc-1.4.1/src)])],
  [MPC_SRC=$withval], [MPC_SRC=mpc-1.4.1/src])
AC_SUBST([GMP_SRC])
AC_SUBST([MPFR_SRC])
AC_SUBST([MPC_SRC])

# Detect the upstream trees but do NOT error when absent: just record whether
# 'make regen' will be usable.  Only a non-default --with-*-src that points at a
# bad path is worth a warning (the user clearly intended to use it).
have_upstream=yes
AC_MSG_CHECKING([for the pristine upstream sources (for 'make regen')])
AS_IF([test -f "$srcdir/$GMP_SRC/mini-gmp.c" \
       && test -f "$srcdir/$MPFR_SRC/mpfr-impl.h" \
       && test -f "$srcdir/$MPC_SRC/mpc-impl.h"],
      [AC_MSG_RESULT([yes])],
      [have_upstream=no
       AC_MSG_RESULT([no -- 'make' works; 'make regen' unavailable])])
AS_IF([test "x$have_upstream" = xno && test "x$with_gmp_src$with_mpfr_src$with_mpc_src" != x],
      [AC_MSG_WARN([a --with-*-src option was given but the upstream tree is incomplete; 'make regen' will not work.])])
AC_SUBST([HAVE_UPSTREAM], [$have_upstream])

# --- optional: system libmpfr/libmpc/libgmp ---------------------------------
# Used by the CPU-reference programs ('make coexist', 'make cputest',
# 'make cpubench', 'make bench3') that link THIS library's GPU port and the
# ordinary CPU libmpfr / libmpc into one binary and check the GPU against the CPU.
#
# --with-mp-prefix=DIR points at a non-standard GMP/MPFR/MPC install (e.g.
# ~/local): its include/ and lib/ are added to the autoconf probes below AND
# propagated to the Makefile as SYS_MP_INC / SYS_MP_LIBS (lib path + rpath), so
# the CPU-reference targets compile, link, and run without LD_LIBRARY_PATH.
AC_ARG_WITH([mp-prefix],
  [AS_HELP_STRING([--with-mp-prefix=DIR],
     [prefix of an existing GMP/MPFR/MPC install (adds DIR/include and DIR/lib, with rpath)])],
  [MP_PREFIX=$withval], [MP_PREFIX=])
SYS_MP_INC=
SYS_MP_LIBS="-lmpc -lmpfr -lgmp"
AS_IF([test -n "$MP_PREFIX" && test "x$MP_PREFIX" != xno && test "x$MP_PREFIX" != xyes],
  [SYS_MP_INC="-I$MP_PREFIX/include"
   SYS_MP_LIBS="-L$MP_PREFIX/lib -Xlinker -rpath -Xlinker $MP_PREFIX/lib -lmpc -lmpfr -lgmp"
   CPPFLAGS="$CPPFLAGS -I$MP_PREFIX/include"
   LDFLAGS="$LDFLAGS -L$MP_PREFIX/lib -Wl,-rpath,$MP_PREFIX/lib"])
AC_SUBST([SYS_MP_INC])
AC_SUBST([SYS_MP_LIBS])

have_system_mpfr=no
have_system_mpc=no
AC_CHECK_LIB([gmp], [__gmpz_init],
  [AC_CHECK_LIB([mpfr], [mpfr_get_version],
     [have_system_mpfr=yes
      AC_CHECK_LIB([mpc], [mpc_get_version], [have_system_mpc=yes], [], [-lmpfr -lgmp])],
     [], [-lgmp])])
# The CPU-reference .cpp also needs the development headers <mpfr.h>/<mpc.h>.
AC_CHECK_HEADERS([mpfr.h])
AC_CHECK_HEADERS([mpc.h])
AC_SUBST([HAVE_SYSTEM_MPFR], [$have_system_mpfr])
AC_SUBST([HAVE_SYSTEM_MPC],  [$have_system_mpc])

# --- optional: FLINT (only for the nfloat comparison in 'make bench-host-fixed')
AC_ARG_WITH([flint-prefix],
  [AS_HELP_STRING([--with-flint-prefix=DIR], [prefix of FLINT >= 3.1 for 'make bench-host-fixed' (default: /usr/local)])],
  [FLINT_PREFIX=$withval], [FLINT_PREFIX=/usr/local])
have_flint=no
AS_IF([test -f "$FLINT_PREFIX/include/flint/nfloat.h"], [have_flint=yes])
AC_SUBST([FLINT_PREFIX])

# --- pkg-config: the CPU-only install is header-only ------------------------
AS_IF([test "x$HAVE_CUDA" = xyes],
  [PC_LIBS='-L${libdir} -lmpc_cuda'; PC_LIBS_PRIVATE='-lcudart'],
  [PC_LIBS=''; PC_LIBS_PRIVATE=''])
AC_SUBST([PC_LIBS])
AC_SUBST([PC_LIBS_PRIVATE])

AC_CONFIG_FILES([Makefile mpc_cuda.pc])
AC_OUTPUT

AS_IF([test "x$HAVE_CUDA" = xyes],
[AC_MSG_NOTICE([
  mpc_cuda $PACKAGE_VERSION configured (CUDA + host):
    nvcc .............. $NVCC
    python3 ........... $PYTHON
    CUDA arch ......... $CUDA_ARCH
    upstream src ...... $have_upstream   (for 'make regen'; not needed to build)
    system libmpfr .... $have_system_mpfr   (for 'make coexist' / 'make bench3')
    system libmpc ..... $have_system_mpc   (for 'make cputest' / 'make cpubench')
    MP prefix ......... ${MP_PREFIX:-(system default search path)}
    FLINT ............. $have_flint   ($FLINT_PREFIX; for 'make bench-host-fixed')
    install prefix .... $prefix

  Next:  make            # build the device-object library
         make check      # build and run the test suite
         make demos      # build the AXPY / matrix / benchmark demos
         make check-fixed-host   # host (CPU) fixed-precision check
])],
[AC_MSG_NOTICE([
  mpc_cuda $PACKAGE_VERSION configured: CPU-ONLY (no CUDA)
    C++ compiler ...... $CXX
    system libmpfr .... $have_system_mpfr   (needed by 'make check')
    system libmpc ..... $have_system_mpc   (needed by 'make check')
    MP prefix ......... ${MP_PREFIX:-(system default search path)}
    FLINT ............. $have_flint   ($FLINT_PREFIX; for 'make bench-host-fixed')
    install prefix .... $prefix

  The fixed-precision types are header-only:  #include "mpc_cuda_host.h"
  Next:  make            # build the host sample
         make check      # bit-exact checks vs MPFR/MPC (fixed precision, fma/dot,
                         # elementary functions)
         make bench-host-fixed   # timing vs FLINT nfloat and MPFR/MPC
         make install    # headers + pkg-config file
])])
