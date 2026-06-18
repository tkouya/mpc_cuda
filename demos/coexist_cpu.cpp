/* coexist_cpu.cpp -- the CPU half of the coexistence demo.
 *
 * This translation unit uses the *real system* MPFR (libmpfr.so) for a CPU
 * reference.  The system development headers are not installed here, so the few
 * needed functions are declared with their stable libmpfr.so.6 C ABI directly
 * (mpfr_t layout = {prec, sign, exp, limb-ptr}).  When the system <mpfr.h> IS
 * installed, replace these declarations with `#include <mpfr.h>`.
 *
 * The point: this file calls the unprefixed `mpfr_*` symbols, which resolve from
 * the system libmpfr, while the GPU half (coexist_demo.cu) calls this library's
 * `cu_mpfr_*` symbols -- both linked into the SAME binary with no collision.
 */
#include <cstdio>

extern "C" {
typedef long  mpfr_prec_t;
typedef int   mpfr_sign_t;
typedef long  mpfr_exp_t;
typedef struct { mpfr_prec_t _mpfr_prec; mpfr_sign_t _mpfr_sign;
                 mpfr_exp_t _mpfr_exp;   void *_mpfr_d; } __mpfr_struct;
typedef __mpfr_struct mpfr_t[1];
typedef __mpfr_struct *mpfr_ptr;
typedef const __mpfr_struct *mpfr_srcptr;
typedef enum { MPFR_RNDN = 0 } mpfr_rnd_t;

void        mpfr_init2     (mpfr_ptr, mpfr_prec_t);
int         mpfr_set_ui    (mpfr_ptr, unsigned long, mpfr_rnd_t);
int         mpfr_const_pi  (mpfr_ptr, mpfr_rnd_t);
int         mpfr_log       (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
double      mpfr_get_d     (mpfr_srcptr, mpfr_rnd_t);
void        mpfr_clear     (mpfr_ptr);
const char *mpfr_get_version(void);
}

/* Compute pi and log(2) at `prec` bits with the SYSTEM libmpfr. */
extern "C" void
system_mpfr_ref (int prec, double *pi_out, double *log2_out)
{
  mpfr_t r, x;
  mpfr_init2 (r, prec);
  mpfr_init2 (x, prec);
  mpfr_const_pi (r, MPFR_RNDN);          *pi_out   = mpfr_get_d (r, MPFR_RNDN);
  mpfr_set_ui  (x, 2, MPFR_RNDN);
  mpfr_log     (r, x, MPFR_RNDN);        *log2_out = mpfr_get_d (r, MPFR_RNDN);
  mpfr_clear (r);
  mpfr_clear (x);
}

extern "C" const char *
system_mpfr_version (void)
{
  return mpfr_get_version ();
}
