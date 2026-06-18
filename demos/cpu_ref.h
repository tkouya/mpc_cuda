/* cpu_ref.h -- shared interface between the GPU half (a .cu calling this
 * library's cu_mpfr_* / cu_mpc_* API) and the CPU half (cpu_ref.cpp, which calls
 * the *system* libmpfr / libmpc for a reference).
 *
 * This header is deliberately free of any GMP/MPFR/MPC type: it only declares
 * the plain-C entry points and the shared function-id enums, so it can be
 * included by BOTH the cudafied .cu (which pulls this library's headers) and
 * the system-library .cpp (which pulls <mpfr.h>/<mpc.h>) without either side's
 * type names leaking into the other.
 */
#ifndef MPC_CUDA_CPU_REF_H
#define MPC_CUDA_CPU_REF_H

#ifdef __cplusplus
extern "C" {
#endif

/* real (MPFR) functions, in the order CR_NAME[] lists them */
enum { CR_SQRT, CR_CBRT, CR_EXP, CR_EXPM1, CR_LOG, CR_LOG1P,
       CR_SIN, CR_COS, CR_TAN, CR_ATAN, CR_SINH, CR_COSH, CR_NMPFR };

/* complex (MPC) functions, in the order CC_NAME[] lists them */
enum { CC_SQR, CC_SQRT, CC_EXP, CC_LOG, CC_SIN, CC_COS, CC_TAN,
       CC_SINH, CC_COSH, CC_ASIN, CC_ACOS, CC_ATAN, CC_NMPC };

extern const char *const CR_NAME[CR_NMPFR];
extern const char *const CC_NAME[CC_NMPC];

/* Evaluate one function at `prec` bits with the SYSTEM library and return the
 * result rounded to double (the imaginary part via the *_im pointer for MPC). */
double cpu_mpfr_eval (int fid, int prec, double x);
void   cpu_mpc_eval  (int fid, int prec, double zr, double zi,
                      double *wr, double *wi);

/* Batch versions of the above: evaluate f over all n inputs.  These are
 * OpenMP-parallelized (one independent MPFR/MPC eval per element) so the CPU
 * reference uses every available core -- needed for a fair "fastest CPU"
 * benchmark number against the GPU.  Each iteration owns its own temporaries,
 * so the loop is embarrassingly parallel. */
void cpu_mpfr_eval_array (int fid, int prec, int n,
                          const double *X, double *Y);
void cpu_mpc_eval_array  (int fid, int prec, int n,
                          const double *Zr, const double *Zi,
                          double *Wr, double *Wi);

/* Pin the OpenMP team to the largest thread count the machine offers and return
 * it (returns 1 when built without OpenMP).  Call once at program start. */
int cpu_set_max_threads (void);

const char *cpu_mpfr_version (void);
const char *cpu_mpc_version  (void);

#ifdef __cplusplus
}
#endif

#endif /* MPC_CUDA_CPU_REF_H */
