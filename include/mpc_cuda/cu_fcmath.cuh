/* cu_fcmath.cuh -- fixed-precision complex elementary functions for
 * cu_fp::cu_fcomplex<PB>, built on the real cu_fmath + the exact complex
 * mul/fmms/fmma.  High-accuracy (NOT correctly-rounded); ~<=1 ULP vs MPC in the
 * common range.  Provided: cexp, clog, csqrt, csin, ccos, ctan, csinh, ccosh,
 * cdiv, plus atan2.  (Complex special functions are future work.)
 */
#ifndef CU_MPC_CUDA_FCMATH_CUH
#define CU_MPC_CUDA_FCMATH_CUH

#include "mpc_cuda/cu_fmath.cuh"
#include "mpc_cuda/cu_fcomplex.cuh"

namespace cu_fp {

template<int TO,int FROM> __host__ __device__ static inline cu_fcomplex<TO>
cu_cconvert (const cu_fcomplex<FROM>&z){ return cu_fcomplex<TO>(cu_convert<TO>(z.re),cu_convert<TO>(z.im)); }

/* atan2(y,x) at working precision */
template<int W> __host__ __device__ static cu_freal<W>
cu_atan2_w (const cu_freal<W>&y, const cu_freal<W>&x)
{
  cu_freal<W> pih=cu_scale2<W>(cu_const_pi<W>(),-1);
  if (x.is_zero()){ if (y.is_zero()){ cu_freal<W> z; z.set_zero(); return z; }
                    cu_freal<W> r=pih; r.sign=y.sign; return r; }
  cu_freal<W> base = cu_atan_w<W>( cu_fmul<W>(y, cu_recip_w<W>(x)) );
  if (x.sign>0) return base;
  cu_freal<W> pi=cu_const_pi<W>();
  return (y.sign>=0)? cu_fadd<W>(base,pi) : cu_fsub<W>(base,pi);
}

/* complex divide a/b at working precision */
template<int W> __host__ __device__ static cu_fcomplex<W>
cu_cdiv_w (const cu_fcomplex<W>&a, const cu_fcomplex<W>&b)
{
  cu_freal<W> den = cu_fmma<W>(b.re,b.re,b.im,b.im);          /* |b|^2 */
  cu_freal<W> rden = cu_recip_w<W>(den);
  cu_freal<W> nr = cu_fmma<W>(a.re,b.re,a.im,b.im);           /* ar*br+ai*bi */
  cu_freal<W> ni = cu_fmms<W>(a.im,b.re,a.re,b.im);           /* ai*br-ar*bi */
  return cu_fcomplex<W>( cu_fmul<W>(nr,rden), cu_fmul<W>(ni,rden) );
}
template<int PB> __host__ __device__ static cu_fcomplex<PB>
cu_cdiv (const cu_fcomplex<PB>&a, const cu_fcomplex<PB>&b){
  const int W=PB+CU_FGUARD; return cu_cconvert<PB>( cu_cdiv_w<W>( cu_cconvert<W>(a), cu_cconvert<W>(b) ) ); }

/* ---- the functions ---- */
template<int W> __host__ __device__ static cu_fcomplex<W>
cu_cexp_w (const cu_fcomplex<W>&z){
  cu_freal<W> ex=cu_exp_w<W>(z.re), s,c; cu_sincos_w<W>(z.im,s,c);
  return cu_fcomplex<W>( cu_fmul<W>(ex,c), cu_fmul<W>(ex,s) );
}
template<int W> __host__ __device__ static cu_fcomplex<W>
cu_clog_w (const cu_fcomplex<W>&z){
  cu_freal<W> r2=cu_fmma<W>(z.re,z.re,z.im,z.im);             /* |z|^2 */
  cu_freal<W> lr=cu_scale2<W>( cu_log_w<W>(r2), -1 );         /* 0.5 log|z|^2 */
  return cu_fcomplex<W>( lr, cu_atan2_w<W>(z.im,z.re) );
}
template<int W> __host__ __device__ static cu_fcomplex<W>
cu_csqrt_w (const cu_fcomplex<W>&z){
  if (z.re.is_zero() && z.im.is_zero()) return cu_fcomplex<W>(z.re,z.im);
  cu_freal<W> mod=cu_sqrt_w<W>( cu_fmma<W>(z.re,z.re,z.im,z.im) );  /* |z| */
  if (z.re.sign>0){
    cu_freal<W> u=cu_sqrt_w<W>( cu_scale2<W>(cu_fadd<W>(mod,z.re),-1) );
    cu_freal<W> v=cu_fmul<W>( z.im, cu_recip_w<W>( cu_scale2<W>(u,1) ) );
    return cu_fcomplex<W>(u,v);
  } else {
    cu_freal<W> t=cu_sqrt_w<W>( cu_scale2<W>(cu_fsub<W>(mod,z.re),-1) );
    cu_freal<W> v=t; v.sign = t.is_zero()?1:z.im.sign;
    cu_freal<W> aim=z.im; aim.sign=1;
    cu_freal<W> u=cu_fmul<W>( aim, cu_recip_w<W>( cu_scale2<W>(t,1) ) );
    return cu_fcomplex<W>(u,v);
  }
}
template<int W> __host__ __device__ static cu_fcomplex<W>
cu_csin_w (const cu_fcomplex<W>&z){
  cu_freal<W> sr,cr; cu_sincos_w<W>(z.re,sr,cr);
  cu_freal<W> ch=cu_cosh_w<W>(z.im), sh=cu_sinh_w<W>(z.im);
  return cu_fcomplex<W>( cu_fmul<W>(sr,ch), cu_fmul<W>(cr,sh) );
}
template<int W> __host__ __device__ static cu_fcomplex<W>
cu_ccos_w (const cu_fcomplex<W>&z){
  cu_freal<W> sr,cr; cu_sincos_w<W>(z.re,sr,cr);
  cu_freal<W> ch=cu_cosh_w<W>(z.im), sh=cu_sinh_w<W>(z.im);
  return cu_fcomplex<W>( cu_fmul<W>(cr,ch), cu_neg<W>(cu_fmul<W>(sr,sh)) );
}
template<int W> __host__ __device__ static cu_fcomplex<W>
cu_csinh_w (const cu_fcomplex<W>&z){
  cu_freal<W> si,ci; cu_sincos_w<W>(z.im,si,ci);
  cu_freal<W> shr=cu_sinh_w<W>(z.re), chr=cu_cosh_w<W>(z.re);
  return cu_fcomplex<W>( cu_fmul<W>(shr,ci), cu_fmul<W>(chr,si) );
}
template<int W> __host__ __device__ static cu_fcomplex<W>
cu_ccosh_w (const cu_fcomplex<W>&z){
  cu_freal<W> si,ci; cu_sincos_w<W>(z.im,si,ci);
  cu_freal<W> shr=cu_sinh_w<W>(z.re), chr=cu_cosh_w<W>(z.re);
  return cu_fcomplex<W>( cu_fmul<W>(chr,ci), cu_fmul<W>(shr,si) );
}
template<int W> __host__ __device__ static cu_fcomplex<W>
cu_ctan_w (const cu_fcomplex<W>&z){ return cu_cdiv_w<W>( cu_csin_w<W>(z), cu_ccos_w<W>(z) ); }

/* ---- public wrappers ---- */
#define CU_CWRAP(name) \
template<int PB> __host__ __device__ static cu_fcomplex<PB> \
name (const cu_fcomplex<PB>&z){ const int W=PB+CU_FGUARD; \
  return cu_cconvert<PB>( name##_w<W>( cu_cconvert<W>(z) ) ); }
CU_CWRAP(cu_cexp)
CU_CWRAP(cu_clog)
CU_CWRAP(cu_csqrt)
CU_CWRAP(cu_csin)
CU_CWRAP(cu_ccos)
CU_CWRAP(cu_ctan)
CU_CWRAP(cu_csinh)
CU_CWRAP(cu_ccosh)
#undef CU_CWRAP

} /* namespace cu_fp */
#endif /* CU_MPC_CUDA_FCMATH_CUH */
