/* cu_fwarp.cuh -- warp-cooperative (coalesced) loads and stores of arrays of
 * cu_freal<PB> / cu_fcomplex<PB> (or any trivially copyable type whose size
 * is a multiple of 8 bytes) on the GPU.
 *
 * An array of these structs, read one element per thread, is accessed with
 * 8- or 16-byte loads at a stride of sizeof(T) = 16+8N bytes: every warp-wide
 * load instruction touches 32 separate sectors, which caps the streaming
 * bandwidth far below the device's (H100 NVL: ~1-1.6 TB/s of ~3.3).  Here a
 * warp moves its tile of 32 consecutive elements as one contiguous block of
 * 16-byte (or 8-byte) words -- fully coalesced -- through a per-warp
 * shared-memory buffer of 32 elements, and each lane then takes its own
 * element from shared memory (store: the reverse).  Memory-bound
 * element-wise kernels on large arrays gain up to ~1.7x (H100); kernels whose
 * time is dominated by arithmetic (complex multiply from ~1024 bits) do not.
 *
 *   #include "mpc_cuda.cuh"                 // or "mpc_cuda/cu_fwarp.cuh"
 *   using namespace cu_fp;
 *   typedef cu_freal<1024> F;
 *   constexpr int TPB = 128;                // threads per block (multiple of 32)
 *
 *   __global__ void axpy (long n, F a, const F *x, const F *y, F *r)
 *   {
 *     __shared__ cu_fwarp_buf<F, TPB/32> buf;           // 32 elements per warp
 *     F *wb = buf.warp ();                              // this warp's slice
 *     for (long base = cu_warp_first_tile (); base < n; base += cu_warp_tile_stride ())
 *       {                                               // base is warp-uniform
 *         F xi = cu_warp_load (x, base, n, wb);         // = x[base + lane]
 *         F yi = cu_warp_load (y, base, n, wb);
 *         cu_warp_store (r, base, n, cu_fadd<1024> (cu_fmul<1024> (a, xi), yi), wb);
 *       }
 *   }
 *
 * Rules: all 32 lanes of the warp call each function together with the same
 * base (a warp-uniform loop such as the one above); lane l handles element
 * base + l; for base + l >= n the loaded value is unspecified and the store
 * is skipped.
 *
 * Shared memory: cu_fwarp_bytes<T>() = 32*sizeof(T) per warp -- 4.6 KB at
 * 1024-bit real, 17 KB at 2048-bit complex or 4096-bit real.  A block may
 * declare at most 48 KB statically (cu_fwarp_buf<T, warps>); beyond that use
 * dynamic shared memory, which cu_fwarp_prepare() enables on the host:
 *
 *   __global__ void k (...) { F *wb = cu_fwarp_dyn<F> (); ... }
 *   size_t sm = cu_fwarp_prepare<F> (k, TPB);           // sets the attribute
 *   k<<<grid, TPB, sm>>> (...);
 *
 * The 16-byte path needs 16-byte-aligned arrays (cudaMalloc'd arrays of an
 * aligned T: cu_freal/cu_fcomplex with an even limb count), otherwise 8-byte
 * words are used.
 */
#ifndef CU_MPC_CUDA_FWARP_CUH
#define CU_MPC_CUDA_FWARP_CUH

#if defined(__CUDACC__)
#include "mpc_cuda/cu_freal.cuh"

namespace cu_fp {

/* bytes of shared memory one warp needs for T */
template<class T> __host__ __device__ constexpr size_t cu_fwarp_bytes () { return 32*sizeof(T); }

/* static per-block storage for W warps: raw, suitably aligned bytes (no
 * constructors run in shared memory); warp() is the calling warp's slice */
template<class T, int W> struct cu_fwarp_buf {
  alignas(16) unsigned char raw[W*cu_fwarp_bytes<T>()];
  __device__ __forceinline__ T *warp ()
  { return reinterpret_cast<T*>(raw) + 32*(threadIdx.x>>5); }
};

/* the calling warp's slice of dynamic shared memory laid out as
 * cu_fwarp_bytes<T>() per warp from offset 0 */
extern __shared__ __align__(16) unsigned char cu_fwarp_dsmem[];
template<class T> __device__ __forceinline__ static T *cu_fwarp_dyn ()
{ return reinterpret_cast<T*>(cu_fwarp_dsmem) + 32*(threadIdx.x>>5); }

/* host: allow `kernel` the dynamic shared memory of cu_fwarp_dyn<T> for
 * blocks of `threads` threads and return that size (the third launch
 * parameter); 0 if the device cannot provide it */
template<class T, class K> static size_t cu_fwarp_prepare (K kernel, int threads)
{
  const size_t sm = (size_t)((threads+31)/32)*cu_fwarp_bytes<T>();
  if (cudaFuncSetAttribute (kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm) != cudaSuccess)
    { cudaGetLastError (); return 0; }
  return sm;
}

/* first tile of this warp and the tile stride of a grid-stride loop
 * (1-D launch, blockDim.x a multiple of 32) */
__device__ __forceinline__ static long cu_warp_first_tile ()
{ return ((long)blockIdx.x*blockDim.x + (threadIdx.x & ~31u)); }
__device__ __forceinline__ static long cu_warp_tile_stride ()
{ return (long)gridDim.x*blockDim.x; }

namespace detail {
template<class T> struct cu_fwarp_word {
  static_assert (sizeof(T)%8==0, "cu_fwarp: element size must be a multiple of 8 bytes");
  static const bool W16 = (sizeof(T)%16==0) && (alignof(T)>=16);
};
/* copy `bytes` bytes from s to d with the 32 lanes, 16- or 8-byte words */
template<bool W16> __device__ __forceinline__ static void
cu_fwarp_copy (void *d, const void *s, long bytes, int lane)
{
  if constexpr (W16)
    {
      uint4 *dd = (uint4*)d; const uint4 *ss = (const uint4*)s;
      const long nw = bytes>>4;
      for (long k=lane;k<nw;k+=32) dd[k]=ss[k];
    }
  else
    {
      unsigned long long *dd = (unsigned long long*)d; const unsigned long long *ss = (const unsigned long long*)s;
      const long nw = bytes>>3;
      for (long k=lane;k<nw;k+=32) dd[k]=ss[k];
    }
}
} /* namespace detail */

/* lane l returns p[base+l] (unspecified if base+l >= n); wb: this warp's
 * buffer of 32 elements.  Whole warp, warp-uniform base. */
template<class T> __device__ __forceinline__ static T
cu_warp_load (const T *p, long base, long n, T *wb)
{
  const int lane = threadIdx.x & 31;
  const long cnt = (n-base < 32) ? n-base : 32;
  detail::cu_fwarp_copy<detail::cu_fwarp_word<T>::W16> (wb, p+base, cnt*(long)sizeof(T), lane);
  __syncwarp ();
  T v = wb[lane];
  __syncwarp ();                                   /* the buffer may be reused */
  return v;
}

/* p[base+l] = v for lane l (skipped where base+l >= n).  Whole warp,
 * warp-uniform base. */
template<class T> __device__ __forceinline__ static void
cu_warp_store (T *p, long base, long n, const T &v, T *wb)
{
  const int lane = threadIdx.x & 31;
  const long cnt = (n-base < 32) ? n-base : 32;
  wb[lane] = v;
  __syncwarp ();
  detail::cu_fwarp_copy<detail::cu_fwarp_word<T>::W16> (p+base, wb, cnt*(long)sizeof(T), lane);
  __syncwarp ();
}

} /* namespace cu_fp */
#endif /* __CUDACC__ */
#endif /* CU_MPC_CUDA_FWARP_CUH */
