/* cuda_host_shim.h -- lets a simple CUDA test program also build as plain C++
 * (CPU-only, no CUDA toolkit): kernels become host functions launched in a
 * loop, device memory is host memory.  Under nvcc it only defines CU_LAUNCH.
 *
 *   CU_LAUNCH(kern<PB>, n, args...)   -- one "thread" per index 0..n-1
 *
 * Only what the tools/test_f*math*.cu programs use is provided.
 */
#ifndef CU_CUDA_HOST_SHIM_H
#define CU_CUDA_HOST_SHIM_H

#if defined(__CUDACC__)
#  define CU_LAUNCH(kern, n, ...) kern<<<((n)+127)/128,128>>>(__VA_ARGS__)
#else
#  include <cstdlib>
#  include <cstring>
#  define __global__
struct cu_shim_dim3 { unsigned x, y, z; };
static cu_shim_dim3 blockIdx = {0,0,0}, blockDim = {1,1,1}, threadIdx = {0,0,0};
typedef int cudaError_t;
enum { cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2, cudaLimitStackSize = 0 };
template<class T> static inline cudaError_t cudaMalloc (T **p, size_t n)
{ *p = (T*)malloc (n); return *p ? 0 : 2; }
static inline cudaError_t cudaMemcpy (void *d, const void *s, size_t n, int)
{ memcpy (d, s, n); return 0; }
static inline cudaError_t cudaFree (void *p) { free (p); return 0; }
static inline cudaError_t cudaDeviceSynchronize () { return 0; }
static inline cudaError_t cudaDeviceSetLimit (int, size_t) { return 0; }
static inline const char *cudaGetErrorString (cudaError_t) { return "host shim error"; }
#  define CU_LAUNCH(kern, n, ...) do {                                      \
     for (unsigned cu_shim_i = 0; cu_shim_i < (unsigned)(n); cu_shim_i++)   \
       { blockIdx.x = cu_shim_i; kern (__VA_ARGS__); }                      \
   } while (0)
#endif

#endif /* CU_CUDA_HOST_SHIM_H */
