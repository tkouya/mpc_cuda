/* bench_mpn_mul.cu -- microbenchmark: 16-limb (1024-bit) mpn schoolbook
 * multiply, generic pointer-based (current lib idiom) vs fully-unrolled
 * register-resident.  Self-contained: no mpc_cuda library needed.
 *
 *   nvcc -arch=sm_121 -O3 -fmad=false -o /tmp/bench_mpn_mul tools/bench_mpn_mul.cu
 *
 * Both kernels compute, for each of N operand pairs, the full 32-limb product
 * of two 16-limb numbers, and XOR-reduce the result so the work cannot be
 * dead-code eliminated.  We verify the two produce identical products, then
 * time each over many grid-stride iterations.
 */
#include <cstdio>
#include <cstdint>
#include <cstdlib>

typedef unsigned long long limb;   /* 64-bit limb, matches GMP_NUMB_BITS==64 */

#define NLIMB 16                    /* 1024-bit significand */
#define N     (1 << 16)             /* number of operand pairs */
#define REPS  64                    /* grid-stride repetitions for timing */

/* ---- shared 64x64->128 building block (same inline PTX as the library) ---- */
__device__ __forceinline__ void
umul (limb &hi, limb &lo, limb a, limb b)
{
  asm ("mul.lo.u64 %0, %2, %3;\n\t"
       "mul.hi.u64 %1, %2, %3;"
       : "=l"(lo), "=l"(hi) : "l"(a), "l"(b));
}

/* ===================== (A) generic pointer-based ===================== *
 * Mirrors the library: mpn_mul_1 then (n-1) x mpn_addmul_1, carry kept in a
 * register across iterations, operands addressed through pointers.  In the
 * kernel rp/up/vp live in per-thread GLOBAL scratch (like the arena slabs). */
__device__ limb
g_mul_1 (limb *rp, const limb *up, int n, limb vl)
{
  limb cl = 0;
  for (int i = 0; i < n; i++)
    {
      limb lo, hi;
      asm ("mad.lo.cc.u64 %0, %2, %3, %4;\n\t"
           "madc.hi.u64   %1, %2, %3, 0;"
           : "=&l"(lo), "=&l"(hi) : "l"(up[i]), "l"(vl), "l"(cl));
      rp[i] = lo; cl = hi;
    }
  return cl;
}
__device__ limb
g_addmul_1 (limb *rp, const limb *up, int n, limb vl)
{
  limb cl = 0;
  for (int i = 0; i < n; i++)
    {
      limb lo, hi, rl = rp[i];
      asm ("mad.lo.cc.u64 %0, %2, %3, %4;\n\t"
           "madc.hi.u64   %1, %2, %3, 0;\n\t"
           "add.cc.u64    %0, %0, %5;\n\t"
           "addc.u64      %1, %1, 0;"
           : "=&l"(lo), "=&l"(hi) : "l"(up[i]), "l"(vl), "l"(cl), "l"(rl));
      rp[i] = lo; cl = hi;
    }
  return cl;
}
__device__ void
g_mul (limb *rp, const limb *up, const limb *vp, int n)
{
  rp[n] = g_mul_1 (rp, up, n, vp[0]);
  for (int j = 1; j < n; j++)
    rp[n + j] = g_addmul_1 (rp + j, up, n, vp[j]);
}

/* ===================== (B) register-resident, unrolled ===================== *
 * 16x16 schoolbook fully unrolled; all limbs in local arrays that ptxas can
 * promote to registers (constant indices).  No global round-trips mid-multiply. */
__device__ __forceinline__ void
r_mul16 (limb *out, const limb *A, const limb *B)
{
  limb u[NLIMB], v[NLIMB], r[2 * NLIMB];
#pragma unroll
  for (int i = 0; i < NLIMB; i++) { u[i] = A[i]; v[i] = B[i]; }
#pragma unroll
  for (int i = 0; i < 2 * NLIMB; i++) r[i] = 0;

#pragma unroll
  for (int j = 0; j < NLIMB; j++)
    {
      limb cl = 0;
#pragma unroll
      for (int i = 0; i < NLIMB; i++)
        {
          limb lo, hi, rl = r[i + j];
          asm ("mad.lo.cc.u64 %0, %2, %3, %4;\n\t"
               "madc.hi.u64   %1, %2, %3, 0;\n\t"
               "add.cc.u64    %0, %0, %5;\n\t"
               "addc.u64      %1, %1, 0;"
               : "=&l"(lo), "=&l"(hi)
               : "l"(u[i]), "l"(v[j]), "l"(cl), "l"(rl));
          r[i + j] = lo; cl = hi;
        }
      r[NLIMB + j] = cl;   /* r[NLIMB+j] is still 0 here -> plain store */
    }
#pragma unroll
  for (int i = 0; i < 2 * NLIMB; i++) out[i] = r[i];
}

/* ===== (A') library-faithful: RUNTIME n + GLOBAL scratch ===== *
 * This is what mpfr_mul actually does at runtime: the limb count is opaque to
 * the compiler (no unroll) and the product accumulates in a global-memory
 * arena slab.  __device__ runtime_n defeats constant propagation. */
__device__ int runtime_n = NLIMB;
__device__ void
g_mul_rt (limb *rp, const limb *up, const limb *vp, int n)
{
  rp[n] = g_mul_1 (rp, up, n, vp[0]);
  for (int j = 1; j < n; j++)
    rp[n + j] = g_addmul_1 (rp + j, up, n, vp[j]);
}

/* ---------------- kernels ---------------- */
__global__ void
kern_libfaithful (const limb *U, const limb *V, limb *acc, limb *slab, int npair)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int n = runtime_n;
  limb *scratch = slab + (size_t)tid * 2 * NLIMB;   /* global arena slab */
  for (int rep = 0; rep < REPS; rep++)
    for (int k = tid; k < npair; k += stride)
      {
        g_mul_rt (scratch, U + (size_t)k * NLIMB, V + (size_t)k * NLIMB, n);
        limb x = 0;
        for (int i = 0; i < 2 * n; i++) x ^= scratch[i];
        acc[k] ^= x;
      }
}

/* const n (unrolls) but scratch forced to GLOBAL -> isolates memory effect */
__global__ void
kern_constn_global (const limb *U, const limb *V, limb *acc, limb *slab, int npair)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  limb *scratch = slab + (size_t)tid * 2 * NLIMB;
  for (int rep = 0; rep < REPS; rep++)
    for (int k = tid; k < npair; k += stride)
      {
        g_mul (scratch, U + (size_t)k * NLIMB, V + (size_t)k * NLIMB, NLIMB);
        limb x = 0;
#pragma unroll
        for (int i = 0; i < 2 * NLIMB; i++) x ^= scratch[i];
        acc[k] ^= x;
      }
}

__global__ void
kern_generic (const limb *U, const limb *V, limb *acc, int npair)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  limb scratch[2 * NLIMB];
  for (int rep = 0; rep < REPS; rep++)
    for (int k = tid; k < npair; k += stride)
      {
        g_mul (scratch, U + (size_t)k * NLIMB, V + (size_t)k * NLIMB, NLIMB);
        limb x = 0;
#pragma unroll
        for (int i = 0; i < 2 * NLIMB; i++) x ^= scratch[i];
        acc[k] ^= x;
      }
}
__global__ void
kern_register (const limb *U, const limb *V, limb *acc, int npair)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  limb out[2 * NLIMB];
  for (int rep = 0; rep < REPS; rep++)
    for (int k = tid; k < npair; k += stride)
      {
        r_mul16 (out, U + (size_t)k * NLIMB, V + (size_t)k * NLIMB);
        limb x = 0;
#pragma unroll
        for (int i = 0; i < 2 * NLIMB; i++) x ^= out[i];
        acc[k] ^= x;
      }
}

/* verify products match (single pass, separate buffers) */
__global__ void
kern_verify (const limb *U, const limb *V, limb *Rg, limb *Rr, int npair)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (int k = tid; k < npair; k += stride)
    {
      g_mul (Rg + (size_t)k * 2 * NLIMB, U + (size_t)k * NLIMB,
             V + (size_t)k * NLIMB, NLIMB);
      r_mul16 (Rr + (size_t)k * 2 * NLIMB, U + (size_t)k * NLIMB,
               V + (size_t)k * NLIMB);
    }
}

static limb xorshift (limb *s){ limb x=*s; x^=x<<13; x^=x>>7; x^=x<<17; return *s=x; }

int main (void)
{
  size_t npair = N;
  limb *U =(limb*)malloc(npair*NLIMB*sizeof(limb));
  limb *V =(limb*)malloc(npair*NLIMB*sizeof(limb));
  limb seed = 0x123456789ULL;
  for (size_t i=0;i<npair*NLIMB;i++){ U[i]=xorshift(&seed); V[i]=xorshift(&seed);}

  limb *dU,*dV,*dAcc,*dRg,*dRr;
  cudaMalloc(&dU,npair*NLIMB*sizeof(limb));
  cudaMalloc(&dV,npair*NLIMB*sizeof(limb));
  cudaMalloc(&dAcc,npair*sizeof(limb));
  cudaMalloc(&dRg,npair*2*NLIMB*sizeof(limb));
  cudaMalloc(&dRr,npair*2*NLIMB*sizeof(limb));
  limb *dSlab; /* per-thread global scratch for the lib-faithful kernel */
  cudaMemcpy(dU,U,npair*NLIMB*sizeof(limb),cudaMemcpyHostToDevice);
  cudaMemcpy(dV,V,npair*NLIMB*sizeof(limb),cudaMemcpyHostToDevice);
  cudaMemset(dAcc,0,npair*sizeof(limb));

  int block=128, grid=0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&grid,kern_register,block,0);
  int sm=0; cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,0);
  grid = grid*sm;  /* fill the device */

  /* ---- verify correctness ---- */
  kern_verify<<<grid,block>>>(dU,dV,dRg,dRr,npair);
  cudaDeviceSynchronize();
  limb *Rg=(limb*)malloc(npair*2*NLIMB*sizeof(limb));
  limb *Rr=(limb*)malloc(npair*2*NLIMB*sizeof(limb));
  cudaMemcpy(Rg,dRg,npair*2*NLIMB*sizeof(limb),cudaMemcpyDeviceToHost);
  cudaMemcpy(Rr,dRr,npair*2*NLIMB*sizeof(limb),cudaMemcpyDeviceToHost);
  size_t mism=0; for(size_t i=0;i<npair*2*NLIMB;i++) if(Rg[i]!=Rr[i]) mism++;
  printf("verify: %s (%zu mismatched limbs of %zu)\n",
         mism?"FAIL":"OK", mism, npair*2*NLIMB);

  cudaMalloc(&dSlab,(size_t)grid*block*2*NLIMB*sizeof(limb));

  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  float tg=0,tr=0,tl=0;
  /* warmup + time library-faithful (runtime n, global scratch) */
  kern_libfaithful<<<grid,block>>>(dU,dV,dAcc,dSlab,npair); cudaDeviceSynchronize();
  cudaEventRecord(a);
  kern_libfaithful<<<grid,block>>>(dU,dV,dAcc,dSlab,npair);
  cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&tl,a,b);
  /* warmup + time const-n + global scratch (isolates memory effect) */
  float tcg=0;
  kern_constn_global<<<grid,block>>>(dU,dV,dAcc,dSlab,npair); cudaDeviceSynchronize();
  cudaEventRecord(a);
  kern_constn_global<<<grid,block>>>(dU,dV,dAcc,dSlab,npair);
  cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&tcg,a,b);
  /* warmup + time generic */
  kern_generic<<<grid,block>>>(dU,dV,dAcc,npair); cudaDeviceSynchronize();
  cudaEventRecord(a);
  kern_generic<<<grid,block>>>(dU,dV,dAcc,npair);
  cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&tg,a,b);
  /* warmup + time register */
  kern_register<<<grid,block>>>(dU,dV,dAcc,npair); cudaDeviceSynchronize();
  cudaEventRecord(a);
  kern_register<<<grid,block>>>(dU,dV,dAcc,npair);
  cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&tr,a,b);

  double work = (double)npair*REPS;
  printf("grid=%d block=%d  pairs=%zu reps=%d  (%.1f M muls each)\n",
         grid,block,npair,REPS,work/1e6);
  printf("lib-faith: %8.3f ms  (%.1f M mul/s)  [runtime n + global scratch]\n", tl, work/1e3/tl);
  printf("constN-gl: %8.3f ms  (%.1f M mul/s)  [const n, global scratch]\n", tcg, work/1e3/tcg);
  printf("generic  : %8.3f ms  (%.1f M mul/s)  [const n, regs]\n", tg, work/1e3/tg);
  printf("register : %8.3f ms  (%.1f M mul/s)  [unrolled regs]\n", tr, work/1e3/tr);
  printf("speedup register vs lib-faithful : %.2fx\n", tl/tr);
  printf("speedup register vs generic      : %.2fx\n", tg/tr);
  return 0;
}
