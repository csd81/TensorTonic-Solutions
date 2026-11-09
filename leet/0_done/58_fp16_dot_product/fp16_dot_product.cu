// FP16 dot product (challenge 58).
//
// result = sum_i A[i] * B[i], A and B are half (FP16), accumulation must be
// FP32, the final result is converted back to half.  N <= 100,000,000;
// benchmark N = 1e8, i.e. 400 MB of reads and a single half of output, so the
// kernel is read-bound: the pure-read rate on this GPU is ~87 GB/s -> 4.6 ms.
//
// The starter measures 5.13 ms (78.0 GB/s).  It reads A and B with scalar
// 2-byte loads (a warp request covers only 64 bytes, half a 128-byte sector)
// and pays a cudaMalloc/cudaFree pair plus a single-thread final reduction
// over 2048 block sums per call.
//
// This version:
//
//  1. Loads 8 halves per thread per iteration as one float4 per input (a warp
//     request is four 512-byte transactions), converts each half2 with
//     __half22float2 and accumulates into four independent FP32 accumulators.
//  2. Reduces the block with warp shuffles and a shared-memory partial array,
//     then one atomicAdd per block into a cached single-float accumulator.
//  3. Converts the accumulator to half in a 1-thread kernel, so the solve is
//     three launches (memset of 4 bytes, dot, convert) and no allocations.
//
// Measured at N=1e8: 4.61 ms (86.7 GB/s) versus 5.13 ms (78.0 GB/s), 1.11x --
// essentially the memory wall.  Grid sweep: 1024 blocks 4.61 ms, 8192 blocks
// 4.62, 4096 4.65, 2048 4.75; half2 loads (4 bytes) 4.72; a two-float4
// unrolled variant 4.62.
//
// A scalar fallback covers inputs whose base pointers are not 16-byte
// aligned (cudaMalloc always is, but the signature does not require it).

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr int kBlocks = 1024;
constexpr int kMaxBlocks = 2048;  // when N is small

__device__ __forceinline__ void block_accumulate(float s, float* sum) {
  __shared__ float warp_sums[32];
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
  const int warp = threadIdx.x >> 5;
  if ((threadIdx.x & 31) == 0) warp_sums[warp] = s;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.0f;
    const int nw = (blockDim.x + 31) >> 5;
    for (int w = 0; w < nw; ++w) t += warp_sums[w];
    atomicAdd(sum, t);
  }
}

// Main path: float4 loads of 8 halves, 4 FP32 accumulators.
__global__ void dot_vec8(const float4* __restrict__ A4,
                         const float4* __restrict__ B4,
                         const half* __restrict__ A, const half* __restrict__ B,
                         float* __restrict__ sum, int n4, int N) {
  float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
    const float4 a = A4[i];
    const float4 b = B4[i];
    const __half2* ah = reinterpret_cast<const __half2*>(&a);
    const __half2* bh = reinterpret_cast<const __half2*>(&b);
    float2 fa = __half22float2(ah[0]);
    float2 fb = __half22float2(bh[0]);
    s0 = fmaf(fa.x, fb.x, s0);
    s1 = fmaf(fa.y, fb.y, s1);
    fa = __half22float2(ah[1]);
    fb = __half22float2(bh[1]);
    s2 = fmaf(fa.x, fb.x, s2);
    s3 = fmaf(fa.y, fb.y, s3);
    fa = __half22float2(ah[2]);
    fb = __half22float2(bh[2]);
    s0 = fmaf(fa.x, fb.x, s0);
    s1 = fmaf(fa.y, fb.y, s1);
    fa = __half22float2(ah[3]);
    fb = __half22float2(bh[3]);
    s2 = fmaf(fa.x, fb.x, s2);
    s3 = fmaf(fa.y, fb.y, s3);
  }
  // Tail (N mod 8 <= 7 elements): one thread handles it.
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    for (int i = n4 * 8; i < N; ++i)
      s0 = fmaf(__half2float(A[i]), __half2float(B[i]), s0);
  }
  block_accumulate((s0 + s1) + (s2 + s3), sum);
}

// Fallback for unaligned inputs or N < 8: scalar half loads.
__global__ void dot_scalar(const half* __restrict__ A, const half* __restrict__ B,
                           float* __restrict__ sum, int N) {
  float s = 0.0f;
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += stride)
    s = fmaf(__half2float(A[i]), __half2float(B[i]), s);
  block_accumulate(s, sum);
}

__global__ void convert_kernel(const float* sum, half* result) {
  *result = __float2half(*sum);
}

float* g_sum = nullptr;  // 1-element accumulator, allocated once

}  // namespace

// A, B, result are device pointers.
extern "C" void solve(const half* A, const half* B, half* result, int N) {
  if (N <= 0) return;
  if (!g_sum) cudaMalloc(&g_sum, sizeof(float));
  cudaMemsetAsync(g_sum, 0, sizeof(float));

  const bool aligned = ((reinterpret_cast<uintptr_t>(A) & 15) == 0) &&
                       ((reinterpret_cast<uintptr_t>(B) & 15) == 0);
  if (aligned && N >= 8) {
    const int n4 = N >> 3;
    int blocks = (n4 + kThreads - 1) / kThreads;
    if (blocks > kBlocks) blocks = kBlocks;
    dot_vec8<<<blocks, kThreads>>>(reinterpret_cast<const float4*>(A),
                                   reinterpret_cast<const float4*>(B), A, B,
                                   g_sum, n4, N);
  } else {
    int blocks = (N + kThreads - 1) / kThreads;
    if (blocks > kMaxBlocks) blocks = kMaxBlocks;
    dot_scalar<<<blocks, kThreads>>>(A, B, g_sum, N);
  }
  convert_kernel<<<1, 1>>>(g_sum, result);
}
