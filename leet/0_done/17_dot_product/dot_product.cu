#include <cuda_runtime.h>

#include <cstdint>

// Dot product of two float vectors.
//
// At the benchmark size (N = 4,194,304) the kernel must read 32 MB and writes
// four bytes, so it is purely a bandwidth question: 32 MB at this GPU's
// practical 87 GB/s is ~0.39 ms.  The starter already gets 0.397 ms (84.5
// GB/s) with scalar loads, a grid-stride loop, a warp/block reduction and one
// atomic per block -- essentially the roofline.
//
// This version only changes the mechanics: float4 loads (512 B per warp
// instruction instead of 128 B), the same reduction, and a scalar grid-stride
// tail for N not divisible by 4.  The measured difference is inside run-to-run
// noise, which is what a bandwidth-bound kernel at 97% of peak looks like.

#ifndef THREADS
#define THREADS 256
#endif
#ifndef MAX_BLOCKS
#define MAX_BLOCKS 1024
#endif

__device__ __forceinline__ float warp_reduce(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_down_sync(0xFFFFFFFFu, v, off);
  }
  return v;
}

__device__ __forceinline__ float block_reduce(float v) {
  __shared__ float shared[THREADS / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  v = warp_reduce(v);
  if (lane == 0) shared[warp] = v;
  __syncthreads();
  v = (threadIdx.x < THREADS / 32) ? shared[lane] : 0.0f;
  if (warp == 0) v = warp_reduce(v);
  return v;
}

__global__ void dot_kernel(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ result, int N, int n4,
                           int vec_ok) {
  float sum = 0.0f;
  const int stride = blockDim.x * gridDim.x;
  const int gtid = blockIdx.x * blockDim.x + threadIdx.x;

  if (vec_ok) {
    const float4* __restrict__ A4 = reinterpret_cast<const float4*>(A);
    const float4* __restrict__ B4 = reinterpret_cast<const float4*>(B);
    for (int i = gtid; i < n4; i += stride) {
      const float4 a = A4[i];
      const float4 b = B4[i];
      sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
  }
  // Scalar tail (elements past the last full float4).
  for (int i = (vec_ok ? n4 * 4 : 0) + gtid; i < N; i += stride) {
    sum += A[i] * B[i];
  }

  sum = block_reduce(sum);
  if (threadIdx.x == 0) atomicAdd(result, sum);
}

// A, B, result are device pointers; the result is written to result[0].
extern "C" void solve(const float* A, const float* B, float* result, int N) {
  if (N <= 0) return;
  cudaMemsetAsync(result, 0, sizeof(float));

  const bool aligned =
      ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)) &
       15u) == 0;
  const int n4 = aligned ? (N >> 2) : 0;

  int blocks = (N + THREADS - 1) / THREADS;
  if (blocks > MAX_BLOCKS) blocks = MAX_BLOCKS;
  dot_kernel<<<blocks, THREADS>>>(A, B, result, N, n4, aligned ? 1 : 0);
  cudaDeviceSynchronize();
}
