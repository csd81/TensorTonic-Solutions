#include <cuda_runtime.h>

#include <cstdint>

// Leaky ReLU with alpha = 0.01: out[i] = x > 0 ? x : 0.01 * x.
//
// At the benchmark size (N = 50,000,000) the kernel reads 200 MB and writes
// 200 MB, so it is a pure read+write streaming problem: 400 MB at this GPU's
// ~82 GB/s read+write rate is ~4.86 ms, and a pure float4 copy of the same
// traffic measures the same.  The starter is already at that wall; the only
// things to add are the general-case mechanics (vectorized only when the
// pointers are aligned, a correct scalar tail) and the alignment check the
// starter lacks -- without it the starter faults with "misaligned address"
// when either pointer is not 16-byte aligned.

#ifndef THREADS
#define THREADS 256
#endif

__device__ __forceinline__ float leaky(float x) {
  return x > 0.0f ? x : 0.01f * x;
}

__global__ void leaky_vec_kernel(const float4* __restrict__ input4,
                                 float4* __restrict__ output4, int n4, int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) {
    const float4 v = __ldg(&input4[i]);
    output4[i] = make_float4(leaky(v.x), leaky(v.y), leaky(v.z), leaky(v.w));
  } else if (i == n4) {
    // Scalar tail: the last N % 4 elements, handled by a single thread.
    const float* input = reinterpret_cast<const float*>(input4);
    float* output = reinterpret_cast<float*>(output4);
    for (int k = 4 * n4; k < N; ++k) output[k] = leaky(input[k]);
  }
}

// Fallback for input/output pointers that are not 16-byte aligned.
__global__ void leaky_scalar_kernel(const float* __restrict__ input,
                                    float* __restrict__ output, int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) output[i] = leaky(__ldg(&input[i]));
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
  if (N <= 0) return;

  const bool aligned =
      ((reinterpret_cast<uintptr_t>(input) | reinterpret_cast<uintptr_t>(output)) &
       15u) == 0;
  if (aligned) {
    const int n4 = N >> 2;
    const int blocks = (n4 + THREADS) / THREADS;  // covers the tail thread
    leaky_vec_kernel<<<blocks, THREADS>>>(
        reinterpret_cast<const float4*>(input),
        reinterpret_cast<float4*>(output), n4, N);
  } else {
    const int blocks = (N + THREADS - 1) / THREADS;
    leaky_scalar_kernel<<<blocks, THREADS>>>(input, output, N);
  }
  cudaDeviceSynchronize();
}
