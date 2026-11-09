#include <cuda_runtime.h>

#include <cstdint>

// Exact (erf-based) GELU: out[i] = 0.5*x*(1 + erf(x/sqrt(2))).
//
// At the benchmark (N = 50,000,000) the kernel reads 200 MB and writes 200 MB,
// so it is memory-bound: a pure float4 copy of the same 400 MB measures
// 4.87 ms (82 GB/s).  The scalar erff form measures 4.88 ms — the
// transcendental is fully hidden behind the loads at this size (unlike the
// smaller GEGLU/SiLU shapes).  float4 measures 4.861 ms, the copy floor, so
// the shipped kernel is the float4 form; the difference from scalar is inside
// noise but it is never slower.
//
// The starter does not compile as shipped: it launches with `blocksGrid`
// while the variable is `blocksPerGrid`.  The timing above is from a copy
// with that typo fixed (nothing else changed).

#ifndef THREADS
#define THREADS 256
#endif

__device__ __forceinline__ float gelu(float x) {
  return 0.5f * x * (1.0f + erff(x * 0.70710678f));
}

__global__ void gelu_vec_kernel(const float4* __restrict__ X,
                                float4* __restrict__ Y, int n4, int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) {
    const float4 v = __ldg(&X[i]);
    Y[i] = make_float4(gelu(v.x), gelu(v.y), gelu(v.z), gelu(v.w));
  } else if (i == n4) {
    // Scalar tail: the last N % 4 elements, handled by a single thread.
    const float* x = reinterpret_cast<const float*>(X);
    float* y = reinterpret_cast<float*>(Y);
    for (int k = 4 * n4; k < N; ++k) y[k] = gelu(x[k]);
  }
}

// Fallback for input/output pointers that are not 16-byte aligned.
__global__ void gelu_scalar_kernel(const float* __restrict__ X,
                                   float* __restrict__ Y, int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) Y[i] = gelu(__ldg(&X[i]));
}

// X, Y are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* X, float* Y, int N) {
  if (N <= 0) return;

  const bool aligned =
      ((reinterpret_cast<uintptr_t>(X) | reinterpret_cast<uintptr_t>(Y)) & 15u) == 0;
  if (aligned) {
    const int n4 = N >> 2;
    const int blocks = (n4 + THREADS) / THREADS;  // covers the tail thread
    gelu_vec_kernel<<<blocks, THREADS>>>(
        reinterpret_cast<const float4*>(X), reinterpret_cast<float4*>(Y), n4, N);
  } else {
    const int blocks = (N + THREADS - 1) / THREADS;
    gelu_scalar_kernel<<<blocks, THREADS>>>(X, Y, N);
  }
  cudaDeviceSynchronize();
}
