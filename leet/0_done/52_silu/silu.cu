#include <cuda_runtime.h>

#include <cstdint>

// SiLU: out[i] = x / (1 + exp(-x)).
//
// N is at most 10,000 (the benchmark uses 50,000), so the kernel moves only
// 400 KB.  Like the other tiny elementwise challenges this one is
// launch-bound: an empty kernel costs 6.8 us on this GPU and a float4 copy of
// the same 400 KB costs 7.8 us; the SiLU math itself is a handful of
// instructions per element.  So the implementation is one launch, one float4
// per thread, no grid-stride loop, and the scalar tail is folded into the
// same kernel (a second launch would cost more than the tail itself).
//
// The starter's kernel body is empty, so it produces no output at all and is
// not a usable baseline.

#ifndef THREADS
#define THREADS 256
#endif

__device__ __forceinline__ float silu(float x) {
  // __expf + __fdividef instead of expf + IEEE division: the math costs 2.7 us
  // less (7.9 us vs 10.8 us, i.e. the copy floor), at a worst-case relative
  // error of ~9e-7 instead of ~2e-7 against a float64 reference.
  return __fdividef(x, 1.0f + __expf(-x));
}

__global__ void silu_vec_kernel(const float4* __restrict__ input4,
                                float4* __restrict__ output4, int n4, int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) {
    const float4 v = __ldg(&input4[i]);
    output4[i] = make_float4(silu(v.x), silu(v.y), silu(v.z), silu(v.w));
  } else if (i == n4) {
    // Scalar tail: the last N % 4 elements, handled by a single thread.
    const float* input = reinterpret_cast<const float*>(input4);
    float* output = reinterpret_cast<float*>(output4);
    for (int k = 4 * n4; k < N; ++k) output[k] = silu(input[k]);
  }
}

// Fallback for input/output pointers that are not 16-byte aligned.
__global__ void silu_scalar_kernel(const float* __restrict__ input,
                                   float* __restrict__ output, int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) output[i] = silu(__ldg(&input[i]));
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
    silu_vec_kernel<<<blocks, THREADS>>>(
        reinterpret_cast<const float4*>(input),
        reinterpret_cast<float4*>(output), n4, N);
  } else {
    const int blocks = (N + THREADS - 1) / THREADS;
    silu_scalar_kernel<<<blocks, THREADS>>>(input, output, N);
  }
  cudaDeviceSynchronize();
}
