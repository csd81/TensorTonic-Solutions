#include <cuda_runtime.h>

#include <cstdint>

// ReLU: out[i] = max(0, in[i]).
//
// At the benchmark size (N = 25,000,000) the kernel reads 100 MB and writes
// 100 MB, so it is a pure read+write streaming problem.  A plain float4 copy
// over the same 200 MB measures 2.44 ms on this GPU (~82 GB/s in the same
// run), and the starter -- one scalar element per thread -- is already at
// that wall (2.45 ms).  There is no algorithmic headroom: every byte of the
// input must be read and every byte of the output written.
//
// This version changes only the mechanics: one float4 per thread (4x fewer
// threads, 512 B per warp load/store instead of 128 B), a scalar kernel for
// pointers that are not 16-byte aligned, and an in-kernel scalar tail for
// N not divisible by 4.  The measured difference from the starter is inside
// run-to-run noise, which is what a bandwidth-bound kernel at the wall looks
// like.  Streaming cache hints (__ldcs/__stcs) were measured and are slightly
// slower, so plain accesses are used.

#ifndef THREADS
#define THREADS 512
#endif

__global__ void relu_vec_kernel(const float4* __restrict__ input4,
                                float4* __restrict__ output4, int n4, int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) {
    const float4 v = __ldg(&input4[i]);
    output4[i] = make_float4(fmaxf(0.0f, v.x), fmaxf(0.0f, v.y),
                             fmaxf(0.0f, v.z), fmaxf(0.0f, v.w));
  } else if (i == n4) {
    // Scalar tail: the last N % 4 elements, handled by a single thread.
    const float* input = reinterpret_cast<const float*>(input4);
    float* output = reinterpret_cast<float*>(output4);
    for (int k = 4 * n4; k < N; ++k) output[k] = fmaxf(0.0f, input[k]);
  }
}

// Fallback for input/output pointers that are not 16-byte aligned.
__global__ void relu_scalar_kernel(const float* __restrict__ input,
                                   float* __restrict__ output, int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) output[i] = fmaxf(0.0f, __ldg(&input[i]));
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
  if (N <= 0) return;

  const bool aligned =
      ((reinterpret_cast<uintptr_t>(input) | reinterpret_cast<uintptr_t>(output)) &
       15u) == 0;
  if (aligned) {
    const int n4 = N >> 2;
    const int blocks = (n4 + 1 + THREADS - 1) / THREADS;
    relu_vec_kernel<<<blocks, THREADS>>>(
        reinterpret_cast<const float4*>(input),
        reinterpret_cast<float4*>(output), n4, N);
  } else {
    const int blocks = (N + THREADS - 1) / THREADS;
    relu_scalar_kernel<<<blocks, THREADS>>>(input, output, N);
  }
  cudaDeviceSynchronize();
}
