#include <cuda_runtime.h>

// Naive dot product: one GPU thread per element, each multiplying its pair and
// adding the product straight into result[0] with an atomicAdd.  No shared
// memory, no warp intrinsics, no vector types, no grid-stride loop -- the
// plainest correct form.
__global__ void dot_product_naive_kernel(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ result, int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    atomicAdd(result, A[i] * B[i]);
  }
}

// A, B, result are device pointers; the result is written to result[0].
extern "C" void solve(const float* A, const float* B, float* result, int N) {
  if (N <= 0) return;
  cudaMemsetAsync(result, 0, sizeof(float));
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  dot_product_naive_kernel<<<blocks, threads>>>(A, B, result, N);
  cudaDeviceSynchronize();
}
