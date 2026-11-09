#include <cuda_runtime.h>

// Naive interleave: one thread per input index, two scalar loads and two
// scalar stores.  No vector types, no grid-stride loop -- the plainest
// correct form (and the starter's shape).

__global__ void interleave_naive_kernel(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ output, int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  output[2 * i] = A[i];
  output[2 * i + 1] = B[i];
}

// A, B, output are device pointers
extern "C" void solve(const float* A, const float* B, float* output, int N) {
  if (N <= 0) return;
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  interleave_naive_kernel<<<blocks, threads>>>(A, B, output, N);
  cudaDeviceSynchronize();
}
