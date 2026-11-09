#include <cuda_runtime.h>

// Naive matrix copy: one thread per element, plain scalar load and store.
// No vector types, no grid-stride loop -- the plainest correct form.

__global__ void copy_naive_kernel(const float* __restrict__ a,
                                  float* __restrict__ b, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) b[i] = a[i];
}

// A, B are device pointers
extern "C" void solve(const float* A, float* B, int N) {
  if (N <= 0) return;
  const int total = N * N;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  copy_naive_kernel<<<blocks, threads>>>(A, B, total);
  cudaDeviceSynchronize();
}
