#include <cuda_runtime.h>

// Naive leaky ReLU: one thread per element, plain scalar load/store and a
// ternary.  No vector types, no grid-stride loop -- the plainest correct
// form.

__global__ void leaky_naive_kernel(const float* __restrict__ input,
                                   float* __restrict__ output, int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  const float x = input[i];
  output[i] = x > 0.0f ? x : 0.01f * x;
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
  if (N <= 0) return;
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  leaky_naive_kernel<<<blocks, threads>>>(input, output, N);
  cudaDeviceSynchronize();
}
