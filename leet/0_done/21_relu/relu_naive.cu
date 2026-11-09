#include <cuda_runtime.h>

// Naive ReLU: one thread per element, plain scalar loads/stores and a simple
// branch instead of fmaxf.  No vector types, no grid-stride loop -- the
// plainest correct form.

__global__ void relu_naive_kernel(const float* __restrict__ input,
                                  float* __restrict__ output, int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  const float v = input[i];
  output[i] = (v < 0.0f) ? 0.0f : v;
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
  if (N <= 0) return;
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  relu_naive_kernel<<<blocks, threads>>>(input, output, N);
  cudaDeviceSynchronize();
}
