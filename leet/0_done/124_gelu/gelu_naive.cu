#include <cuda_runtime.h>
#include <math.h>

// Naive GELU: one thread per element, plain scalar load/store and erff.
// No vector types, no grid-stride loop -- the plainest correct form.

__global__ void gelu_naive_kernel(const float* __restrict__ X,
                                  float* __restrict__ Y, int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  const float x = X[i];
  Y[i] = 0.5f * x * (1.0f + erff(x * 0.70710678f));
}

// X, Y are device pointers
extern "C" void solve(const float* X, float* Y, int N) {
  if (N <= 0) return;
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  gelu_naive_kernel<<<blocks, threads>>>(X, Y, N);
  cudaDeviceSynchronize();
}
