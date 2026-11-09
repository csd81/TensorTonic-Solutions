#include <cuda_runtime.h>

// Naive rainbow table: one thread per element with a plain scalar load, the
// R-round FNV-1a chain, and a scalar store.  No vector types, no ILP across
// elements -- the plainest correct form.

__device__ __forceinline__ unsigned int fnv1a_naive(unsigned int x) {
  const unsigned int P = 16777619u;
  unsigned int h = 2166136261u;
  for (int b = 0; b < 4; ++b) h = (h ^ ((x >> (b * 8)) & 0xFFu)) * P;
  return h;
}

__global__ void rainbow_naive_kernel(const int* __restrict__ input,
                                     unsigned int* __restrict__ output, int N,
                                     int R) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  unsigned int v = static_cast<unsigned int>(input[i]);
  for (int r = 0; r < R; ++r) v = fnv1a_naive(v);
  output[i] = v;
}

// input, output are device pointers
extern "C" void solve(const int* input, unsigned int* output, int N, int R) {
  if (N <= 0) return;
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  rainbow_naive_kernel<<<blocks, threads>>>(input, output, N, R);
  cudaDeviceSynchronize();
}
