#include <cuda_runtime.h>

// Naive reference implementation: one GPU thread per output element, direct
// global-memory loads and stores, no vectorisation, no shared memory, one
// straightforward kernel launch.
__global__ void vector_add_naive_kernel(const float* A, const float* B,
                                        float* C, int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    C[i] = A[i] + B[i];
  }
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
  if (A == nullptr || B == nullptr || C == nullptr || N <= 0) return;
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  vector_add_naive_kernel<<<blocks, threads>>>(A, B, C, N);
  cudaDeviceSynchronize();
}
