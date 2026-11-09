#include <cuda_runtime.h>

// Naive element-wise N x N matrix addition: one GPU thread per output
// element, direct global-memory loads and stores, no vectorisation, no shared
// memory, one straightforward kernel launch.
__global__ void matrix_addition_naive_kernel(const float* A, const float* B,
                                             float* C, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    C[i] = A[i] + B[i];
  }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU).
extern "C" void solve(const float* A, const float* B, float* C, int N) {
  if (A == nullptr || B == nullptr || C == nullptr || N <= 0) return;
  const int n = N * N;
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  matrix_addition_naive_kernel<<<blocks, threads>>>(A, B, C, n);
  cudaDeviceSynchronize();
}
