#include <cuda_runtime.h>

// Naive matrix multiplication: one thread per output element of C.
//
//   C = A * B, row-major: A is M x N, B is N x K, C is M x K.
//
// Each thread walks the whole N-length reduction with plain global-memory
// loads: no shared memory, no vector types, no register blocking.
// Consecutive threads own consecutive columns of C, so the B loads are
// coalesced and the A load is a warp-wide broadcast.

__global__ void matrix_multiplication_naive_kernel(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int M, int N, int K) {
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long total = (long long)M * K;
  if (idx >= total) return;

  const int row = (int)(idx / K);
  const int col = (int)(idx % K);

  const float* a = A + (long long)row * N;
  const float* b = B + col;

  float acc = 0.0f;
  for (int n = 0; n < N; ++n) {
    acc += a[n] * b[(long long)n * K];
  }
  C[idx] = acc;
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
  const long long total = (long long)M * K;
  const int threadsPerBlock = 256;
  const long long blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;

  matrix_multiplication_naive_kernel<<<(unsigned int)blocksPerGrid, threadsPerBlock>>>(
      A, B, C, M, N, K);
  cudaDeviceSynchronize();
}
