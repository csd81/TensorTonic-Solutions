#include <cuda_runtime.h>

// Naive SpMV: one thread per row.  Each thread walks its entire row with plain
// scalar global-memory loads and a serial accumulation, then writes y[row].
// No vector types, no shared memory, no warp intrinsics -- the plainest
// correct form of the dense-storage SpMV.

__global__ void spmv_naive_kernel(const float* __restrict__ A,
                                  const float* __restrict__ x,
                                  float* __restrict__ y, int M, int N) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M) return;

  const float* row_A = A + (size_t)row * N;
  float sum = 0.0f;
  for (int j = 0; j < N; ++j) sum += row_A[j] * x[j];
  y[row] = sum;
}

// A, x, y are device pointers
extern "C" void solve(const float* A, const float* x, float* y, int M, int N,
                      int nnz) {
  (void)nnz;
  if (M <= 0 || N <= 0) return;
  const int threads = 256;
  const int blocks = (M + threads - 1) / threads;
  spmv_naive_kernel<<<blocks, threads>>>(A, x, y, M, N);
}
