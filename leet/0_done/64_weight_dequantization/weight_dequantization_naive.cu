#include <cuda_runtime.h>

// Naive dequantization: one thread per element with 2-D blocks, a scalar
// load/store, and the two integer divisions for the tile index.  No vector
// types, no per-group scale reuse -- the plainest correct form (and the
// starter's shape).

__global__ void dequant_naive_kernel(const float* __restrict__ X,
                                     const float* __restrict__ S,
                                     float* __restrict__ Y, int M, int N, int T,
                                     int S_cols) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= M || j >= N) return;
  const size_t idx = (size_t)i * N + j;
  Y[idx] = X[idx] * S[(i / T) * S_cols + j / T];
}

// X, S, Y are device pointers
extern "C" void solve(const float* X, const float* S, float* Y, int M, int N,
                      int TILE_SIZE) {
  if (M <= 0 || N <= 0 || TILE_SIZE <= 0) return;
  const int S_cols = (N + TILE_SIZE - 1) / TILE_SIZE;
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  dequant_naive_kernel<<<grid, block>>>(X, S, Y, M, N, TILE_SIZE, S_cols);
  cudaDeviceSynchronize();
}
