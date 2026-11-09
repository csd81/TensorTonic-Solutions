#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Naive tiled GEMM: a 32x32 block tile with one thread per output element.
// The shared-memory tiles reduce global traffic, but every FMA still reads
// both operands from shared memory (two loads per FMA), so the kernel is
// shared-memory-bandwidth-bound -- the plainest correct form, and the
// starter's shape.

#define TILE_SIZE 32

__global__ void gemm_naive_kernel(const half* __restrict__ A,
                                  const half* __restrict__ B,
                                  half* __restrict__ C, int M, int N, int K,
                                  float alpha, float beta) {
  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE];

  const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  float acc = 0.0f;

  const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
  for (int t = 0; t < num_tiles; ++t) {
    const int a_col = t * TILE_SIZE + threadIdx.x;
    const int b_row = t * TILE_SIZE + threadIdx.y;
    As[threadIdx.y][threadIdx.x] =
        (row < M && a_col < K) ? __half2float(A[(size_t)row * K + a_col]) : 0.0f;
    Bs[threadIdx.y][threadIdx.x] =
        (b_row < K && col < N) ? __half2float(B[(size_t)b_row * N + col]) : 0.0f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k)
      acc = fmaf(As[threadIdx.y][k], Bs[k][threadIdx.x], acc);
    __syncthreads();
  }

  if (row < M && col < N) {
    float res = alpha * acc;
    if (beta != 0.0f) res += beta * __half2float(C[(size_t)row * N + col]);
    C[(size_t)row * N + col] = __float2half(res);
  }
}

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N,
                      int K, float alpha, float beta) {
  if (M <= 0 || N <= 0 || K <= 0) return;
  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
  gemm_naive_kernel<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
  cudaDeviceSynchronize();
}
