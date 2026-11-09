#include <cuda_runtime.h>

#include <cstdint>

// Weight dequantization: Y[i][j] = X[i][j] * S[i/T][j/T], where S is the
// per-tile scale matrix (ceil(M/T) x ceil(N/T)).
//
// At the benchmark (8192 x 8192) the kernel reads 268 MB and writes 268 MB —
// 537 MB — and S is a 64x64 table (16 KB) that stays in cache, so this is a
// plain streaming problem.  A float4 copy of the same traffic measures
// ~6.75 ms (80 GB/s); the starter's scalar 16x16-block kernel measures
// 7.35 ms (73 GB/s) because it issues one scalar load, one scalar store and
// two integer divisions per element.
//
// This version processes four adjacent columns per thread: one float4 load,
// one float4 store, and a single S lookup per group.  Because TILE_SIZE is a
// multiple of 4 in {16,32,64,128} and the groups are 4-aligned, all four
// cells always fall in the same tile, so one scale factor suffices.
//
// Any other shape (T not a multiple of 4, N % 4 != 0, or misaligned
// pointers) falls back to the scalar one-element-per-thread kernel.

namespace {

__global__ void dequant_vec_kernel(const float* __restrict__ X,
                                   const float* __restrict__ S,
                                   float* __restrict__ Y, int M, int N, int T,
                                   int S_cols) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= M) return;
  const int j = g * 4;
  if (j >= N) return;
  const size_t idx = (size_t)i * N + j;
  const int s_idx = (i / T) * S_cols + j / T;
  if (j + 3 < N) {
    const float4 v = __ldg(reinterpret_cast<const float4*>(X + idx));
    const float s = __ldg(&S[s_idx]);
    const float4 r = make_float4(v.x * s, v.y * s, v.z * s, v.w * s);
    *reinterpret_cast<float4*>(Y + idx) = r;
  } else {
    for (int k = 0; j + k < N; ++k)
      Y[idx + k] = X[idx + k] * S[(i / T) * S_cols + (j + k) / T];
  }
}

__global__ void dequant_scalar_kernel(const float* __restrict__ X,
                                      const float* __restrict__ S,
                                      float* __restrict__ Y, int M, int N,
                                      int T, int S_cols) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= M || j >= N) return;
  const size_t idx = (size_t)i * N + j;
  Y[idx] = X[idx] * S[(i / T) * S_cols + j / T];
}

}  // namespace

// X, S, Y are device pointers
extern "C" void solve(const float* X, const float* S, float* Y, int M, int N,
                      int TILE_SIZE) {
  if (M <= 0 || N <= 0 || TILE_SIZE <= 0) return;
  const int S_cols = (N + TILE_SIZE - 1) / TILE_SIZE;

  const bool aligned =
      (((reinterpret_cast<uintptr_t>(X) | reinterpret_cast<uintptr_t>(Y)) & 15u) ==
       0) &&
      ((N & 3) == 0) && ((TILE_SIZE & 3) == 0);
  if (aligned) {
    const int gx = (N + 3) / 4;
    dim3 block(64, 4);
    dim3 grid((gx + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    dequant_vec_kernel<<<grid, block>>>(X, S, Y, M, N, TILE_SIZE, S_cols);
  } else {
    dim3 block(32, 8);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    dequant_scalar_kernel<<<grid, block>>>(X, S, Y, M, N, TILE_SIZE, S_cols);
  }
  cudaDeviceSynchronize();
}
