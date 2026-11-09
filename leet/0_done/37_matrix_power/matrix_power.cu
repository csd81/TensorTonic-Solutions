// Matrix power A^P (challenge 37).
//
// Square-and-multiply over a dense FP32 square matrix.  N <= 1024, P <= 20;
// benchmark N = 512.  The number of matrix products is fixed by the exponent
// (P=20 needs 5: four squarings and one multiply), so the only levers are the
// GEMM speed and the data movement between products.
//
// The starter measures 4.55 ms at N=512, P=20.  It uses a 32x32 shared tile
// with one output per thread (two shared loads per multiply-add, ~0.3
// TFLOP/s) and performs a separate cudaMemcpy after every product, plus a
// cudaMalloc/cudaFree pair per call.
//
// This version:
//
//  1. Uses the register-blocked GEMM family from the other matmul
//     challenges, sized for N = 512: 64x128 block tile, 16x16 threads,
//     4x8 per thread with interleaved columns (shared loads per multiply-add
//     0.375 instead of 2).
//  2. Runs left-to-right binary exponentiation with the read-only input
//     matrix as the base, so only two scratch buffers are needed and the
//     result ping-pongs between them by pointer swap -- exactly two 1 MB
//     device copies per call (input -> register, result -> output) instead of
//     the starter's seven.
//  3. Caches the two scratch buffers across calls, so a repeated benchmark
//     pays no cudaMalloc/cudaFree.
//
// Measured at N=512, P=20: 2.15 ms versus 4.55 ms (2.1x).  Per-exponent
// numbers and the GEMM shape sweep are in the README.

#include <cuda_runtime.h>

namespace {

constexpr int BM = 64;   // block tile rows
constexpr int BN = 128;  // block tile cols
constexpr int BK = 16;   // contraction-tile depth
constexpr int TM = 4;    // per-thread rows
constexpr int TN = 8;    // per-thread cols
constexpr int TX = 16;   // threads along cols (TX * TN == BN)
constexpr int TY = 16;   // threads along rows (TY * TM == BM)
constexpr int TPB = TX * TY;

__global__ void __launch_bounds__(TPB)
gemm_kernel(const float* __restrict__ A, const float* __restrict__ B,
            float* __restrict__ C, int N) {
  __shared__ float As[BM][BK + 1];
  __shared__ float Bs[BK][BN + 1];

  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;
  const int tid = threadIdx.y * TX + threadIdx.x;
  const int ty = threadIdx.y;
  const int tx = threadIdx.x;

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

  for (int t = 0; t < N; t += BK) {
    for (int idx = tid; idx < BM * BK; idx += TPB) {
      const int row = idx / BK, k = idx % BK;
      const int gm = m0 + row, gk = t + k;
      As[row][k] = (gm < N && gk < N) ? A[(size_t)gm * N + gk] : 0.0f;
    }
    for (int idx = tid; idx < BK * BN; idx += TPB) {
      const int kb = idx / BN, col = idx % BN;
      const int gk = t + kb, gn = n0 + col;
      Bs[kb][col] = (gk < N && gn < N) ? B[(size_t)gk * N + gn] : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; ++k) {
      float a_frag[TM];
      float b_frag[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) a_frag[i] = As[ty * TM + i][k];
#pragma unroll
      for (int j = 0; j < TN; ++j) b_frag[j] = Bs[k][tx + j * TX];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = fmaf(a_frag[i], b_frag[j], acc[i][j]);
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int gm = m0 + ty * TM + i;
    if (gm >= N) continue;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int gn = n0 + tx + j * TX;
      if (gn >= N) continue;
      C[(size_t)gm * N + gn] = acc[i][j];
    }
  }
}

// Two scratch buffers, allocated once and reused across solve() calls.
float* g_w0 = nullptr;
float* g_w1 = nullptr;
size_t g_cap = 0;

}  // namespace

// input and output are device pointers; input is never modified.
extern "C" void solve(const float* input, float* output, int N, int P) {
  if (N <= 0 || P <= 0) return;
  const size_t bytes = (size_t)N * N * sizeof(float);

  // P == 1 is a pure copy; the general loop below would do the same but this
  // avoids touching the scratch buffers.
  if (P == 1) {
    cudaMemcpyAsync(output, input, bytes, cudaMemcpyDeviceToDevice);
    return;
  }

  if ((size_t)N * N > g_cap) {
    if (g_w0) cudaFree(g_w0);
    if (g_w1) cudaFree(g_w1);
    cudaMalloc(&g_w0, bytes);
    cudaMalloc(&g_w1, bytes);
    g_cap = (size_t)N * N;
  }

  const dim3 block(TX, TY);
  const dim3 grid((N + BN - 1) / BN, (N + BM - 1) / BM);
  auto mm = [&](const float* A, const float* B, float* C) {
    gemm_kernel<<<grid, block>>>(A, B, C, N);
  };

  // Left-to-right binary exponentiation: R = A; then for each bit below the
  // most significant one, R = R^2 and, if the bit is set, R = R * A.  The
  // input stays the base and is only read, so two scratch buffers suffice.
  cudaMemcpyAsync(g_w0, input, bytes, cudaMemcpyDeviceToDevice);
  float* R = g_w0;
  float* T = g_w1;
  const int msb = 31 - __builtin_clz((unsigned)P);
  for (int bit = msb - 1; bit >= 0; --bit) {
    mm(R, R, T);
    float* tmp = R;
    R = T;
    T = tmp;
    if ((P >> bit) & 1) {
      mm(R, input, T);
      tmp = R;
      R = T;
      T = tmp;
    }
  }
  cudaMemcpyAsync(output, R, bytes, cudaMemcpyDeviceToDevice);
}
