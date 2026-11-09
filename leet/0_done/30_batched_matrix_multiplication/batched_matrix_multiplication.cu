// Batched matrix multiplication (challenge 30).
//
// C_b = A_b · B_b for b < BATCH; A [BATCH, M, K], B [BATCH, K, N],
// C [BATCH, M, N], all row-major FP32.  Constraints: BATCH <= 128,
// M, N, K <= 1024; performance is measured at M = N = K = 256 (the batch size
// is not stated by the challenge, so several are reported).
//
// The starter measures 0.135 ms at BATCH=1 and 14.5 ms at BATCH=128: a 16x16
// shared tile with one output per thread, i.e. two shared loads per
// multiply-add, ~0.25-0.30 TFLOP/s.
//
// This version uses the register-blocked GEMM family: 64x64 block tile,
// 16x16 threads, 4x4 per thread (shared loads per multiply-add 0.5) and one
// grid.z slice per batch.  At M=N=K=256 this leaves 16 blocks per batch, and
// it was the fastest of the shapes swept at every batch size -- the 128x128
// 8x8 tile, which wins at larger matrices, only gets 4 blocks per batch here
// and loses badly (0.32-0.38 TFLOP/s against 0.52-0.71).
//
// Measured (same process, interleaved; see README for the full table):
//   BATCH=1 0.064 ms (2.1x), 8 0.424 ms (2.4x), 32 1.534 ms (2.6x),
//   128 6.03 ms (2.4x).

#include <cuda_runtime.h>

namespace {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TM = 4;
constexpr int TN = 4;
constexpr int TX = 16;  // TX * TN == BN
constexpr int TY = 16;  // TY * TM == BM
constexpr int TPB = TX * TY;

__global__ void __launch_bounds__(TPB)
bmm_kernel(const float* __restrict__ A, const float* __restrict__ B,
           float* __restrict__ C, int M, int N, int K) {
  __shared__ float As[BM][BK + 1];
  __shared__ float Bs[BK][BN + 1];

  const int b = blockIdx.z;
  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;
  const float* Ab = A + (size_t)b * M * K;
  const float* Bb = B + (size_t)b * K * N;
  float* Cb = C + (size_t)b * M * N;
  const int tid = threadIdx.y * TX + threadIdx.x;
  const int ty = threadIdx.y;
  const int tx = threadIdx.x;

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

  for (int t = 0; t < K; t += BK) {
    for (int idx = tid; idx < BM * BK; idx += TPB) {
      const int row = idx / BK, k = idx % BK;
      const int gm = m0 + row, gk = t + k;
      As[row][k] = (gm < M && gk < K) ? Ab[(size_t)gm * K + gk] : 0.0f;
    }
    for (int idx = tid; idx < BK * BN; idx += TPB) {
      const int kb = idx / BN, col = idx % BN;
      const int gk = t + kb, gn = n0 + col;
      Bs[kb][col] = (gk < K && gn < N) ? Bb[(size_t)gk * N + gn] : 0.0f;
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
    if (gm >= M) continue;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int gn = n0 + tx + j * TX;
      if (gn >= N) continue;
      Cb[(size_t)gm * N + gn] = acc[i][j];
    }
  }
}

}  // namespace

// A, B, C are device pointers.
extern "C" void solve(const float* A, const float* B, float* C, int BATCH,
                      int M, int N, int K) {
  if (BATCH <= 0 || M <= 0 || N <= 0 || K <= 0) return;
  dim3 block(TX, TY);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, BATCH);
  bmm_kernel<<<grid, block>>>(A, B, C, M, N, K);
}
