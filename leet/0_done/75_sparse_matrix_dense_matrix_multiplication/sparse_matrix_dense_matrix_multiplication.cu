// Sparse matrix x dense matrix multiplication (challenge 75).
//
// C[M,K] = A[M,N] * B[N,K].  A is stored dense but is 60-70% zeros; nnz is
// given but not the positions.  All matrices are row-major float32.
// Benchmark: M=4096, N=2048, K=512 -> 4.29e9 multiply-adds (8.6 GFLOP
// dense-equivalent) on 32 MB of A, 4 MB of B and 8 MB of C.
//
// The starter measures 28.0 ms (0.307 TFLOP/s): a 32x32 shared tile with one
// output per thread, i.e. two shared-memory loads per multiply-add, so the
// shared-memory pipe is the bottleneck.
//
// This version is the register-blocked GEMM shape that won the GEMM
// challenges on this GPU: 128x128 block tile, 16x16 threads, 8x8 per thread
// (64 FP32 accumulators), BK=16, interleaved per-thread columns so shared
// reads are conflict-free.  Shared loads per multiply-add drop from 2 to
// 0.25.  It measures 6.82 ms = 1.26 TFLOP/s dense-equivalent (4.1x).
//
// On exploiting the sparsity
// --------------------------
// The nonzeros are random, which defeats every cheap way to skip work:
//
//   * Block-level skipping: a 128-row x 16-column A tile holds 2048 values;
//     all-zero has probability 0.65^2048 ~ 0.
//   * Warp-uniform precision: `ballot(a != 0)` over 32 lanes is nonzero with
//     probability ~1 - 0.65^32, so a whole warp never skips an FMA group.
//   * Per-element skipping: `if (a_frag[i] != 0)` compiles to predicated
//     FMAs, and the instructions issue whether the predicate is on or off.
//     Measured: 7.88 ms, 16% *slower* than the dense loop (the compare and
//     select per fragment row cost more than the issued FMAs save).
//   * A CSR build (counting pass + scatter over the 32 MB of A costs
//     ~1.5 ms) followed by row-per-warp SpMM does 1.5e9 FMAs instead of
//     4.3e9, but every formulation needs one shared-memory B load per FMA,
//     which is ~5.6 ms of shared bandwidth on this GPU before any other
//     cost -- more than the dense kernel spends in total.
//
// So the dense register-blocked kernel is the right answer here: the
// sparsity does not reduce the data that must be read, and it cannot be made
// to reduce instructions without divergence.
//
// Measured tile variants at this shape (same-process interleaved):
//   128x128 tile, 16x16 thr, 8x8  6.82 ms  (shipped)
//   256x128 tile, 16x32 thr, 8x8  7.72 ms
//   128x256 tile, 32x16 thr, 8x8  7.69 ms
//   128x128 + zero-skip           7.88 ms

#include <cuda_runtime.h>

namespace {

constexpr int BM = 128;  // block tile rows (M)
constexpr int BN = 128;  // block tile cols (K)
constexpr int BK = 16;   // contraction-tile depth
constexpr int TM = 8;    // per-thread rows
constexpr int TN = 8;    // per-thread cols
constexpr int TX = 16;   // threads along K
constexpr int TY = 16;   // threads along M
constexpr int TPB = TX * TY;

__global__ void __launch_bounds__(TPB)
spmm_kernel(const float* __restrict__ A, const float* __restrict__ B,
            float* __restrict__ C, int M, int N, int K) {
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
    // A tile (BM x BK) from A[M, N]: contiguous along N.
    for (int idx = tid; idx < BM * BK; idx += TPB) {
      const int row = idx / BK, k = idx % BK;
      const int gm = m0 + row, gk = t + k;
      As[row][k] = (gm < M && gk < N) ? A[(size_t)gm * N + gk] : 0.0f;
    }
    // B tile (BK x BN) from B[N, K]: contiguous along K.
    for (int idx = tid; idx < BK * BN; idx += TPB) {
      const int kb = idx / BN, col = idx % BN;
      const int gk = t + kb, gn = n0 + col;
      Bs[kb][col] = (gk < N && gn < K) ? B[(size_t)gk * K + gn] : 0.0f;
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
      if (gn >= K) continue;
      C[(size_t)gm * K + gn] = acc[i][j];
    }
  }
}

}  // namespace

// A, B, C are device pointers; nnz is provided but not needed by the dense
// traversal (the nonzeros are found by reading A, which is unavoidable).
extern "C" void solve(const float* A, const float* B, float* C, int M, int N,
                      int K, int nnz) {
  (void)nnz;
  if (M <= 0 || N <= 0 || K <= 0) return;
  dim3 block(TX, TY);
  dim3 grid((K + BN - 1) / BN, (M + BM - 1) / BM);
  spmm_kernel<<<grid, block>>>(A, B, C, M, N, K);
}
