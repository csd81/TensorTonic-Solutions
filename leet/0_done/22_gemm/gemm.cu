#include <cuda_fp16.h>
#include <cuda_runtime.h>

// FP16 GEMM with FP32 accumulation: C = alpha*(A@B) + beta*C, all matrices
// row-major half, alpha/beta float.
//
// At the benchmark (1024 x 1024 x 1024) this is 2.15 GFLOP on 6 MB of
// matrices, so it is compute-bound.  WMMA is not available on this GPU
// (compute capability 6.1 has no tensor cores), so the win comes from classic
// register blocking and tiling:
//
//   * 128x128 block tile, 16x16 threads, 8x8 per thread (64 FP32
//     accumulators).  One A-fragment and one B-fragment load then feed 64
//     FMAs, cutting shared traffic per FMA by 8x versus the starter's
//     two-loads-per-FMA structure.
//   * Interleaved thread tiles (thread tx owns columns tx, tx+16, tx+32, ...)
//     so the B-fragment reads from shared memory are consecutive across lanes
//     (conflict-free) and the global writes stay coalesced.  With contiguous
//     per-thread columns the B reads are 4-way bank-conflicted and the same
//     kernel measures 0.871 TFLOP/s vs 1.036 here.
//
// Larger block tiles were swept (256x128 with 512 threads, 192x128, 128x192,
// BK 8/32, 4x4 and 16x4 thread tiles): all measured slower than this shape.
//
// The tile loaders use scalar half loads (conversion to float on the way into
// shared memory) so odd K row strides, which would break float4 alignment,
// need no separate path.  Edge tiles are bounds-checked and zero-filled; the
// epilogue applies alpha/beta and converts back to half.

namespace {

constexpr int BM = 128;  // block tile rows (M)
constexpr int BN = 128;  // block tile cols (N)
constexpr int BK = 16;   // k-tile depth
constexpr int TM = 8;    // per-thread rows
constexpr int TN = 8;    // per-thread cols
constexpr int TX = 16;   // threads along N (tx * TN == BN)
constexpr int TY = 16;   // threads along M (ty * TM == BM)
constexpr int TPB = TX * TY;

__global__ void __launch_bounds__(TPB)
gemm_fp16_kernel(const half* __restrict__ A, const half* __restrict__ B,
                 half* __restrict__ C, int M, int N, int K, float alpha,
                 float beta) {
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

  for (int t = 0; t < K; t += BK) {
    // A tile (BM x BK) and B tile (BK x BN); consecutive lanes read
    // consecutive elements of a row.
    for (int idx = tid; idx < BM * BK; idx += TPB) {
      const int row = idx / BK, k = idx % BK;
      const int gm = m0 + row, gk = t + k;
      As[row][k] = (gm < M && gk < K) ? __half2float(A[(size_t)gm * K + gk])
                                      : 0.0f;
    }
    for (int idx = tid; idx < BK * BN; idx += TPB) {
      const int kb = idx / BN, col = idx % BN;
      const int gk = t + kb, gn = n0 + col;
      Bs[kb][col] = (gk < K && gn < N) ? __half2float(B[(size_t)gk * N + gn])
                                       : 0.0f;
    }
    __syncthreads();

    // 8x8 micro-kernel with interleaved columns (tx + j*TX).
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
      float res = alpha * acc[i][j];
      if (beta != 0.0f) res += beta * __half2float(C[(size_t)gm * N + gn]);
      C[(size_t)gm * N + gn] = __float2half(res);
    }
  }
}

}  // namespace

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N,
                      int K, float alpha, float beta) {
  if (M <= 0 || N <= 0 || K <= 0) return;
  dim3 block(TX, TY);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  gemm_fp16_kernel<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
  cudaDeviceSynchronize();
}
