// FP16 batched matrix multiplication (challenge 57).
//
// C_b = A_b · B_b for BATCH batches; A [BATCH, M, K], B [BATCH, K, N],
// C [BATCH, M, N], all row-major FP16, FP32 accumulation, half output.
// BATCH <= 128, M, N, K <= 1024; the benchmark shape is M = N = K = 256 and
// the batch size is not stated, so B = 1, 8, 32, 128 are all reported.
//
// The starter measures 0.161 ms (B=1) to 17.0 ms (B=128), i.e.
// 0.21-0.25 TFLOP/s: 32x32 half shared tiles with one output per thread, and
// each multiply-add converts both shared halves to float individually.
//
// This version follows the FP16 GEMM that won challenge 22:
//
//  * The shared tiles hold *float*, converted once when the tile is loaded
//    (each element is used BK-1 times more in the micro-kernel than in the
//    loader, so converting once is ~30x fewer conversions than the starter).
//  * 64x64 block tile, 16x16 threads, 4x4 per thread (0.5 shared loads per
//    multiply-add), one grid.z slice per batch.
//  * The epilogue converts the FP32 accumulators back to half.
//
// Measured: 0.066 / 0.420 / 1.541 / 6.062 ms for B = 1/8/32/128
// (0.51-0.71 TFLOP/s), 2.4-3.1x faster than the starter.  The 128x128/8x8
// tile ties or wins slightly at B >= 32 (0.71-0.73 TFLOP/s) but is 60% slower
// at B=1 because M=N=K=256 leaves it only four blocks per batch, so the
// 64x64/4x4 shape is shipped as the robust choice.

#include <cuda_fp16.h>
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
bmm_fp16_kernel(const half* __restrict__ A, const half* __restrict__ B,
                half* __restrict__ C, int M, int N, int K) {
  __shared__ float As[BM][BK + 1];
  __shared__ float Bs[BK][BN + 1];

  const int b = blockIdx.z;
  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;
  const half* Ab = A + (size_t)b * M * K;
  const half* Bb = B + (size_t)b * K * N;
  half* Cb = C + (size_t)b * M * N;
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
      As[row][k] =
          (gm < M && gk < K) ? __half2float(Ab[(size_t)gm * K + gk]) : 0.0f;
    }
    for (int idx = tid; idx < BK * BN; idx += TPB) {
      const int kb = idx / BN, col = idx % BN;
      const int gk = t + kb, gn = n0 + col;
      Bs[kb][col] =
          (gk < K && gn < N) ? __half2float(Bb[(size_t)gk * N + gn]) : 0.0f;
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
      Cb[(size_t)gm * N + gn] = __float2half(acc[i][j]);
    }
  }
}

}  // namespace

// A, B, C are device pointers.
extern "C" void solve(const half* A, const half* B, half* C, int BATCH, int M,
                      int N, int K) {
  if (BATCH <= 0 || M <= 0 || N <= 0 || K <= 0) return;
  dim3 block(TX, TY);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, BATCH);
  bmm_fp16_kernel<<<grid, block>>>(A, B, C, M, N, K);
}
