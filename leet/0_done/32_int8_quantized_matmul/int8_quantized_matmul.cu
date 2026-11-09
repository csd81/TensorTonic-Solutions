// INT8 quantized matrix multiplication (challenge 32).
//
// C = clamp(rint( (sum_k (A-za)(B-zb) * sa*sb) / sc ) + zc, -128, 127)
// with int32 accumulation and half-to-even rounding (rintf) in the epilogue.
// Benchmark: M=8192, N=4096, K=2048 -> 6.87e10 int8 multiply-adds.
//
// The starter is already a dp4a GEMM (64x64 tile, 4x4 per thread, exact
// epilogue with precomputed row/column sums) and measures 267.5 ms
// (256 G int8-MAC/s).  It also has a real bug: compute_row_sums performs
// int32 reinterpret-cast loads at `A_row + k` where A_row = A + row*K, which
// is misaligned whenever K is not a multiple of 4 -- the kernel faults with
// "misaligned address" from the first such case and poisons the CUDA
// context, so every later launch fails.  The naive baseline keeps the
// starter's structure with scalar row-sum loads.
//
// The shipped kernel keeps the same algorithm (dp4a, exact zero-point
// epilogue) but fixes the two things that hold the starter at 6% of the
// dp4a peak (~1.1 T dp4a/s measured on this GPU):
//
//   * A 256x128 block tile with an 8x8 register tile (512 threads) instead of
//     64x64 with 4x4: shared loads per dp4a drop from 0.5 to 0.25.
//   * BK=32 and a 4-byte pad on the shared pitch (As[256][36],
//     Bs[128][36]), keeping the int32 fragment loads aligned.
//
// Measured (same process, interleaved, full solve):
//   64x64/4x4/BK16 (starter shape)   195-268 ms
//   128x128/8x4/BK32                  82.2 ms
//   128x128/8x8/BK32 (256 thr)        70.5 ms
//   256x128/8x8/BK32 (shipped)        68.0 ms  = 1011 G int8-MAC/s (3.94x)
//   128x256/8x8/BK32                  70.4 ms
//   BK=64 variants                    78-88 ms
//
// dp4a peak on this Quadro P2000: 1.10 T dp4a/s = 4.4 T int8-MAC/s, so the
// kernel runs at ~23% of the instruction peak; the remaining gap is shared
// memory and tile-load traffic.

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

// row_sums[i] = sum_k A[i,k] (scalar loads: A + row*K is not 4-byte aligned
// for odd row strides, so the starter's int32 loads are invalid).
__global__ void row_sums_kernel(const int8_t* __restrict__ A,
                                int32_t* __restrict__ rs, int M, int K) {
  const int row = blockIdx.x;
  const int8_t* Ar = A + (size_t)row * K;
  int32_t s = 0;
  for (int k = threadIdx.x; k < K; k += blockDim.x) s += (int32_t)Ar[k];
  for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
  __shared__ int32_t sh[32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) sh[warp] = s;
  __syncthreads();
  if (warp == 0) {
    s = (lane < (blockDim.x >> 5)) ? sh[lane] : 0;
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    if (lane == 0) rs[row] = s;
  }
}

// col_sums[j] = sum_k B[k,j]
__global__ void col_sums_kernel(const int8_t* __restrict__ B,
                                int32_t* __restrict__ cs, int K, int N) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= N) return;
  int32_t s = 0;
  for (int k = 0; k < K; ++k) s += (int32_t)B[(size_t)k * N + col];
  cs[col] = s;
}

constexpr int BM = 256;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int PAD = 4;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int TX = 16;  // TX * TN == BN
constexpr int TY = 32;  // TY * TM == BM
constexpr int TPB = TX * TY;

__global__ void __launch_bounds__(TPB)
qgemm_kernel(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
             int8_t* __restrict__ C, const int32_t* __restrict__ rs,
             const int32_t* __restrict__ cs, int M, int N, int K, float combined,
             int zA, int zB, int zC) {
  __shared__ int8_t As[BM][BK + PAD];
  __shared__ int8_t Bs[BN][BK + PAD];

  const int tid = threadIdx.y * TX + threadIdx.x;
  const int by = blockIdx.y, bx = blockIdx.x;
  const int ty = threadIdx.y * TM, tx = threadIdx.x * TN;

  int32_t acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0;

  for (int kb0 = 0; kb0 < K; kb0 += BK) {
    for (int i = tid; i < BM * BK; i += TPB) {
      const int r = i / BK, c = i % BK;
      const int gr = by * BM + r, gc = kb0 + c;
      As[r][c] = (gr < M && gc < K) ? A[(size_t)gr * K + gc] : 0;
    }
    // B is transposed into shared so four consecutive k values of one column
    // are one aligned int32 for dp4a.
    for (int i = tid; i < BK * BN; i += TPB) {
      const int r = i / BN, c = i % BN;
      const int gr = kb0 + r, gc = bx * BN + c;
      Bs[c][r] = (gr < K && gc < N) ? B[(size_t)gr * N + gc] : 0;
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; k += 4) {
      int32_t af[TM], bf[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i)
        af[i] = *reinterpret_cast<const int32_t*>(&As[ty + i][k]);
#pragma unroll
      for (int j = 0; j < TN; ++j)
        bf[j] = *reinterpret_cast<const int32_t*>(&Bs[tx + j][k]);
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = __dp4a(af[i], bf[j], acc[i][j]);
    }
    __syncthreads();
  }

  // Exact int32 result: sum((A-zA)(B-zB)) = dot - zB*row_sum - zA*col_sum
  // + K*zA*zB.
  const int32_t kz = K * zA * zB;
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int gr = by * BM + ty + i;
    if (gr >= M) continue;
    const int32_t rsum = rs[gr];
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int gc = bx * BN + tx + j;
      if (gc >= N) continue;
      const int32_t exact = acc[i][j] - zB * rsum - zA * cs[gc] + kz;
      int v = (int)rintf((float)exact * combined) + zC;
      v = max(-128, min(127, v));
      C[(size_t)gr * N + gc] = (int8_t)v;
    }
  }
}

// Reused workspace for the row/column sums; allocated once.
int32_t* g_rs = nullptr;
int32_t* g_cs = nullptr;
int g_rs_cap = 0;
int g_cs_cap = 0;

}  // namespace

extern "C" void solve(const int8_t* A, const int8_t* B, int8_t* C, int M, int N,
                      int K, float scale_A, float scale_B, float scale_C,
                      int zero_point_A, int zero_point_B, int zero_point_C) {
  if (M <= 0 || N <= 0 || K <= 0) return;
  if (M > g_rs_cap) {
    if (g_rs) cudaFree(g_rs);
    cudaMalloc(&g_rs, (size_t)M * sizeof(int32_t));
    g_rs_cap = M;
  }
  if (N > g_cs_cap) {
    if (g_cs) cudaFree(g_cs);
    cudaMalloc(&g_cs, (size_t)N * sizeof(int32_t));
    g_cs_cap = N;
  }

  row_sums_kernel<<<M, 256>>>(A, g_rs, M, K);
  col_sums_kernel<<<(N + 255) / 256, 256>>>(B, g_cs, K, N);

  dim3 block(TX, TY);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  const float combined = (scale_A * scale_B) / scale_C;
  qgemm_kernel<<<grid, block>>>(A, B, C, g_rs, g_cs, M, N, K, combined,
                                zero_point_A, zero_point_B, zero_point_C);
}
