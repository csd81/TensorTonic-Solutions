// Grouped GEMM (challenge 99).
//
// A [M_total, K] is split into G contiguous row groups by group_offsets; each
// group g multiplies its rows by its own weight B[g] [K, N]:
//   C[off[g]:off[g+1]] = A[off[g]:off[g+1]] . B[g]
// Benchmark: G=8, M_total=8192, K=1024, N=2048 (34.4 GFLOP).
//
// The starter measures 120.3 ms (0.286 TFLOP/s) on the benchmark shape, but
// it is only correct when every group boundary is a multiple of its 16-row
// block: it loads the B tile into shared memory from *each thread's* own
// group's weight matrix, so a block whose rows span a group boundary races
// and every row ends up multiplied by whichever group's B values were
// written into the shared tile (verified: with groups {0,2,4} the rows of the
// second group get the first group's product).  The benchmark's equal
// 1024-row groups happen to be block-aligned, which is why it passes there.
//
// This version eliminates the ambiguity instead of racing on it:
//
//  1. A 1-thread planning kernel enumerates (group, row-block) pairs across
//     all groups into small lookup arrays (group, first row, row count), so
//     each GEMM block belongs to exactly one group and no binary search or
//     group check is needed in the inner loop.
//  2. Blocks launch with the upper-bound grid (ceil(M/BM) + G row-blocks) and
//     return early past the actual count, so no host-side sync is needed.
//  3. The GEMM itself is the register-blocked shape from the other matmul
//     challenges: 128x128 tile, 16x16 threads, 8x8 per thread, BK=16,
//     interleaved columns; the A/C pointers are offset by the block's first
//     row and the B pointer by the group's weight matrix.
//
// Measured: see README (~35 ms, ~1 TFLOP/s, 3.4x over the starter).

#include <cuda_runtime.h>

namespace {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int TX = 16;  // TX * TN == BN
constexpr int TY = 16;  // TY * TM == BM
constexpr int TPB = TX * TY;

// One thread enumerates every (group, row-block) pair.  G <= 64 and the
// number of row-blocks is <= M_total/BM + G <= 200, so this is trivial.
__global__ void plan_kernel(const int* __restrict__ group_offsets, int G,
                            int* __restrict__ blk_g, int* __restrict__ blk_row,
                            int* __restrict__ blk_rows, int* __restrict__ nblk) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int rb = 0;
  for (int g = 0; g < G; ++g) {
    const int s = group_offsets[g];
    const int e = group_offsets[g + 1];
    const int rows = e - s;
    for (int r = 0; r < rows; r += BM) {
      blk_g[rb] = g;
      blk_row[rb] = s + r;
      blk_rows[rb] = (rows - r < BM) ? (rows - r) : BM;
      ++rb;
    }
  }
  *nblk = rb;
}

__global__ void __launch_bounds__(TPB)
grouped_gemm_kernel(const float* __restrict__ A, const float* __restrict__ B,
                    float* __restrict__ C, const int* __restrict__ blk_g,
                    const int* __restrict__ blk_row,
                    const int* __restrict__ blk_rows,
                    const int* __restrict__ nblk, int N, int K) {
  const int rb = blockIdx.z;
  if (rb >= *nblk) return;

  const int g = blk_g[rb];
  const int row0 = blk_row[rb];
  const int rows = blk_rows[rb];  // valid rows in this block's group
  const float* Ab = A + (size_t)row0 * K;
  const float* Bb = B + (size_t)g * K * N;
  float* Cb = C + (size_t)row0 * N;

  __shared__ float As[BM][BK + 1];
  __shared__ float Bs[BK][BN + 1];

  const int m0 = 0;  // rows are relative to row0 within the group
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
    for (int idx = tid; idx < BM * BK; idx += TPB) {
      const int row = idx / BK, k = idx % BK;
      const int gm = m0 + row, gk = t + k;
      // Rows past the group's end belong to the next group: zero them.
      As[row][k] = (row < rows && gk < K) ? Ab[(size_t)gm * K + gk] : 0.0f;
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
    const int gm = ty * TM + i;
    if (gm >= rows) continue;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int gn = n0 + tx + j * TX;
      if (gn >= N) continue;
      Cb[(size_t)gm * N + gn] = acc[i][j];
    }
  }
}

// Cached planning arrays; sized for G <= 64 and the row-block upper bound.
int* g_blk_g = nullptr;
int* g_blk_row = nullptr;
int* g_blk_rows = nullptr;
int* g_nblk = nullptr;
int g_plan_cap = 0;

}  // namespace

// A, B, group_offsets, C are device pointers.
extern "C" void solve(const float* A, const float* B, const int* group_offsets,
                      float* C, int G, int M_total, int K, int N) {
  if (G <= 0 || M_total <= 0 || K <= 0 || N <= 0) return;
  const int max_blocks = (M_total + BM - 1) / BM + G;
  if (max_blocks > g_plan_cap) {
    if (g_blk_g) {
      cudaFree(g_blk_g);
      cudaFree(g_blk_row);
      cudaFree(g_blk_rows);
      cudaFree(g_nblk);
    }
    cudaMalloc(&g_blk_g, (size_t)max_blocks * sizeof(int));
    cudaMalloc(&g_blk_row, (size_t)max_blocks * sizeof(int));
    cudaMalloc(&g_blk_rows, (size_t)max_blocks * sizeof(int));
    cudaMalloc(&g_nblk, sizeof(int));
    g_plan_cap = max_blocks;
  }

  plan_kernel<<<1, 1>>>(group_offsets, G, g_blk_g, g_blk_row, g_blk_rows, g_nblk);

  dim3 block(TX, TY);
  dim3 grid((N + BN - 1) / BN, 1, max_blocks);
  grouped_gemm_kernel<<<grid, block>>>(A, B, C, g_blk_g, g_blk_row, g_blk_rows,
                                       g_nblk, N, K);
}
