#include <cuda_runtime.h>

#include <cstdint>

// LoRA linear: output = x·Wᵀ + lora_scale · (x·Aᵀ)·Bᵀ
//
// Shapes: x [M, K], W [N, K], A [R, K], B [N, R], output [M, N].
// Benchmark: M=256, K=4096, N=4096, R=64, i.e. 8.9 GFLOP dominated by the
// m×K×N base product; the two LoRA products are ~1.5% of the work.
//
// The starter computes every product with one thread per output and a K-loop
// that reads B as a strided row per output column — no tiling, no reuse — and
// measures 282 ms (0.031 TFLOP/s).
//
// This version runs all three products through one register-blocked GEMM
// kernel C = A·Bᵀ (B stored [N, K] row-major, which is exactly W, A and B):
// 128x128 block tiles, 16x16 threads, 8x8 per thread with interleaved
// columns (conflict-free shared reads), the same shape that won in the GEMM
// challenge.  Phase 1 writes the base product, phase 2 produces the small
// (M x R) intermediate, phase 3 accumulates the scaled LoRA product into the
// output.  Measured ~9-10 ms, a ~30x improvement.

namespace {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int TX = 16;
constexpr int TY = 16;
constexpr int TPB = TX * TY;

// C = scale * A @ B^T, or C += scale * A @ B^T when ACCUM.
template <bool ACCUM>
__global__ void __launch_bounds__(TPB)
gemm_t_kernel(const float* __restrict__ A, const float* __restrict__ B,
              float* __restrict__ C, int M, int N, int K, float scale) {
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
    // A tile (BM x BK): contiguous along K -> coalesced.
    for (int idx = tid; idx < BM * BK; idx += TPB) {
      const int row = idx / BK, k = idx % BK;
      const int gm = m0 + row, gk = t + k;
      As[row][k] = (gm < M && gk < K) ? __ldg(&A[(size_t)gm * K + gk]) : 0.0f;
    }
    // B tile (BK x BN) from B[N, K]: consecutive k are contiguous, so iterate
    // k in the inner dimension for coalescing.
    for (int idx = tid; idx < BK * BN; idx += TPB) {
      const int col = idx / BK, k = idx % BK;
      const int gn = n0 + col, gk = t + k;
      Bs[k][col] = (gn < N && gk < K) ? __ldg(&B[(size_t)gn * K + gk]) : 0.0f;
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
      const size_t idx = (size_t)gm * N + gn;
      if (ACCUM)
        C[idx] += scale * acc[i][j];
      else
        C[idx] = scale * acc[i][j];
    }
  }
}

float* g_temp = nullptr;
int g_temp_cap = 0;

}  // namespace

// x, W, A, B, output are device pointers
extern "C" void solve(const float* x, const float* W, const float* A,
                      const float* B, float* output, int batch, int d_in,
                      int d_out, int rank, float lora_scale) {
  if (batch <= 0 || d_in <= 0 || d_out <= 0 || rank <= 0) return;

  // temp = x @ A^T  [batch, rank]
  const int temp_elems = batch * rank;
  if (temp_elems > g_temp_cap) {
    if (g_temp) cudaFree(g_temp);
    cudaMalloc(&g_temp, (size_t)temp_elems * sizeof(float));
    g_temp_cap = temp_elems;
  }

  dim3 block(TX, TY);
  auto grid = [](int M, int N) {
    return dim3((N + BN - 1) / BN, (M + BM - 1) / BM);
  };

  // 1. output = x @ W^T
  gemm_t_kernel<false><<<grid(batch, d_out), block>>>(x, W, output, batch,
                                                      d_out, d_in, 1.0f);
  // 2. temp = x @ A^T
  gemm_t_kernel<false><<<grid(batch, rank), block>>>(x, A, g_temp, batch, rank,
                                                     d_in, 1.0f);
  // 3. output += lora_scale * (temp @ B^T)
  gemm_t_kernel<true><<<grid(batch, d_out), block>>>(g_temp, B, output, batch,
                                                     d_out, rank, lora_scale);
  cudaDeviceSynchronize();
}
