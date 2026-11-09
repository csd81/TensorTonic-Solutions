// Multi-adapter LoRA batched linear (challenge 133).
//
// y_t = x_t W + scale * (x_t A_{a_t}) B_{a_t} for a_t >= 0; just x_t W for
// a_t = -1.  x [T,K], W [K,N], A [L,K,R] (row-major: A_a[k,r]), B [L,R,N]
// (row-major: B_a[r,n]), y [T,N].  Benchmark: T=1024, K=4096, N=4096, R=16,
// L=8.  Everything is float32.
//
// Where the starter's time goes at the benchmark shape (measured separately):
//
//   * base GEMM   299 ms -- one thread per output element, K-loop reading W
//                  as a col-strided row; 0.116 TFLOP/s.
//   * LoRA path  3400 ms -- for *every output element* it recomputes
//                 x_t . A_a (R*K = 65,536 multiply-adds) from scratch:
//                 O(T*N*R*K) = 2.7e11 operations.  That is 91% of the total
//                 3.74 s.
//
// The fix is to not recompute the low-rank path per output element:
//
//   1. Base product with the same register-blocked GEMM shape that won the
//      GEMM challenge here: 128x128 block tile, 16x16 threads, 8x8 per
//      thread, BK=16, interleaved per-thread columns (conflict-free shared
//      reads), ~1.0 TFLOP/s.
//   2. One block per token computes the R-vector temp = x_t A_a in shared
//      memory, then applies temp B_a to the token's N outputs.  The K-loop
//      for temp runs once per token instead of once per output element, so
//      the LoRA path drops from O(T*N*R*K) to O(T*K*R + T*R*N).
//
// Measured (see README): ~34 ms total, ~110x faster than the starter.

#include <cuda_runtime.h>

namespace {

// ---- Base GEMM: y = x @ W, all row-major float32 ----

constexpr int BM = 128;  // block tile rows (T)
constexpr int BN = 128;  // block tile cols (N)
constexpr int BK = 16;   // k-tile depth
constexpr int TM = 8;    // per-thread rows
constexpr int TN = 8;    // per-thread cols
constexpr int TX = 16;   // threads along N
constexpr int TY = 16;   // threads along T
constexpr int TPB = TX * TY;

__global__ void __launch_bounds__(TPB)
base_gemm_kernel(const float* __restrict__ x, const float* __restrict__ W,
                 float* __restrict__ y, int T, int K, int N) {
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
    // A tile (BM x BK) from x[T, K]: contiguous in K.
    for (int idx = tid; idx < BM * BK; idx += TPB) {
      const int row = idx / BK, k = idx % BK;
      const int gm = m0 + row, gk = t + k;
      As[row][k] = (gm < T && gk < K) ? x[(size_t)gm * K + gk] : 0.0f;
    }
    // B tile (BK x BN) from W[K, N]: contiguous in N.
    for (int idx = tid; idx < BK * BN; idx += TPB) {
      const int kb = idx / BN, col = idx % BN;
      const int gk = t + kb, gn = n0 + col;
      Bs[kb][col] = (gk < K && gn < N) ? W[(size_t)gk * N + gn] : 0.0f;
    }
    __syncthreads();

    // 8x8 micro-kernel, interleaved columns (tx + j*TX): consecutive lanes
    // read consecutive shared elements.
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
    if (gm >= T) continue;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int gn = n0 + tx + j * TX;
      if (gn >= N) continue;
      y[(size_t)gm * N + gn] = acc[i][j];
    }
  }
}

// ---- LoRA path: one block per token ----

constexpr int LORA_THREADS = 128;
constexpr int MAX_R = LORA_THREADS;  // challenge caps R at 64

__global__ void lora_kernel(const float* __restrict__ x,
                            const float* __restrict__ A,
                            const float* __restrict__ B,
                            const int* __restrict__ adapter_ids,
                            float* __restrict__ y, int T, int K, int N, int R,
                            float scale) {
  const int t = blockIdx.x;
  const int a = adapter_ids[t];
  if (a < 0) return;

  __shared__ float temp[MAX_R];

  const float* x_t = x + (size_t)t * K;
  const float* A_a = A + (size_t)a * K * R;
  const float* B_a = B + (size_t)a * R * N;

  // temp[r] = sum_k x_t[k] * A_a[k, r].  Lanes are split over both r and
  // chunks of K, then the per-chunk partials are reduced in shared memory,
  // so all LORA_THREADS lanes contribute when R < LORA_THREADS.
  if (R <= LORA_THREADS) {
    const int lanes = LORA_THREADS / R;  // chunks of K (>= 1)
    const int r = threadIdx.x % R;
    const int chunk = threadIdx.x / R;
    float s = 0.0f;
    if (chunk < lanes) {
      for (int k = chunk; k < K; k += lanes) s += x_t[k] * A_a[(size_t)k * R + r];
    }
    __shared__ float partial[LORA_THREADS];
    partial[threadIdx.x] = s;
    __syncthreads();
    if (threadIdx.x < R) {
      float tot = 0.0f;
      for (int c = 0; c < lanes; ++c) tot += partial[c * R + threadIdx.x];
      temp[threadIdx.x] = tot;
    }
  } else {
    // R > LORA_THREADS cannot happen under the challenge constraints
    // (R <= 64) and solve() refuses R > MAX_R; keep a serial fallback.
    for (int r = threadIdx.x; r < R; r += LORA_THREADS) {
      float s = 0.0f;
      for (int k = 0; k < K; ++k) s += x_t[k] * A_a[(size_t)k * R + r];
      temp[r] = s;
    }
  }
  __syncthreads();

  // y[t, n] += scale * sum_r temp[r] * B_a[r, n].
  for (int n = threadIdx.x; n < N; n += LORA_THREADS) {
    float s = 0.0f;
    for (int r = 0; r < R; ++r) s += temp[r] * B_a[(size_t)r * N + n];
    y[(size_t)t * N + n] += scale * s;
  }
}

}  // namespace

// x, W, A, B, adapter_ids, y are device pointers.
extern "C" void solve(const float* x, const float* W, const float* A,
                      const float* B, const int* adapter_ids, float* y, int T,
                      int K, int N, int R, int L, float scale) {
  if (T <= 0 || K <= 0 || N <= 0 || R <= 0) return;
  if (R > MAX_R) return;  // outside the challenge constraints (R <= 64)

  dim3 block(TX, TY);
  dim3 grid((N + BN - 1) / BN, (T + BM - 1) / BM);
  base_gemm_kernel<<<grid, block>>>(x, W, y, T, K, N);

  lora_kernel<<<T, LORA_THREADS>>>(x, A, B, adapter_ids, y, T, K, N, R, scale);
}
