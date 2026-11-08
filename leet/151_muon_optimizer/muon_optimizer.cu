#include <cuda_runtime.h>

#include <cmath>

// Muon optimizer update: V = momentum * Buf + G, then five quintic
// Newton-Schulz steps on X0 = V / (||V||_F + 1e-7):
//     A = X X^T,  B = b A + c A^2,  X <- a X + B X
// and finally output = X * sqrt(max(M, N)).
//
// When M > N the iteration runs on the transposed matrix so the Gram matrix
// is K x K with K = min(M, N) (the challenge allows this).
//
// All matrix products use a register-blocked GEMM: 16x16 threads, each
// computing a 4x4 block of a 64x64 output tile, with 64x16 and 16x64 shared
// tiles.  One templated kernel serves all three products; a residual
// epilogue computes C = alpha * (A B) + beta * R:
//     A  = X X^T      (transposed B operand, alpha = 1, beta = 0)
//     B  = c A^2 + b A (alpha = c, beta = b, residual = A)
//     X' = B X + a X   (alpha = 1, beta = a, residual = X)

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4
#define THREADS 256

__global__ void compute_v_kernel(const float* __restrict__ G,
                                 const float* __restrict__ Buf,
                                 float* __restrict__ X,
                                 float* __restrict__ norm_sq, int M, int N,
                                 float momentum, bool transpose) {
  __shared__ float sdata[THREADS];
  const int tid = threadIdx.x;
  const int total = M * N;

  float sum = 0.0f;
  for (int i = blockIdx.x * THREADS + tid; i < total; i += gridDim.x * THREADS) {
    const float v = fmaf(momentum, Buf[i], G[i]);
    sum = fmaf(v, v, sum);
    if (transpose) {
      const int r = i / N;
      const int c = i - r * N;
      X[c * M + r] = v;
    } else {
      X[i] = v;
    }
  }

  sdata[tid] = sum;
  __syncthreads();
  for (int s = THREADS / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(norm_sq, sdata[0]);
}

__global__ void normalize_kernel(float* __restrict__ X,
                                 const float* __restrict__ norm_sq, int total) {
  const float inv = 1.0f / (sqrtf(*norm_sq) + 1e-7f);
  for (int i = blockIdx.x * THREADS + threadIdx.x; i < total;
       i += gridDim.x * THREADS) {
    X[i] *= inv;
  }
}

// C = alpha * op(A) * op(B) + beta * R, with A (M x K), B (K x N) and
// optional transposition of B (stored as N x K).
template <bool TRANS_B, bool SYMMETRIC>
__global__ void gemm_kernel(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C, int M, int N, int K,
                            int lda, int ldb, float alpha, float beta,
                            const float* __restrict__ R, int ldr) {
  __shared__ float As[BM][BK + 1];
  __shared__ float Bs[BK][BN + 1];

  // For a symmetric result only the upper-triangular tiles are computed;
  // a mirror kernel fills the rest afterwards.
  if (SYMMETRIC && blockIdx.x < blockIdx.y) return;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * 16 + tx;
  const int row0 = blockIdx.y * BM;
  const int col0 = blockIdx.x * BN;

  const int a_row = row0 + ty * TM;
  const int b_col = col0 + tx * TN;

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

  for (int k0 = 0; k0 < K; k0 += BK) {
    // A tile: BM x BK.
#pragma unroll
    for (int i = 0; i < BM * BK / THREADS; ++i) {
      const int idx = tid + i * THREADS;
      const int r = idx / BK;
      const int c = idx - r * BK;
      As[r][c] = (row0 + r < M && k0 + c < K) ? A[(row0 + r) * lda + k0 + c]
                                              : 0.0f;
    }
    // B tile: BK x BN.  When B is transposed the fast index is k so the
    // global reads stay contiguous.
#pragma unroll
    for (int i = 0; i < BK * BN / THREADS; ++i) {
      const int idx = tid + i * THREADS;
      if (TRANS_B) {
        const int r = idx % BK;
        const int c = idx / BK;
        Bs[r][c] = (k0 + r < K && col0 + c < N)
                       ? B[(col0 + c) * ldb + k0 + r]
                       : 0.0f;
      } else {
        const int r = idx / BN;
        const int c = idx - r * BN;
        Bs[r][c] = (k0 + r < K && col0 + c < N)
                       ? B[(k0 + r) * ldb + col0 + c]
                       : 0.0f;
      }
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      float af[TM], bf[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) af[i] = As[ty * TM + i][kk];
#pragma unroll
      for (int j = 0; j < TN; ++j) bf[j] = Bs[kk][tx * TN + j];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(af[i], bf[j], acc[i][j]);
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int r = a_row + i;
    if (r < M) {
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const int c = b_col + j;
        if (c < N) {
          float v = alpha * acc[i][j];
          if (beta != 0.0f) v = fmaf(beta, R[r * ldr + c], v);
          C[r * N + c] = v;
        }
      }
    }
  }
}

// Fill the strictly-lower 32x32 tiles from their transposed counterparts.
__global__ void mirror_lower_kernel(float* __restrict__ C, int K) {
  __shared__ float tile[32][33];
  const int bi = blockIdx.y;
  const int bj = blockIdx.x;
  if (bi < bj) return;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int src_row = bj * 32 + ty;
  const int src_col = bi * 32 + tx;
  if (src_row < K && src_col < K) tile[ty][tx] = C[src_row * K + src_col];
  __syncthreads();
  const int dst_row = bi * 32 + ty;
  const int dst_col = bj * 32 + tx;
  if (dst_row >= K || dst_col >= K) return;
  if (bi == bj) {
    if (dst_row > dst_col) C[dst_row * K + dst_col] = tile[tx][ty];
  } else {
    C[dst_row * K + dst_col] = tile[tx][ty];
  }
}

__global__ void finalize_kernel(const float* __restrict__ X,
                                float* __restrict__ output, int M, int N,
                                float scale, bool transpose) {
  const int total = M * N;
  for (int i = blockIdx.x * THREADS + threadIdx.x; i < total;
       i += gridDim.x * THREADS) {
    const float v = transpose ? X[(i % N) * M + (i / N)] : X[i];
    output[i] = v * scale;
  }
}

static void launch_gemm(const float* A, const float* B, float* C, int M, int N,
                        int K, int lda, int ldb, float alpha, float beta,
                        const float* R, int ldr, bool trans_b,
                        bool symmetric = false) {
  const dim3 block(16, 16);
  const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  if (trans_b) {
    gemm_kernel<true, false><<<grid, block>>>(A, B, C, M, N, K, lda, ldb,
                                              alpha, beta, R, ldr);
  } else if (symmetric) {
    gemm_kernel<false, true><<<grid, block>>>(A, B, C, M, N, K, lda, ldb,
                                              alpha, beta, R, ldr);
  } else {
    gemm_kernel<false, false><<<grid, block>>>(A, B, C, M, N, K, lda, ldb,
                                               alpha, beta, R, ldr);
  }
}

// G, Buf, output are device pointers.
extern "C" void solve(const float* G, const float* Buf, float* output, int M,
                      int N, float momentum) {
  if (G == nullptr || Buf == nullptr || output == nullptr) return;
  const int total = M * N;
  if (total <= 0) return;

  const bool transpose = M > N;
  const int K = transpose ? N : M;  // smaller dimension
  const int L = transpose ? M : N;  // larger dimension

  const float a = 3.4445f;
  const float b = -4.7750f;
  const float c = 2.0315f;
  const float scale = sqrtf(static_cast<float>(L));

  // Workspace: X_curr, X_next (K x L), A, B (K x K), norm_sq (1).
  const size_t kl = static_cast<size_t>(K) * L;
  const size_t kk = static_cast<size_t>(K) * K;
  float* workspace = nullptr;
  cudaMalloc(&workspace, (2 * kl + 2 * kk + 1) * sizeof(float));
  float* X_curr = workspace;
  float* X_next = X_curr + kl;
  float* A = X_next + kl;
  float* B = A + kk;
  float* norm_sq = B + kk;
  cudaMemsetAsync(norm_sq, 0, sizeof(float));

  const int elem_blocks = (total + THREADS - 1) / THREADS;
  compute_v_kernel<<<elem_blocks, THREADS>>>(G, Buf, X_curr, norm_sq, M, N,
                                             momentum, transpose);
  normalize_kernel<<<elem_blocks, THREADS>>>(X_curr, norm_sq, total);

  for (int step = 0; step < 5; ++step) {
    // A = X X^T  (K x K), B operand X stored as L x K by using ldb = L.
    launch_gemm(X_curr, X_curr, A, K, K, L, L, L, 1.0f, 0.0f, nullptr, 0,
                /*trans_b=*/true);
    mirror_lower_kernel<<<dim3((K + 31) / 32, (K + 31) / 32), dim3(32, 32)>>>(A, K);
    // B = c A^2 + b A.  Both A and B are symmetric: compute the upper
    // triangle of each and mirror the lower half.
    launch_gemm(A, A, B, K, K, K, K, K, c, b, A, K, /*trans_b=*/false,
                /*symmetric=*/true);
    mirror_lower_kernel<<<dim3((K + 31) / 32, (K + 31) / 32), dim3(32, 32)>>>(B, K);
    // X_next = B X + a X.
    launch_gemm(B, X_curr, X_next, K, L, K, K, L, 1.0f, a, X_curr, L,
                /*trans_b=*/false);
    float* tmp = X_curr;
    X_curr = X_next;
    X_next = tmp;
  }

  finalize_kernel<<<elem_blocks, THREADS>>>(X_curr, output, M, N, scale,
                                            transpose);
  cudaFree(workspace);
  cudaDeviceSynchronize();
}
