#include <cuda_runtime.h>

#include <cmath>

// Naive Muon optimizer update: every kernel uses one GPU thread per output
// element and reads its operands directly from global memory.  The three
// matrix products per Newton-Schulz step are plain triple-loop GEMMs with no
// shared-memory tiling, no register blocking and no symmetry shortcut:
//
//   V = momentum * Buf + G                      (norm via one global atomic)
//   X = V / (||V||_F + 1e-7)
//   5x:  A = X X^T;  B = b A + c A^2;  X = a X + B X
//   output = X * sqrt(max(M, N))
//
// When M > N the reference may transpose, but the untransposed formula is
// mathematically identical, so this version always uses M x M Grams.

#define MU_THREADS 256

__global__ void compute_v_kernel(const float* __restrict__ G,
                                 const float* __restrict__ Buf,
                                 float* __restrict__ X,
                                 float* __restrict__ norm_sq, int total,
                                 float momentum) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const float v = momentum * Buf[i] + G[i];
  X[i] = v;
  atomicAdd(norm_sq, v * v);
}

__global__ void normalize_kernel(float* __restrict__ X,
                                 const float* __restrict__ norm_sq,
                                 int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const float norm = sqrtf(*norm_sq);
  X[i] = X[i] / (norm + 1e-7f);
}

// C[r][c] = alpha * sum_k A[r][k] * B[.][.] + beta * R[r][c], one thread per
// output element.  trans_b selects whether B is stored row-major (K x N) or
// transposed (N x K).
__global__ void gemm_naive_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C, int M, int N, int K,
                                  int lda, int ldb, float alpha, float beta,
                                  const float* __restrict__ R, int ldr,
                                  int trans_b) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * N) return;
  const int r = idx / N;
  const int c = idx - r * N;

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    const float av = A[r * lda + k];
    const float bv = trans_b ? B[c * ldb + k] : B[k * ldb + c];
    acc += av * bv;
  }

  float v = alpha * acc;
  if (beta != 0.0f) v += beta * R[r * ldr + c];
  C[r * N + c] = v;
}

__global__ void finalize_kernel(const float* __restrict__ X,
                                float* __restrict__ output, int total,
                                float scale) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  output[i] = X[i] * scale;
}

// G, Buf, output are device pointers.
extern "C" void solve(const float* G, const float* Buf, float* output, int M,
                      int N, float momentum) {
  if (G == nullptr || Buf == nullptr || output == nullptr) return;
  if (M <= 0 || N <= 0) return;

  const int total = M * N;
  const float a = 3.4445f;
  const float b = -4.7750f;
  const float c = 2.0315f;
  const float scale = sqrtf(static_cast<float>(M > N ? M : N));

  const size_t mn = static_cast<size_t>(M) * N;
  const size_t mm = static_cast<size_t>(M) * M;
  float* X = nullptr;
  float* X_next = nullptr;
  float* Amat = nullptr;
  float* Bmat = nullptr;
  float* norm_sq = nullptr;
  cudaMalloc(&X, mn * sizeof(float));
  cudaMalloc(&X_next, mn * sizeof(float));
  cudaMalloc(&Amat, mm * sizeof(float));
  cudaMalloc(&Bmat, mm * sizeof(float));
  cudaMalloc(&norm_sq, sizeof(float));
  cudaMemsetAsync(norm_sq, 0, sizeof(float));

  const int elem_blocks = (total + MU_THREADS - 1) / MU_THREADS;
  const int gram_blocks = (M * M + MU_THREADS - 1) / MU_THREADS;

  compute_v_kernel<<<elem_blocks, MU_THREADS>>>(G, Buf, X, norm_sq, total,
                                                momentum);
  normalize_kernel<<<elem_blocks, MU_THREADS>>>(X, norm_sq, total);

  for (int step = 0; step < 5; ++step) {
    // A = X X^T  (M x M, reduction over N).
    gemm_naive_kernel<<<gram_blocks, MU_THREADS>>>(
        X, X, Amat, M, M, N, N, N, 1.0f, 0.0f, nullptr, 0,
        /*trans_b=*/1);
    // B = c A^2 + b A  (M x M, reduction over M).
    gemm_naive_kernel<<<gram_blocks, MU_THREADS>>>(
        Amat, Amat, Bmat, M, M, M, M, M, c, b, Amat, M, /*trans_b=*/0);
    // X_next = B X + a X  (M x N, reduction over M).
    gemm_naive_kernel<<<elem_blocks, MU_THREADS>>>(
        Bmat, X, X_next, M, N, M, M, N, 1.0f, a, X, N, /*trans_b=*/0);
    float* tmp = X;
    X = X_next;
    X_next = tmp;
  }

  finalize_kernel<<<elem_blocks, MU_THREADS>>>(X, output, total, scale);

  cudaFree(X);
  cudaFree(X_next);
  cudaFree(Amat);
  cudaFree(Bmat);
  cudaFree(norm_sq);
  cudaDeviceSynchronize();
}
