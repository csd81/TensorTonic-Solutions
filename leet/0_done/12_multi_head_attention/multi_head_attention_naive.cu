#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>

// Naive multi-head self-attention.
//
// One thread per output element (row, column).  The column selects the head
// (contiguous d_k-wide slice) and the position j inside it.  Each thread walks
// all N keys from global memory twice: first to find the maximum score, then
// to accumulate the softmax denominator and the weighted sum over V.

__global__ void mha_naive_kernel(const float* __restrict__ Q,
                                 const float* __restrict__ K,
                                 const float* __restrict__ V,
                                 float* __restrict__ output, int N,
                                 int d_model, int d_k, float scale) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * d_model) return;

  const int row = idx / d_model;
  const int col = idx - row * d_model;
  const int head = col / d_k;
  const int j = col - head * d_k;
  const int off = head * d_k;

  const float* q = Q + static_cast<size_t>(row) * d_model + off;

  // Pass 1: row maximum of the scaled scores for this head.
  float max_score = -FLT_MAX;
  for (int n = 0; n < N; ++n) {
    const float* k = K + static_cast<size_t>(n) * d_model + off;
    float dot = 0.0f;
    for (int t = 0; t < d_k; ++t) dot += q[t] * k[t];
    dot *= scale;
    if (dot > max_score) max_score = dot;
  }

  // Pass 2: softmax denominator and the weighted sum of this output column.
  float sum = 0.0f;
  float acc = 0.0f;
  for (int n = 0; n < N; ++n) {
    const float* k = K + static_cast<size_t>(n) * d_model + off;
    float dot = 0.0f;
    for (int t = 0; t < d_k; ++t) dot += q[t] * k[t];
    const float p = expf(dot * scale - max_score);
    sum += p;
    acc += p * V[static_cast<size_t>(n) * d_model + off + j];
  }

  output[idx] = acc / sum;
}

// Q, K, V, output are device pointers (N x d_model, row-major).
extern "C" void solve(const float* Q, const float* K, const float* V,
                      float* output, int N, int d_model, int h) {
  if (Q == nullptr || K == nullptr || V == nullptr || output == nullptr) return;
  if (N <= 0 || d_model <= 0 || h <= 0 || d_model % h != 0) return;

  const int d_k = d_model / h;
  const float scale = 1.0f / sqrtf(static_cast<float>(d_k));
  const long long total = static_cast<long long>(N) * d_model;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  mha_naive_kernel<<<blocks, threads>>>(Q, K, V, output, N, d_model, d_k, scale);
  cudaDeviceSynchronize();
}
