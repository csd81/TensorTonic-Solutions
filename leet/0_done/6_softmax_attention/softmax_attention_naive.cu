#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>

// Naive softmax attention: out = softmax(Q K^T / sqrt(d)) V.
//
// One thread per output element (m, j).  Each thread recomputes all N scores
// for its row directly from global memory: a first pass finds the maximum
// score, a second pass accumulates sum(exp(score - max)) and
// sum(exp(score - max) * V[n][j]).  No shared memory, no vector loads, no
// warp intrinsics -- every read goes straight to global memory.

__global__ void softmax_attention_naive_kernel(const float* __restrict__ Q,
                                               const float* __restrict__ K,
                                               const float* __restrict__ V,
                                               float* __restrict__ output,
                                               int M, int N, int d,
                                               float scale) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * d) return;

  const int m = idx / d;
  const int j = idx - m * d;
  const float* q = Q + static_cast<size_t>(m) * d;

  // Pass 1: row maximum of the scaled scores.
  float max_score = -FLT_MAX;
  for (int n = 0; n < N; ++n) {
    const float* k = K + static_cast<size_t>(n) * d;
    float dot = 0.0f;
    for (int t = 0; t < d; ++t) dot += q[t] * k[t];
    dot *= scale;
    if (dot > max_score) max_score = dot;
  }

  // Pass 2: softmax denominator and the weighted sum of this column of V.
  float sum = 0.0f;
  float acc = 0.0f;
  for (int n = 0; n < N; ++n) {
    const float* k = K + static_cast<size_t>(n) * d;
    float dot = 0.0f;
    for (int t = 0; t < d; ++t) dot += q[t] * k[t];
    const float p = expf(dot * scale - max_score);
    sum += p;
    acc += p * V[static_cast<size_t>(n) * d + j];
  }

  output[idx] = acc / sum;
}

// Q, K, V, output are device pointers.
extern "C" void solve(const float* Q, const float* K, const float* V,
                      float* output, int M, int N, int d) {
  if (M <= 0 || N <= 0 || d <= 0) return;

  const float scale = 1.0f / sqrtf(static_cast<float>(d));
  const int total = M * d;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  softmax_attention_naive_kernel<<<blocks, threads>>>(
      Q, K, V, output, M, N, d, scale);
  cudaDeviceSynchronize();
}
