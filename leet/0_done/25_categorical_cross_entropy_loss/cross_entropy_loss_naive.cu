#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>

// Naive categorical cross-entropy.
//
// One thread per row.  The thread reads its whole row of logits twice (once
// for the maximum, once for the sum of exp), computes
// log-sum-exp minus the true logit, and adds its share of the mean to the
// single output element with an atomicAdd.  No shared memory, no shuffles.

__global__ void ce_naive_kernel(const float* __restrict__ logits,
                                const int* __restrict__ labels,
                                float* __restrict__ loss, int N, int C) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= N) return;

  const float* r = logits + static_cast<size_t>(row) * C;

  float m = -FLT_MAX;
  for (int c = 0; c < C; ++c) m = fmaxf(m, r[c]);

  float s = 0.0f;
  for (int c = 0; c < C; ++c) s += expf(r[c] - m);

  const float row_loss = (m + logf(s)) - r[labels[row]];
  atomicAdd(loss, row_loss / static_cast<float>(N));
}

// logits, true_labels, loss are device pointers.
extern "C" void solve(const float* logits, const int* true_labels, float* loss,
                      int N, int C) {
  if (logits == nullptr || true_labels == nullptr || loss == nullptr) return;
  if (N <= 0 || C <= 0) return;

  cudaMemsetAsync(loss, 0, sizeof(float));

  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  ce_naive_kernel<<<blocks, threads>>>(logits, true_labels, loss, N, C);
  cudaDeviceSynchronize();
}
