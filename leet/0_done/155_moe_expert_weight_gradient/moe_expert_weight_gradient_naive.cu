#include <cuda_runtime.h>

// Naive MoE expert weight gradient: one GPU thread per output element
// (expert g, row k, column n).  The thread walks the expert's ragged token
// range directly in global memory and writes its own entry of dW[g], so empty
// experts naturally write zeros and every entry is written.

#define MOE_THREADS 256

__global__ void moe_weight_grad_kernel(const float* __restrict__ X,
                                       const float* __restrict__ dY,
                                       const int* __restrict__ group_offsets,
                                       float* __restrict__ dW, int K, int N) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= K * N) return;

  const int g = blockIdx.y;
  const int k = idx / N;
  const int n = idx - k * N;

  const int s = group_offsets[g];
  const int e = group_offsets[g + 1];

  float acc = 0.0f;
  for (int m = s; m < e; ++m) {
    acc += X[static_cast<size_t>(m) * K + k] *
           dY[static_cast<size_t>(m) * N + n];
  }
  dW[(static_cast<size_t>(g) * K + k) * N + n] = acc;
}

// X, dY, group_offsets, dW are device pointers.
extern "C" void solve(const float* X, const float* dY,
                      const int* group_offsets, float* dW, int G, int M_total,
                      int K, int N) {
  if (X == nullptr || dY == nullptr || group_offsets == nullptr ||
      dW == nullptr)
    return;
  if (G <= 0 || M_total <= 0 || K <= 0 || N <= 0) return;

  const int per_expert = K * N;
  const dim3 grid((per_expert + MOE_THREADS - 1) / MOE_THREADS, G);
  moe_weight_grad_kernel<<<grid, MOE_THREADS>>>(X, dY, group_offsets, dW, K,
                                                N);
  cudaDeviceSynchronize();
}
