#include <cuda_runtime.h>

// Naive causal depthwise 1D convolution over a channels-last (B, L, D) tensor:
//
//   out[b, l, d] = bias[d] + sum_{k<K} weight[d, k] * x[b, l-k, d]
//
// with x[b, l-k, d] = 0 for l-k < 0.  One thread per output element; each
// thread reads its K inputs directly from global memory.  No shared memory,
// no vector types, no register sliding window.

__global__ void causal_depthwise_conv1d_naive_kernel(
    const float* __restrict__ x, const float* __restrict__ weight,
    const float* __restrict__ bias, float* __restrict__ output,
    int B, int L, int D, int K) {
  const long long total = (long long)B * L * D;
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  const int d = (int)(idx % D);
  const long long t = idx / D;
  const int l = (int)(t % L);
  const int b = (int)(t / L);

  const long long base = ((long long)b * L + l) * D + d;
  float acc = bias[d];
  for (int k = 0; k < K; ++k) {
    const int ll = l - k;
    if (ll >= 0) {
      acc += weight[d * K + k] * x[base - (long long)k * D];
    }
  }
  output[idx] = acc;
}

// x, weight, bias, output are device pointers.
extern "C" void solve(const float* x, const float* weight, const float* bias,
                      float* output, int B, int L, int D, int K) {
  if (B <= 0 || L <= 0 || D <= 0 || K <= 0 || K > 8) return;

  const long long total = (long long)B * L * D;
  const int threadsPerBlock = 256;
  const long long blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;

  causal_depthwise_conv1d_naive_kernel<<<(unsigned int)blocksPerGrid,
                                         threadsPerBlock>>>(
      x, weight, bias, output, B, L, D, K);
  cudaDeviceSynchronize();
}
