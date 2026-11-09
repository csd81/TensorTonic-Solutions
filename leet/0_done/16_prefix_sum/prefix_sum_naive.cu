#include <cuda_runtime.h>

// Naive inclusive prefix sum: one GPU thread per output element, each summing
// input[0..i] directly from global memory.  O(N^2) work, no shared memory, no
// vector types, no warp intrinsics, no lookback -- the plainest correct form.
__global__ void prefix_sum_naive_kernel(const float* __restrict__ input,
                                        float* __restrict__ output, int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    float s = 0.0f;
    for (int j = 0; j <= i; ++j) {
      s += input[j];
    }
    output[i] = s;
  }
}

// input, output are device pointers.
extern "C" void solve(const float* input, float* output, int N) {
  if (input == nullptr || output == nullptr || N <= 0) return;
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  prefix_sum_naive_kernel<<<blocks, threads>>>(input, output, N);
  cudaDeviceSynchronize();
}
