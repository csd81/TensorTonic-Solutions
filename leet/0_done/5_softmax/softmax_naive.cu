#include <cuda_runtime.h>
#include <float.h>

// Naive softmax: the input is one flat array, i.e. a single softmax "row", so
// the plainest mapping is one thread for that row.  A single GPU thread walks
// the array three times (global max, exp + sum, normalize) with plain
// global-memory reads/writes and no vector types or intrinsics beyond expf.

__global__ void softmax_naive_kernel(const float* __restrict__ input,
                                     float* __restrict__ output, int N) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  float max_val = -FLT_MAX;
  for (int i = 0; i < N; ++i) {
    max_val = fmaxf(max_val, input[i]);
  }

  float sum_val = 0.0f;
  for (int i = 0; i < N; ++i) {
    const float e = expf(input[i] - max_val);
    output[i] = e;  // numerator, normalized below
    sum_val += e;
  }

  const float inv_sum = 1.0f / sum_val;
  for (int i = 0; i < N; ++i) {
    output[i] *= inv_sum;
  }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
  softmax_naive_kernel<<<1, 1>>>(input, output, N);
  cudaDeviceSynchronize();
}
