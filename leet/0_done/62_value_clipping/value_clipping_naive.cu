#include <cuda_runtime.h>

// Naive value clipping: one thread per element with plain scalar loads and
// stores and two comparisons.  No vector types, no grid-stride loop -- the
// plainest correct form.

__global__ void clip_naive_kernel(const float* __restrict__ input,
                                  float* __restrict__ output, float lo,
                                  float hi, int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  float v = input[i];
  if (v < lo) v = lo;
  if (v > hi) v = hi;
  output[i] = v;
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, float lo, float hi,
                      int N) {
  if (N <= 0) return;
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  clip_naive_kernel<<<blocks, threads>>>(input, output, lo, hi, N);
  cudaDeviceSynchronize();
}
