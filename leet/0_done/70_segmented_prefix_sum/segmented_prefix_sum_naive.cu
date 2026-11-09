#include <cuda_runtime.h>

// Naive exclusive segmented prefix sum:
//   output[i] = sum of values[j] for j < i inside i's segment (0 at a flag),
//   where flags[i] == 1 starts a new segment.
//
// One thread per output element.  A thread whose element starts a segment
// writes 0; every other thread walks backwards through global memory until it
// reaches the segment start, accumulating the values in a double.  This is the
// O(N * segment length) direct approach: no shared memory, no scans, no
// lookback protocol, no scratch buffers.

__global__ void segscan_naive_kernel(const float* __restrict__ values,
                                     const int* __restrict__ flags,
                                     float* __restrict__ output, int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  if (flags[i]) {
    output[i] = 0.0f;
    return;
  }

  double acc = 0.0;
  int j = i - 1;
  while (j >= 0) {
    acc += static_cast<double>(values[j]);
    if (flags[j]) break;
    --j;
  }
  output[i] = static_cast<float>(acc);
}

// values, flags, output are device pointers.
extern "C" void solve(const float* values, const int* flags, float* output,
                      int N) {
  if (values == nullptr || flags == nullptr || output == nullptr || N <= 0) {
    return;
  }

  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  segscan_naive_kernel<<<blocks, threads>>>(values, flags, output, N);
  cudaDeviceSynchronize();
}
