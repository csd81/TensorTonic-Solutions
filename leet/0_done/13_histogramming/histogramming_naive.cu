#include <cuda_runtime.h>

// Naive reference implementation of the histogram of 32-bit integers in
// [0, num_bins).
//
// One GPU thread per input element, and every element performs a single
// global-memory atomicAdd into the histogram.  There is no shared-memory
// privatisation, no per-warp histogram, no vector load and no grid-stride
// loop.  The only staging is the memset that zeroes the output, which the
// semantics require.  Correctness-first baseline, not a fast kernel.

__global__ void histogram_naive_kernel(const int* __restrict__ input,
                                       int* __restrict__ histogram, int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    atomicAdd(&histogram[input[i]], 1);
  }
}

// input, histogram are device pointers.
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
  if (N <= 0 || num_bins <= 0) return;

  cudaMemsetAsync(histogram, 0, static_cast<size_t>(num_bins) * sizeof(int));

  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  histogram_naive_kernel<<<blocks, threads>>>(input, histogram, N);
}
