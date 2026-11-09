#include <cuda_runtime.h>

// Naive reference implementation of an in-place ascending float sort.
//
// One GPU thread per output position, and that thread computes the position's
// rank directly: it scans the whole array and counts the elements that are
// strictly smaller, plus the equal elements that appear earlier (so duplicate
// values keep their original relative order).  The value is then written to
// scratch[rank], and the scratch buffer is copied back over the input because
// an in-place rank write would race with the reads.  This is O(N^2); it is a
// correctness-first baseline, not a practical sort.
//
// The scratch buffer is required by the semantics of a rank-based sort and is
// cached across calls.

__global__ void rank_sort_naive_kernel(const float* __restrict__ input,
                                       float* __restrict__ scratch, int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  const float v = input[i];
  int rank = 0;
  for (int j = 0; j < N; ++j) {
    const float u = input[j];
    if (u < v || (u == v && j < i)) ++rank;
  }
  scratch[rank] = v;
}

static float* g_scratch = nullptr;
static int g_capacity = 0;

// data is a device pointer; the sorted result is written back in place.
extern "C" void solve(float* data, int N) {
  if (N <= 1) return;

  if (N > g_capacity) {
    cudaFree(g_scratch);
    cudaMalloc(&g_scratch, static_cast<size_t>(N) * sizeof(float));
    g_capacity = N;
  }

  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  rank_sort_naive_kernel<<<blocks, threads>>>(data, g_scratch, N);
  cudaMemcpyAsync(data, g_scratch, static_cast<size_t>(N) * sizeof(float),
                  cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
}
