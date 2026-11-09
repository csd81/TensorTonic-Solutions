#include <cuda_runtime.h>

// Naive row-wise scatter-add: one thread per (row, feature) element with one
// global atomicAdd each, after zeroing the output.  This is the starter's
// shape with a 1-D grid-stride loop: the starter's own 2-D grid overflows
// grid.y (N/8 > 65,535) for N > 524,280, so its kernel never runs at the
// benchmark shape.

__global__ void scatter_add_naive_kernel(const float* __restrict__ src,
                                         const int* __restrict__ index,
                                         float* __restrict__ output,
                                         long total, int D) {
  for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < total;
       t += (long)gridDim.x * blockDim.x) {
    const int i = (int)(t / D);
    const int d = (int)(t % D);
    atomicAdd(&output[(size_t)index[i] * D + d], src[t]);
  }
}

// src, index, output are device pointers
extern "C" void solve(const float* src, const int* index, float* output, int N,
                      int D, int M) {
  cudaMemsetAsync(output, 0, (size_t)M * D * sizeof(float));
  if (N <= 0 || D <= 0) return;
  const long total = (long)N * D;
  const int threads = 256;
  long want = (total + threads - 1) / threads;
  const int blocks = (int)(want > 65535 ? 65535 : want);
  scatter_add_naive_kernel<<<blocks, threads>>>(src, index, output, total, D);
  cudaDeviceSynchronize();
}
