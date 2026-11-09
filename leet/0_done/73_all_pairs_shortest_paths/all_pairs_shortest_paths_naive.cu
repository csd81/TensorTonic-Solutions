#include <cuda_runtime.h>

// Naive Floyd-Warshall: one kernel per intermediate vertex k, each launching
// one thread per (i,j) pair over the whole matrix, with a device-wide
// synchronisation between steps.  No blocking, no tile reuse -- the plainest
// correct form (and the starter's shape).

#define BLOCK_SIZE 32

__global__ void floyd_warshall_naive_kernel(float* __restrict__ output, int N,
                                            int k) {
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N || j >= N) return;
  const size_t ij = (size_t)i * N + j;
  const float current = output[ij];
  const float ik = output[(size_t)i * N + k];
  const float kj = output[(size_t)k * N + j];
  const float nd = ik + kj;  // inf + finite = inf, inf + inf = inf
  if (nd < current) output[ij] = nd;
}

// dist, output are device pointers
extern "C" void solve(const float* dist, float* output, int N) {
  if (N <= 0) return;
  cudaMemcpy(output, dist, (size_t)N * N * sizeof(float),
             cudaMemcpyDeviceToDevice);
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
  for (int k = 0; k < N; ++k) {
    floyd_warshall_naive_kernel<<<grid, block>>>(output, N, k);
  }
  cudaDeviceSynchronize();
}
