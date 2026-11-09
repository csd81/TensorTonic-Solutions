#include <cuda_runtime.h>

// Naive merge-path: one binary search per output element, each a ~25-probe
// walk through the full input arrays, then one output written.  No tiles, no
// sharing, no coalescing structure -- the plainest correct form (and the
// starter's shape).

__global__ void merge_path_naive_kernel(const float* __restrict__ A, int M,
                                        const float* __restrict__ B, int N,
                                        float* __restrict__ C, int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int d = idx; d < total; d += stride) {
    int low = max(0, d - N), high = min(d, M);
    while (low < high) {
      const int mid = low + ((high - low) >> 1);
      const int jb = d - 1 - mid;
      if (A[mid] <= B[jb]) low = mid + 1; else high = mid;
    }
    const int pos_A = low;
    const int pos_B = d - low;
    if (pos_A < M && (pos_B >= N || A[pos_A] <= B[pos_B]))
      C[d] = A[pos_A];
    else
      C[d] = B[pos_B];
  }
}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
  const int total = M + N;
  if (total <= 0) return;
  const int threads = 256;
  int blocks = (total + threads - 1) / threads;
  if (blocks > 65535) blocks = 65535;
  merge_path_naive_kernel<<<blocks, threads>>>(A, M, B, N, C, total);
  cudaDeviceSynchronize();
}
