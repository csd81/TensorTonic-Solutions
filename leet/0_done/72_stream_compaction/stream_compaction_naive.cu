#include <cuda_runtime.h>

// Naive stream compaction: count the positives of each contiguous block,
// sequentially scan the block counts, then let one thread per block write the
// block's positives in order.  All reads and writes go straight to global
// memory; there is no shared-memory scan, no vector load and no lookback.

#define SC_ELEMS_PER_BLOCK 1024
#define SC_THREADS 256

__global__ void count_kernel(const float* __restrict__ A, int N,
                             int* __restrict__ counts) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  const int base = b * SC_ELEMS_PER_BLOCK;
  if (base >= N) return;
  int end = base + SC_ELEMS_PER_BLOCK;
  if (end > N) end = N;

  int c = 0;
  for (int i = base; i < end; ++i) {
    if (A[i] > 0.0f) ++c;
  }
  counts[b] = c;
}

// Plain sequential exclusive scan of the per-block counts.
__global__ void scan_kernel(const int* __restrict__ counts,
                            int* __restrict__ offsets, int nblocks) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int running = 0;
  for (int b = 0; b < nblocks; ++b) {
    offsets[b] = running;
    running += counts[b];
  }
}

__global__ void scatter_kernel(const float* __restrict__ A, int N,
                               const int* __restrict__ offsets,
                               float* __restrict__ out) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  const int base = b * SC_ELEMS_PER_BLOCK;
  if (base >= N) return;
  int end = base + SC_ELEMS_PER_BLOCK;
  if (end > N) end = N;

  int pos = offsets[b];
  for (int i = base; i < end; ++i) {
    const float v = A[i];
    if (v > 0.0f) out[pos++] = v;
  }
}

// A, out are device pointers.  out is pre-zeroed by the caller, so only the
// selected elements are written.
extern "C" void solve(const float* A, int N, float* out) {
  if (A == nullptr || out == nullptr || N <= 0) return;

  const int nblocks = (N + SC_ELEMS_PER_BLOCK - 1) / SC_ELEMS_PER_BLOCK;
  int* counts = nullptr;
  int* offsets = nullptr;
  cudaMalloc(&counts, static_cast<size_t>(nblocks) * sizeof(int));
  cudaMalloc(&offsets, static_cast<size_t>(nblocks) * sizeof(int));

  const int grid = (nblocks + SC_THREADS - 1) / SC_THREADS;
  count_kernel<<<grid, SC_THREADS>>>(A, N, counts);
  scan_kernel<<<1, 1>>>(counts, offsets, nblocks);
  scatter_kernel<<<grid, SC_THREADS>>>(A, N, offsets, out);

  cudaFree(counts);
  cudaFree(offsets);
  cudaDeviceSynchronize();
}
