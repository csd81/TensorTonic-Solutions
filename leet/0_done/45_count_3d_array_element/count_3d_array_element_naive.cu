#include <cuda_runtime.h>

// Naive count: scalar grid-stride loop with one atomicAdd per thread that
// found matches.  No vector loads, no warp reduction -- the plainest correct
// form (and the starter's shape).

__global__ void count_naive_kernel(const int* __restrict__ input,
                                   int* __restrict__ output,
                                   int total_elements, int P) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  int local_count = 0;
  for (int i = idx; i < total_elements; i += stride) {
    if (input[i] == P) ++local_count;
  }
  if (local_count > 0) atomicAdd(output, local_count);
}

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int M, int K,
                      int P) {
  const int total_elements = N * M * K;
  if (total_elements <= 0) return;
  cudaMemsetAsync(output, 0, sizeof(int));
  const int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;
  if (blocks > 1024) blocks = 1024;
  count_naive_kernel<<<blocks, threads>>>(input, output, total_elements, P);
  cudaDeviceSynchronize();
}
