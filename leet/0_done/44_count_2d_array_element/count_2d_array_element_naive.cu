#include <cuda_runtime.h>

// Naive 2D count: scalar grid-stride loop with a shared-memory tree reduction
// and one atomicAdd per block.  No vector loads -- the plainest correct form
// (and the starter's shape).

__global__ void count_naive_kernel(const int* input, int* output,
                                   long total_elements, int P) {
  extern __shared__ int shared_counts[];
  const int tid = threadIdx.x;
  const long stride = (long)blockDim.x * gridDim.x;

  int local_count = 0;
  for (long i = (long)blockIdx.x * blockDim.x + tid; i < total_elements;
       i += stride)
    local_count += (input[i] == P);

  shared_counts[tid] = local_count;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) shared_counts[tid] += shared_counts[tid + s];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(output, shared_counts[0]);
}

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int M, int P) {
  const long total = (long)N * M;
  cudaMemsetAsync(output, 0, sizeof(int));
  if (total <= 0) return;
  const int threads = 256;
  long want = (total + threads - 1) / threads;
  const int blocks = (int)(want > 4096 ? 4096 : want);
  count_naive_kernel<<<blocks, threads, threads * sizeof(int)>>>(input, output,
                                                                total, P);
  cudaDeviceSynchronize();
}
