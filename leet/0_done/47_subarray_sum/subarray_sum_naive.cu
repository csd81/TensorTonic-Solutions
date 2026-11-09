#include <cuda_runtime.h>

// Naive subarray sum: scalar grid-stride loop with a shared-memory tree
// reduction and one atomicAdd per block.  No vector loads -- the plainest
// correct form (and the starter's shape).

__global__ void subarray_sum_naive_kernel(const int* input, int* output, int S,
                                          int E) {
  extern __shared__ int shared_data[];
  int block_sum = 0;
  const int tid = threadIdx.x;
  const int global_id = blockIdx.x * blockDim.x + tid;
  const int grid_size = blockDim.x * gridDim.x;

  for (int i = S + global_id; i <= E; i += grid_size) block_sum += input[i];

  shared_data[tid] = block_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) shared_data[tid] += shared_data[tid + stride];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(output, shared_data[0]);
}

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int S, int E) {
  cudaMemsetAsync(output, 0, sizeof(int));
  const int elements = E - S + 1;
  if (elements <= 0) return;
  const int threads = 256;
  int blocks = (elements + threads - 1) / threads;
  if (blocks > 2048) blocks = 2048;
  subarray_sum_naive_kernel<<<blocks, threads, threads * sizeof(int)>>>(
      input, output, S, E);
  cudaDeviceSynchronize();
}
