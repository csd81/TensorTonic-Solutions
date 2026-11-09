#include <cuda_runtime.h>

// Naive parallel reduction: one thread per input element, each thread adds its
// element straight into output[0] with atomicAdd.  No shared memory, no warp
// shuffles, no vector loads.  output[0] is zeroed first with cudaMemset.

__global__ void reduction_naive_kernel(const float* __restrict__ input,
                                       float* __restrict__ output, int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    atomicAdd(output, input[i]);
  }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
  // The atomic accumulation needs a zeroed destination.
  cudaMemset(output, 0, sizeof(float));

  const int threadsPerBlock = 256;
  const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  reduction_naive_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
  cudaDeviceSynchronize();
}
