#include <cuda_runtime.h>

// Naive count: one GPU thread per input element.  Each thread reads its
// element directly from global memory and, on a match, performs one
// atomicAdd to the single output element.  No vector loads, no grid-stride
// loop, no shared memory, no warp shuffles; one straightforward kernel
// launch (plus a memset that clears the single output element).
__global__ void count_array_element_naive_kernel(const int* input, int* output,
                                                 int N, int K) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N && input[i] == K) {
    atomicAdd(output, 1);
  }
}

// input, output are device pointers; the result is written to output[0].
extern "C" void solve(const int* input, int* output, int N, int K) {
  if (input == nullptr || output == nullptr || N <= 0) return;
  cudaMemsetAsync(output, 0, sizeof(int));
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  count_array_element_naive_kernel<<<blocks, threads>>>(input, output, N, K);
  cudaDeviceSynchronize();
}
