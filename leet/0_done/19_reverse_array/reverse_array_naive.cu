#include <cuda_runtime.h>

// Naive in-place reversal: one GPU thread per mirror pair (i, N-1-i).  Each
// thread performs plain scalar global loads and stores.  No vector types, no
// shared memory, no shuffles, one straightforward kernel launch.
__global__ void reverse_array_naive_kernel(float* input, int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N / 2) {
    const int k = N - 1 - i;
    const float tmp = input[i];
    input[i] = input[k];
    input[k] = tmp;
  }
}

// input is a device pointer; the reversal is done in place.
extern "C" void solve(float* input, int N) {
  if (input == nullptr || N <= 1) return;
  const int half = N / 2;
  const int threads = 256;
  const int blocks = (half + threads - 1) / threads;
  reverse_array_naive_kernel<<<blocks, threads>>>(input, N);
  cudaDeviceSynchronize();
}
