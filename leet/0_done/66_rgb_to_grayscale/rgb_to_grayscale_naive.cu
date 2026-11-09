#include <cuda_runtime.h>

// Naive RGB -> grayscale: one thread per pixel, three scalar loads and one
// scalar store.  No vector types, no grid-stride loop -- the plainest correct
// form.  (The starter looks similar but has a broken index:
// blockIdx.x * blockIdx.y * blockDim.x with a 1-D grid, so blockIdx.y is
// always 0 and every block rewrites the first 256 pixels.)

__global__ void rgb_to_gray_naive_kernel(const float* __restrict__ input,
                                         float* __restrict__ output,
                                         int pixels) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p < pixels) {
    const int o = 3 * p;
    output[p] = 0.299f * input[o] + 0.587f * input[o + 1] +
                0.114f * input[o + 2];
  }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int width, int height) {
  if (width <= 0 || height <= 0) return;
  const int pixels = width * height;
  const int threads = 256;
  const int blocks = (pixels + threads - 1) / threads;
  rgb_to_gray_naive_kernel<<<blocks, threads>>>(input, output, pixels);
  cudaDeviceSynchronize();
}
