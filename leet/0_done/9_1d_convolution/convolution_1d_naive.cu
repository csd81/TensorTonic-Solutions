#include <cuda_runtime.h>

// Naive "valid" 1D convolution (cross-correlation, as the challenge defines it):
//     out[i] = sum_{j < kernel_size} in[i + j] * k[j],
//     i < input_size - kernel_size + 1.
// One thread per output element, walking the taps with plain global-memory
// loads.  No shared-memory tile, no vector types, no fast-math intrinsics.

__global__ void convolution_1d_naive_kernel(const float* __restrict__ input,
                                            const float* __restrict__ kernel,
                                            float* __restrict__ output,
                                            int input_size, int kernel_size) {
  const int output_size = input_size - kernel_size + 1;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= output_size) return;

  const float* p = input + i;
  float acc = 0.0f;
  for (int j = 0; j < kernel_size; ++j) {
    acc += p[j] * kernel[j];
  }
  output[i] = acc;
}

// input, kernel, output are device pointers.
extern "C" void solve(const float* input, const float* kernel, float* output,
                      int input_size, int kernel_size) {
  if (input == nullptr || kernel == nullptr || output == nullptr) return;
  if (input_size <= 0 || kernel_size <= 0 || kernel_size > 2048) return;
  const int output_size = input_size - kernel_size + 1;
  if (output_size <= 0) return;

  const int threadsPerBlock = 256;
  const int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

  convolution_1d_naive_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      input, kernel, output, input_size, kernel_size);
  cudaDeviceSynchronize();
}
