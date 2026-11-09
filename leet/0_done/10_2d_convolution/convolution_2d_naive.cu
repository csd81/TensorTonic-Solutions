#include <cuda_runtime.h>

// Naive reference implementation of the "valid" 2D convolution:
//   out[i][j] = sum_{m<KR,n<KC} in[i+m][j+n] * k[m][n].
//
// One GPU thread per output element, one block of 16x16 threads per 16x16
// output tile.  Every tap is a direct global-memory load of the input and the
// kernel; there is no shared-memory tiling, no vector type, no unrolling
// trick and no register blocking.  This exists only as a correctness-first
// baseline, not for speed.

__global__ void conv2d_valid_naive_kernel(const float* __restrict__ input,
                                          const float* __restrict__ kernel,
                                          float* __restrict__ output,
                                          int input_rows, int input_cols,
                                          int kernel_rows, int kernel_cols,
                                          int output_rows, int output_cols) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;  // output column
  const int i = blockIdx.y * blockDim.y + threadIdx.y;  // output row
  if (i >= output_rows || j >= output_cols) return;

  float acc = 0.0f;
  for (int m = 0; m < kernel_rows; ++m) {
    const float* in_row =
        input + static_cast<size_t>(i + m) * input_cols + j;
    const float* k_row = kernel + m * kernel_cols;
    for (int n = 0; n < kernel_cols; ++n) {
      acc += in_row[n] * k_row[n];
    }
  }
  output[static_cast<size_t>(i) * output_cols + j] = acc;
}

// input, kernel, output are device pointers.
extern "C" void solve(const float* input, const float* kernel, float* output,
                      int input_rows, int input_cols, int kernel_rows,
                      int kernel_cols) {
  if (input == nullptr || kernel == nullptr || output == nullptr) return;
  if (input_rows <= 0 || input_cols <= 0 || kernel_rows <= 0 ||
      kernel_cols <= 0) {
    return;
  }
  if (kernel_rows > 31 || kernel_cols > 31) return;
  const int output_rows = input_rows - kernel_rows + 1;
  const int output_cols = input_cols - kernel_cols + 1;
  if (output_rows <= 0 || output_cols <= 0) return;

  const dim3 block(16, 16);
  const dim3 grid((output_cols + block.x - 1) / block.x,
                  (output_rows + block.y - 1) / block.y);
  conv2d_valid_naive_kernel<<<grid, block>>>(
      input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols,
      output_rows, output_cols);
  cudaDeviceSynchronize();
}
