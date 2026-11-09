#include <cuda_runtime.h>

// Naive transpose: one GPU thread per output element.  Each thread computes
// its (row, col) in the transposed matrix with a plain division and reads the
// input directly from global memory (strided reads, coalesced writes).  No
// shared-memory tiling, no padding, one straightforward kernel launch.
__global__ void matrix_transpose_naive_kernel(const float* input, float* output,
                                              int rows, int cols) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * cols;
  if (idx < total) {
    const int out_row = idx / rows;   // row of the cols x rows output
    const int out_col = idx - out_row * rows;
    output[idx] = input[out_col * cols + out_row];
  }
}

// input, output are device pointers (i.e. pointers to memory on the GPU).
extern "C" void solve(const float* input, float* output, int rows, int cols) {
  if (input == nullptr || output == nullptr || rows <= 0 || cols <= 0) return;
  const int total = rows * cols;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  matrix_transpose_naive_kernel<<<blocks, threads>>>(input, output, rows, cols);
  cudaDeviceSynchronize();
}
