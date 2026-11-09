#include <cuda_runtime.h>

// Naive 5-point Jacobi stencil: one thread per cell with 2-D blocks, a
// boundary branch, and four scalar neighbour loads.  No vector types, no
// register reuse across columns -- the plainest correct form (and the
// starter's shape, with its 32x32 blocks).

__global__ void jacobi_naive_kernel(const float* __restrict__ input,
                                    float* __restrict__ output, int rows,
                                    int cols) {
  const int r = blockIdx.y * blockDim.y + threadIdx.y;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows || c >= cols) return;
  const size_t i = (size_t)r * cols + c;
  if (r == 0 || r == rows - 1 || c == 0 || c == cols - 1) {
    output[i] = input[i];
  } else {
    const float top = input[i - cols];
    const float bottom = input[i + cols];
    const float left = input[i - 1];
    const float right = input[i + 1];
    output[i] = 0.25f * (top + bottom + left + right);
  }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int rows, int cols) {
  if (rows <= 0 || cols <= 0) return;
  dim3 block(32, 32);
  dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
  jacobi_naive_kernel<<<grid, block>>>(input, output, rows, cols);
  cudaDeviceSynchronize();
}
