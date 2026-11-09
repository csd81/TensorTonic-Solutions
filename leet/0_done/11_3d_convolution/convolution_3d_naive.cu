#include <cuda_runtime.h>

// Naive reference implementation of the "valid" 3D convolution:
//   out[i][j][k] = sum_{d<KD,r<KR,c<KC} in[i+d][j+r][k+c] * k[d][r][c].
//
// One GPU thread per output element.  A block is 16x16 threads over the
// (column, row) plane and blockIdx.z selects the output depth.  Every tap is
// a direct global-memory load of the input and the kernel; no shared-memory
// tiling, no vector type, no register blocking.  Correctness-first baseline.

__global__ void conv3d_naive_kernel(const float* __restrict__ input,
                                    const float* __restrict__ kernel,
                                    float* __restrict__ output, int in_d,
                                    int in_r, int in_c, int kd, int kr, int kc,
                                    int out_d, int out_r, int out_c) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;  // output column
  const int r = blockIdx.y * blockDim.y + threadIdx.y;  // output row
  const int d = blockIdx.z;                             // output depth
  if (c >= out_c || r >= out_r) return;

  float acc = 0.0f;
  for (int kz = 0; kz < kd; ++kz) {
    for (int ky = 0; ky < kr; ++ky) {
      const float* in_row =
          input + ((static_cast<size_t>(d + kz) * in_r + (r + ky)) * in_c + c);
      const float* k_row = kernel + (kz * kr + ky) * kc;
      for (int kx = 0; kx < kc; ++kx) {
        acc += in_row[kx] * k_row[kx];
      }
    }
  }
  output[((static_cast<size_t>(d) * out_r + r) * out_c) + c] = acc;
}

// input, kernel, output are device pointers.
extern "C" void solve(const float* input, const float* kernel, float* output,
                      int input_depth, int input_rows, int input_cols,
                      int kernel_depth, int kernel_rows, int kernel_cols) {
  if (input == nullptr || kernel == nullptr || output == nullptr) return;
  if (input_depth <= 0 || input_rows <= 0 || input_cols <= 0 ||
      kernel_depth <= 0 || kernel_rows <= 0 || kernel_cols <= 0) {
    return;
  }
  if (kernel_depth > 5 || kernel_rows > 5 || kernel_cols > 5) return;
  const int out_d = input_depth - kernel_depth + 1;
  const int out_r = input_rows - kernel_rows + 1;
  const int out_c = input_cols - kernel_cols + 1;
  if (out_d <= 0 || out_r <= 0 || out_c <= 0) return;

  const dim3 block(16, 16);
  const dim3 grid((out_c + block.x - 1) / block.x,
                  (out_r + block.y - 1) / block.y, out_d);
  conv3d_naive_kernel<<<grid, block>>>(
      input, kernel, output, input_depth, input_rows, input_cols,
      kernel_depth, kernel_rows, kernel_cols, out_d, out_r, out_c);
  cudaDeviceSynchronize();
}
