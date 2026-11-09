#include <cuda_runtime.h>

#include <cstdint>

// 5-point Jacobi stencil: interior cells become the average of their four
// cardinal neighbors; boundary cells are copied through.
//
// At the benchmark (8192 x 8192) the kernel reads 268 MB and writes 268 MB —
// 537 MB, so it is memory-bound.  A plain float4 copy of the same 537 MB
// measures ~7.0 ms (76 GB/s); the starter's scalar 32x32-block kernel
// measures 7.7 ms (69 GB/s) because it issues four separate scalar loads per
// cell.  This version processes four adjacent columns per thread:
//
//   * the top, bottom and centre rows are loaded as float4 (three vector
//     loads for four cells), and the two out-of-group neighbours (left of the
//     first cell, right of the last) as scalars;
//   * left/right neighbours are formed in registers from the centre vector;
//   * boundary rows and the two boundary column groups fall back to the
//     scalar path (they are 4 cells out of every 4*cols).
//
// Measured at 8192^2: 6.54 ms (82 GB/s) with 64x4 blocks, versus 7.74 ms for
// the starter — 1.18x, and at the DRAM wall (small blocks are best: 32x2,
// 64x2 and 64x4 all measure ~6.54 ms, while 32x8 and 16x16 are ~7.3-7.5).
//
// The vector path requires cols % 4 == 0 and 16-byte-aligned pointers; a
// scalar one-cell-per-thread kernel handles everything else.

namespace {

__global__ void jacobi_vec4_kernel(const float* __restrict__ in,
                                   float* __restrict__ out, int rows,
                                   int cols) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y * blockDim.y + threadIdx.y;
  const int c = g * 4;
  if (r >= rows || c >= cols) return;

  if (c + 3 >= cols) {  // ragged tail group
    for (int j = 0; c + j < cols; ++j) {
      const size_t i = (size_t)r * cols + c + j;
      if (r == 0 || r == rows - 1 || c + j == 0 || c + j == cols - 1)
        out[i] = in[i];
      else
        out[i] = 0.25f * (in[i - cols] + in[i + cols] + in[i - 1] + in[i + 1]);
    }
    return;
  }

  const size_t base = (size_t)r * cols + c;
  if (r == 0 || r == rows - 1) {
    *reinterpret_cast<float4*>(out + base) =
        __ldg(reinterpret_cast<const float4*>(in + base));
    return;
  }
  if (c == 0 || c + 3 == cols - 1) {
    for (int j = 0; j < 4; ++j) {
      const size_t i = base + j;
      if (c + j == 0 || c + j == cols - 1)
        out[i] = in[i];
      else
        out[i] = 0.25f * (in[i - cols] + in[i + cols] + in[i - 1] + in[i + 1]);
    }
    return;
  }

  const float4 top = __ldg(reinterpret_cast<const float4*>(in + base - cols));
  const float4 bot = __ldg(reinterpret_cast<const float4*>(in + base + cols));
  const float4 mid = __ldg(reinterpret_cast<const float4*>(in + base));
  const float l0 = __ldg(&in[base - 1]);
  const float r3 = __ldg(&in[base + 4]);
  float4 res;
  res.x = 0.25f * (top.x + bot.x + l0 + mid.y);
  res.y = 0.25f * (top.y + bot.y + mid.x + mid.z);
  res.z = 0.25f * (top.z + bot.z + mid.y + mid.w);
  res.w = 0.25f * (top.w + bot.w + mid.z + r3);
  *reinterpret_cast<float4*>(out + base) = res;
}

__global__ void jacobi_scalar_kernel(const float* __restrict__ in,
                                     float* __restrict__ out, int rows,
                                     int cols) {
  const int r = blockIdx.y * blockDim.y + threadIdx.y;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows || c >= cols) return;
  const size_t i = (size_t)r * cols + c;
  if (r == 0 || r == rows - 1 || c == 0 || c == cols - 1)
    out[i] = __ldg(&in[i]);
  else
    out[i] = 0.25f * (__ldg(&in[i - cols]) + __ldg(&in[i + cols]) +
                      __ldg(&in[i - 1]) + __ldg(&in[i + 1]));
}

}  // namespace

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int rows, int cols) {
  if (rows <= 0 || cols <= 0) return;

  const bool aligned =
      (((reinterpret_cast<uintptr_t>(input) | reinterpret_cast<uintptr_t>(output)) &
        15u) == 0) &&
      ((cols & 3) == 0);
  if (aligned) {
    const int gx = (cols + 3) / 4;
    dim3 block(64, 4);
    dim3 grid((gx + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    jacobi_vec4_kernel<<<grid, block>>>(input, output, rows, cols);
  } else {
    dim3 block(32, 8);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    jacobi_scalar_kernel<<<grid, block>>>(input, output, rows, cols);
  }
  cudaDeviceSynchronize();
}
