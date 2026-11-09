#include <cuda_runtime.h>

#include <cstdint>

// 1D "valid" convolution:
//     out[i] = sum_{j < kernel_size} in[i + j] * k[j],
//     i < input_size - kernel_size + 1.
//
// This is compute bound: at input_size = 1,500,000 and kernel_size = 2,047
// there are ~3.07 G multiply-adds.  The kernel is organised so every shared
// load feeds eight FMAs:
//
//   * a block of THREADS = 640 threads produces BLOCK_OUT = 5120 outputs;
//     thread t owns the eight *consecutive* outputs [8t, 8t+7],
//   * the kernel and the input tile (BLOCK_OUT + kernel_size - 1 floats) are
//     staged in shared memory,
//   * the inner loop keeps the eight-element input window of the thread in
//     registers (two float4s) and slides it by four taps per iteration:
//     four kernel taps arrive as one broadcast float4, four new input values
//     as one float4 (a warp reads 512 contiguous bytes, conflict free), and
//     32 FMAs are issued per iteration,
//   * a scalar tail handles kernel_size % 4 remaining taps.
//
// Outputs past the valid region are masked; input elements past input_size
// are zero-filled.  The 48 KB static shared-memory limit is respected
// (36,872 bytes at kernel_size = 2048).

#define MAX_KERNEL_SIZE 2048
#define THREADS 640
#define OUT_PER_THREAD 8
#define BLOCK_OUT (THREADS * OUT_PER_THREAD)  // 5120
#define TILE_SIZE (BLOCK_OUT + MAX_KERNEL_SIZE + 2)

__global__ void convolution_1d_kernel(const float* __restrict__ input,
                                      const float* __restrict__ kernel,
                                      float* __restrict__ output,
                                      int input_size, int kernel_size) {
  __shared__ __align__(16) float s_kernel[MAX_KERNEL_SIZE];
  __shared__ __align__(16) float s_input[TILE_SIZE];

  const int tid = threadIdx.x;
  const int block_start = blockIdx.x * BLOCK_OUT;
  const int output_size = input_size - kernel_size + 1;

  // Stage the kernel and the input tile (with the right halo).
  for (int i = tid; i < kernel_size; i += THREADS) {
    s_kernel[i] = kernel[i];
  }
  const int tile_len = BLOCK_OUT + kernel_size - 1;
  for (int i = tid; i < tile_len; i += THREADS) {
    const int g = block_start + i;
    s_input[i] = (g < input_size) ? input[g] : 0.0f;
  }
  // The last float4 window load may touch up to four elements past the tile.
  if (tid < 4) s_input[tile_len + tid] = 0.0f;

  __syncthreads();

  const int i0 = block_start + tid * OUT_PER_THREAD;
  if (i0 >= output_size) return;

  const int base = tid * OUT_PER_THREAD;
  float4 A0 = *reinterpret_cast<const float4*>(&s_input[base]);
  float4 A1 = *reinterpret_cast<const float4*>(&s_input[base + 4]);
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;

  int j = 0;
  for (; j + 4 <= kernel_size; j += 4) {
    const float4 kk = *reinterpret_cast<const float4*>(&s_kernel[j]);
    const float4 B = *reinterpret_cast<const float4*>(&s_input[base + j + 8]);
    // Tap j: window is A0.x .. A1.w.
    a0 += A0.x * kk.x; a1 += A0.y * kk.x; a2 += A0.z * kk.x; a3 += A0.w * kk.x;
    a4 += A1.x * kk.x; a5 += A1.y * kk.x; a6 += A1.z * kk.x; a7 += A1.w * kk.x;
    // Tap j+1: window slides by one (B.x is the new element).
    a0 += A0.y * kk.y; a1 += A0.z * kk.y; a2 += A0.w * kk.y; a3 += A1.x * kk.y;
    a4 += A1.y * kk.y; a5 += A1.z * kk.y; a6 += A1.w * kk.y; a7 += B.x * kk.y;
    // Tap j+2.
    a0 += A0.z * kk.z; a1 += A0.w * kk.z; a2 += A1.x * kk.z; a3 += A1.y * kk.z;
    a4 += A1.z * kk.z; a5 += A1.w * kk.z; a6 += B.x * kk.z; a7 += B.y * kk.z;
    // Tap j+3.
    a0 += A0.w * kk.w; a1 += A1.x * kk.w; a2 += A1.y * kk.w; a3 += A1.z * kk.w;
    a4 += A1.w * kk.w; a5 += B.x * kk.w; a6 += B.y * kk.w; a7 += B.z * kk.w;
    A0 = A1;
    A1 = B;
  }
  for (; j < kernel_size; ++j) {
    const float kk = s_kernel[j];
    a0 += A0.x * kk; a1 += A0.y * kk; a2 += A0.z * kk; a3 += A0.w * kk;
    a4 += A1.x * kk; a5 += A1.y * kk; a6 += A1.z * kk; a7 += A1.w * kk;
    A0.x = A0.y; A0.y = A0.z; A0.z = A0.w; A0.w = A1.x;
    A1.x = A1.y; A1.y = A1.z; A1.z = A1.w;
    A1.w = s_input[base + j + 8];
  }

  if (i0 + 0 < output_size) output[i0 + 0] = a0;
  if (i0 + 1 < output_size) output[i0 + 1] = a1;
  if (i0 + 2 < output_size) output[i0 + 2] = a2;
  if (i0 + 3 < output_size) output[i0 + 3] = a3;
  if (i0 + 4 < output_size) output[i0 + 4] = a4;
  if (i0 + 5 < output_size) output[i0 + 5] = a5;
  if (i0 + 6 < output_size) output[i0 + 6] = a6;
  if (i0 + 7 < output_size) output[i0 + 7] = a7;
}

// input, kernel, output are device pointers.
extern "C" void solve(const float* input, const float* kernel, float* output,
                      int input_size, int kernel_size) {
  if (input == nullptr || kernel == nullptr || output == nullptr) return;
  if (input_size <= 0 || kernel_size <= 0 || kernel_size > MAX_KERNEL_SIZE) {
    return;
  }
  const int output_size = input_size - kernel_size + 1;
  if (output_size <= 0) return;

  const int blocks = (output_size + BLOCK_OUT - 1) / BLOCK_OUT;
  convolution_1d_kernel<<<blocks, THREADS>>>(input, kernel, output, input_size,
                                             kernel_size);
  cudaDeviceSynchronize();
}
