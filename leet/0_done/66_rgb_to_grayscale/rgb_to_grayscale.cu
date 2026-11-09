#include <cuda_runtime.h>

// RGB -> grayscale: out[p] = 0.299*R + 0.587*G + 0.114*B, with RGB triples
// stored consecutively in the input.
//
// At the benchmark (2048 x 2048 = 4,194,304 pixels) the kernel reads 50.3 MB
// and writes 16.8 MB — 67 MB, so it is memory-bound.  The same traffic with
// minimal math measures 0.80 ms (84 GB/s); this kernel measures 0.80 ms too.
//
// The starter is *incorrect*: its index is
//     blockIdx.x * blockIdx.y * blockDim.x + threadIdx.x
// and the launch is 1-D, so blockIdx.y == 0 for every block and every block
// recomputes pixels 0..255.  The rest of the output is never written.  The
// fix is the plain global 1-D index; this file keeps the scalar
// one-pixel-per-thread mapping because that is what measures best.
//
// A vectorized version (4 pixels per thread: three float4 loads at
// 16-byte-aligned offsets per 48-byte group, one float4 store) was measured
// at 0.807-0.810 ms — 1% slower than scalar, since DRAM, not instruction
// count, is the limit.  So the scalar mapping, with __ldg and 256-thread
// blocks, is what ships.

#ifndef THREADS
#define THREADS 256
#endif

__global__ void rgb_to_gray_kernel(const float* __restrict__ input,
                                   float* __restrict__ output, int pixels) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p < pixels) {
    const int o = 3 * p;
    output[p] = 0.299f * __ldg(&input[o]) + 0.587f * __ldg(&input[o + 1]) +
                0.114f * __ldg(&input[o + 2]);
  }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int width, int height) {
  if (width <= 0 || height <= 0) return;
  const int pixels = width * height;
  const int blocks = (pixels + THREADS - 1) / THREADS;
  rgb_to_gray_kernel<<<blocks, THREADS>>>(input, output, pixels);
  cudaDeviceSynchronize();
}
