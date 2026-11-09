#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Naive W4A16 matmul: one thread per output element, K-loop in steps of one
// packed byte (two weights), re-reading x, the packed weights and the scales
// from global memory for every output.  No shared memory, no reuse -- the
// plainest correct form (and the starter's shape).

__global__ void w4a16_naive_kernel(const __half* __restrict__ x,
                                   const uint8_t* __restrict__ w_q,
                                   const __half* __restrict__ scales,
                                   __half* __restrict__ y, int M, int N, int K,
                                   int group_size) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;

  const int K2 = K / 2;
  const int NG = K / group_size;
  float acc = 0.0f;
  for (int byte_k = 0; byte_k < K2; ++byte_k) {
    const int k0 = byte_k * 2;
    const int k1 = k0 + 1;
    const uint8_t packed = w_q[(size_t)col * K2 + byte_k];
    const int w0 = (int)((packed >> 4) & 0xF) - 8;
    const int w1 = (int)(packed & 0xF) - 8;
    const float s0 = __half2float(scales[(size_t)col * NG + k0 / group_size]);
    const float s1 = __half2float(scales[(size_t)col * NG + k1 / group_size]);
    const float x0 = __half2float(x[(size_t)row * K + k0]);
    const float x1 = __half2float(x[(size_t)row * K + k1]);
    acc += x0 * (w0 * s0) + x1 * (w1 * s1);
  }
  y[(size_t)row * N + col] = __float2half(acc);
}

// x, w_q, scales, y are device pointers
extern "C" void solve(const __half* x, const uint8_t* w_q, const __half* scales,
                      __half* y, int M, int N, int K, int group_size) {
  if (M <= 0 || N <= 0 || K <= 0) return;
  dim3 block(16, 16);
  dim3 grid((N + 15) / 16, (M + 15) / 16);
  w4a16_naive_kernel<<<grid, block>>>(x, w_q, scales, y, M, N, K, group_size);
  cudaDeviceSynchronize();
}
