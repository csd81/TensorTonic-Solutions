#include <cuda_runtime.h>

#include <cstdint>

// Element-wise addition of two N x N float32 matrices: C = A + B.
//
// This is a pure streaming problem: 192 MiB of traffic at N = 4096 (read A,
// read B, write C), so the kernel is bound by DRAM bandwidth (~84 GB/s on the
// P2000) no matter how it is arranged.
//
// The main path loads float4 vectors (16 bytes each) from A and B and stores
// one float4 to C, giving one 128-bit load per input and one 128-bit store
// per thread instead of the starter's scalar accesses.  A scalar fallback
// handles buffers that are not 16-byte aligned, and pixels left over when
// N*N is not a multiple of four (odd N) are handled by thread 0 of block 0 in
// the same launch.

__global__ void add_vec4_kernel(const float4* __restrict__ A,
                                const float4* __restrict__ B,
                                float4* __restrict__ C, int n4,
                                const float* __restrict__ A_tail,
                                const float* __restrict__ B_tail,
                                float* __restrict__ C_tail, int tail) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) {
    const float4 a = A[i];
    const float4 b = B[i];
    C[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
  }
  // At most three leftover elements; handled once, not by every block.
  if (i == 0) {
    for (int p = 0; p < tail; ++p) {
      C_tail[p] = A_tail[p] + B_tail[p];
    }
  }
}

__global__ void add_scalar_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) C[i] = A[i] + B[i];
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU).
extern "C" void solve(const float* A, const float* B, float* C, int N) {
  if (A == nullptr || B == nullptr || C == nullptr || N <= 0) return;

  const int n = N * N;
  const int threads = 256;
  const uintptr_t addr = reinterpret_cast<uintptr_t>(A) |
                         reinterpret_cast<uintptr_t>(B) |
                         reinterpret_cast<uintptr_t>(C);

  if ((addr & 15u) == 0) {
    const int n4 = n >> 2;
    const int tail = n & 3;
    const int blocks = (n4 + threads - 1) / threads;
    if (n4 > 0) {
      add_vec4_kernel<<<blocks, threads>>>(
          reinterpret_cast<const float4*>(A),
          reinterpret_cast<const float4*>(B), reinterpret_cast<float4*>(C), n4,
          A + 4 * n4, B + 4 * n4, C + 4 * n4, tail);
    } else if (tail > 0) {
      add_scalar_kernel<<<1, threads>>>(A, B, C, tail);
    }
  } else {
    const int blocks = (n + threads - 1) / threads;
    add_scalar_kernel<<<blocks, threads>>>(A, B, C, n);
  }
  cudaDeviceSynchronize();
}
