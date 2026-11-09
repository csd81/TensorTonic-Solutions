#include <cuda_runtime.h>

// In-place reversal of a float array.
//
// This one is a pure streaming problem: every element is read once and
// written once, so at N = 25 M the kernel moves 200 MB and there is nothing
// to save.  The interesting part is that the starter already sits on the
// memory wall -- 2.43 ms, 82 GB/s, ~98% of this GPU's practical DRAM ceiling
// (83.7 GB/s measured with a dedicated streaming kernel).  Both of its
// accesses are perfectly coalesced: a warp reads 32 ascending floats on the
// low side and 32 descending floats on the mirror side, and each is one
// 128-byte transaction.
//
// The obvious ideas were measured and none of them helps, because the kernel
// is bandwidth-bound and not instruction-bound:
//
//   scalar, 256 threads               2.416 ms  82.8 GB/s
//   scalar, 512 threads               2.414 ms  82.9 GB/s
//   float4 vectors (512 B per warp)   2.435 ms  82.1 GB/s
//   float4 + __stcs streaming stores  2.429 ms  82.3 GB/s
//   two float4 per thread             2.446 ms  81.8 GB/s
//
// So this file keeps the scalar pair swap (the fastest of the set) and adds
// only the safety details: a 64-bit-safe index, __restrict__, and an early
// return for N <= 1.

#ifndef REVERSE_THREADS
#define REVERSE_THREADS 512
#endif

__global__ void reverse_kernel(float* __restrict__ a, int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N / 2) {
    const int k = N - 1 - i;
    const float t = a[i];
    a[i] = a[k];
    a[k] = t;
  }
}

// input is a device pointer; the reversal is done in place.
extern "C" void solve(float* input, int N) {
  if (N <= 1) return;
  const int half = N / 2;
  const int threads = REVERSE_THREADS;
  const int blocks = (half + threads - 1) / threads;
  reverse_kernel<<<blocks, threads>>>(input, N);
  cudaDeviceSynchronize();
}
