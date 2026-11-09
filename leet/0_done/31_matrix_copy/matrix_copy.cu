#include <cuda_runtime.h>

// Copy an N x N float matrix: B[i] = A[i].  N <= 4096; the benchmark is
// N = 4096, i.e. 64 MB in + 64 MB out = 128 MB of traffic.
//
// This is a pure DRAM copy, and every plausible mapping is already fully
// coalesced.  Measured at N = 4096 (medians of 50 launches, same harness):
//
//   scalar, 256 thr (this file)          1.628-1.646 ms   ~82 GB/s
//   scalar, 512 thr                      1.634-1.644 ms
//   float4, 128 / 256 / 512 thr          1.630-1.649 ms
//   float4 with __stcs streaming stores  1.634-1.644 ms
//   float4 grid-stride, 512-1024 blocks  1.653-1.671 ms
//
// Nothing beats the plain scalar one-element-per-thread copy: a 16-byte load
// per thread saves instructions that are not the bottleneck, and the memory
// system is.  So this file keeps exactly that, with __restrict__/__ldg on the
// source.  There is no alignment or tail special case to add -- scalar
// accesses are valid for any pointer and any N*N.

#ifndef THREADS
#define THREADS 256
#endif

__global__ void copy_matrix_kernel(const float* __restrict__ A,
                                   float* __restrict__ B, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) B[i] = __ldg(&A[i]);
}

// A, B are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, float* B, int N) {
  if (N <= 0) return;
  const int total = N * N;  // N <= 4096, so this fits in int
  const int blocks = (total + THREADS - 1) / THREADS;
  copy_matrix_kernel<<<blocks, THREADS>>>(A, B, total);
  cudaDeviceSynchronize();
}
