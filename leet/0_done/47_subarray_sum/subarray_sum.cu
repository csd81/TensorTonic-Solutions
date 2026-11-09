#include <cuda_runtime.h>

#include <cstdint>

// Sum of input[S..E] (inclusive), int32 values.
//
// At the benchmark (N = 100,000,000, full range) the kernel reads 400 MB, so
// it is read-bound: a pure int4 read of the same array measures ~4.65 ms
// (86 GB/s).  The starter's scalar grid-stride kernel measures 5.09 ms
// (78.5 GB/s) because it issues one 4-byte load per thread per step.
//
// This version uses int4 loads for the aligned body of the range (four
// values per instruction) and handles the at-most-3 unaligned elements at
// each end scalar.  The reduction is a warp shuffle plus one atomicAdd per
// block.

#ifndef MAX_BLOCKS
#define MAX_BLOCKS 4096
#endif

namespace {

__global__ void sum_vec_kernel(const int* __restrict__ input,
                               int* __restrict__ output, int i0, int i1, int S,
                               int E) {
  const int4* v = reinterpret_cast<const int4*>(input);
  const int n4 = (i1 - i0) >> 2;
  const int stride = gridDim.x * blockDim.x;

  int sum = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
    const int4 a = __ldg(&v[(i0 >> 2) + i]);
    sum += a.x + a.y + a.z + a.w;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    for (int i = S; i < i0; ++i) sum += __ldg(&input[i]);
    for (int i = i1; i <= E; ++i) sum += __ldg(&input[i]);
  }

#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    sum += __shfl_down_sync(0xffffffffu, sum, off);
  __shared__ int warp_sums[8];
  if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    int s = 0;
    for (int i = 0; i < (int)(blockDim.x >> 5); ++i) s += warp_sums[i];
    if (s != 0) atomicAdd(output, s);
  }
}

}  // namespace

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int S, int E) {
  cudaMemsetAsync(output, 0, sizeof(int));
  if (N <= 0 || S > E || S < 0) return;
  if (E >= N) E = N - 1;

  // Aligned body [i0, i1): i0 is the first multiple of 4 >= S, and
  // i1 - i0 is a multiple of 4.  At most 3 scalar elements remain at each end.
  int i0 = S;
  while (i0 <= E && (i0 & 3) != 0) ++i0;
  int i1 = E + 1;
  while (i1 > i0 && ((i1 - i0) & 3) != 0) --i1;

  const int n4 = (i1 - i0) >> 2;
  int blocks = (n4 + 255) / 256;
  if (blocks > MAX_BLOCKS) blocks = MAX_BLOCKS;
  if (blocks < 1) blocks = 1;
  sum_vec_kernel<<<blocks, 256>>>(input, output, i0, i1, S, E);
  cudaDeviceSynchronize();
}
