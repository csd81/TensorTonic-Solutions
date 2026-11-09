#include <cuda_runtime.h>

#include <cstdint>

// R rounds of FNV-1a over each 32-bit element: out[i] = fnv1a^R(in[i]).
//
// The memory traffic is 40 MB at the benchmark size (N = 5,000,000), so a
// pure copy is ~0.49 ms.  What dominates is R:
//
//   R = 1   0.49 ms  -- memory-bound, at the copy floor
//   R = 10  0.90 ms  -- compute-bound
//   R = 100 9.24 ms  -- issue/throughput-bound (2e9 dependent multiplies)
//
// Each hash is a chain of four dependent 32-bit multiplies (xor a byte into
// the running value, multiply by the FNV prime), so one element per thread
// leaves the multiply pipeline waiting on its own dependency chain.  This
// version gives each thread one int4 (four independent elements) and
// interleaves the four chains, which the scheduler can overlap.  That is
// worth ~15% at R = 10 and ~3% at R = 100, where total instruction issue,
// not latency, is the limit.  Wider vector loads would only reduce an
// already-negligible memory instruction count.
//
// The starter's scalar shape is correct but leaves that ILP on the table.

#ifndef THREADS
#define THREADS 512
#endif

__device__ __forceinline__ unsigned int fnv1a(unsigned int x) {
  const unsigned int P = 16777619u;
  unsigned int h = 2166136261u;
  h = (h ^ (x & 0xFFu)) * P;
  h = (h ^ ((x >> 8) & 0xFFu)) * P;
  h = (h ^ ((x >> 16) & 0xFFu)) * P;
  h = (h ^ (x >> 24)) * P;
  return h;
}

__global__ void rainbow_vec_kernel(const int4* __restrict__ input4,
                                   uint4* __restrict__ output4, int n4, int R) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) {
    const int4 v = __ldg(&input4[i]);
    unsigned int a = static_cast<unsigned int>(v.x);
    unsigned int b = static_cast<unsigned int>(v.y);
    unsigned int c = static_cast<unsigned int>(v.z);
    unsigned int d = static_cast<unsigned int>(v.w);
    for (int r = 0; r < R; ++r) {
      a = fnv1a(a);
      b = fnv1a(b);
      c = fnv1a(c);
      d = fnv1a(d);
    }
    output4[i] = make_uint4(a, b, c, d);
  }
}

// Scalar tail for the last N % 4 elements.  Kept out of the hot kernel: the
// extra loop in the same kernel costs ~15% at R = 10 (register pressure),
// even though only one thread ever executes it.
__global__ void rainbow_tail_kernel(const int* __restrict__ input,
                                    unsigned int* __restrict__ output, int start,
                                    int N, int R) {
  const int k = start + threadIdx.x;
  if (k < N) {
    unsigned int v = static_cast<unsigned int>(__ldg(&input[k]));
    for (int r = 0; r < R; ++r) v = fnv1a(v);
    output[k] = v;
  }
}

// Fallback for pointers that are not 16-byte aligned (one element per thread).
__global__ void rainbow_scalar_kernel(const int* __restrict__ input,
                                      unsigned int* __restrict__ output, int N,
                                      int R) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    unsigned int v = static_cast<unsigned int>(__ldg(&input[i]));
    for (int r = 0; r < R; ++r) v = fnv1a(v);
    output[i] = v;
  }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, unsigned int* output, int N, int R) {
  if (N <= 0) return;

  const bool aligned =
      ((reinterpret_cast<uintptr_t>(input) | reinterpret_cast<uintptr_t>(output)) &
       15u) == 0;
  if (aligned) {
    const int n4 = N >> 2;
    if (n4 > 0) {
      const int blocks = (n4 + THREADS - 1) / THREADS;
      rainbow_vec_kernel<<<blocks, THREADS>>>(
          reinterpret_cast<const int4*>(input),
          reinterpret_cast<uint4*>(output), n4, R);
    }
    const int tail_start = 4 * n4;
    if (tail_start < N) {
      rainbow_tail_kernel<<<1, 4>>>(input, output, tail_start, N, R);
    }
  } else {
    const int blocks = (N + THREADS - 1) / THREADS;
    rainbow_scalar_kernel<<<blocks, THREADS>>>(input, output, N, R);
  }
  cudaDeviceSynchronize();
}
