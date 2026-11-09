#include <cuda_runtime.h>

#include <cstdint>

// Count elements equal to P in an N x M x K int array.
//
// At the benchmark (500^3 = 125,000,000 elements) the kernel reads 500 MB and
// writes 4 bytes, so it is purely read-bound: a plain int4 read-over-the-
// array measures 5.78 ms (86.5 GB/s), and the starter's scalar grid-stride
// kernel already measures 5.79 ms — at the floor.
//
// This version still cleans up the mechanics: int4 loads (four comparisons
// per load instead of one), a warp/block reduction so there is one atomicAdd
// per block rather than one per thread, and a scalar tail for totals that are
// not a multiple of 4.  It measures 5.78-5.96 ms, the same as the starter
// within noise, as expected for a kernel that was already at the memory wall.

namespace {

__global__ void count_vec_kernel(const int4* __restrict__ in,
                                 int* __restrict__ out, long n4, long total,
                                 int P) {
  int cnt = 0;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
       i += (long)gridDim.x * blockDim.x) {
    const int4 v = __ldg(&in[i]);
    cnt += (v.x == P) + (v.y == P) + (v.z == P) + (v.w == P);
  }
  // Scalar tail: the last total % 4 elements.
  const int* in_s = reinterpret_cast<const int*>(in);
  for (long i = n4 * 4 + (long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += (long)gridDim.x * blockDim.x)
    cnt += (__ldg(&in_s[i]) == P);

  for (int off = 16; off > 0; off >>= 1)
    cnt += __shfl_down_sync(0xffffffffu, cnt, off);
  __shared__ int warp_sums[8];
  if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = cnt;
  __syncthreads();
  if (threadIdx.x == 0) {
    int s = 0;
    for (int i = 0; i < (int)(blockDim.x >> 5); ++i) s += warp_sums[i];
    if (s != 0) atomicAdd(out, s);
  }
}

}  // namespace

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int M, int K,
                      int P) {
  const long total = (long)N * M * K;
  if (total <= 0) return;
  cudaMemsetAsync(output, 0, sizeof(int));

  const long n4 = total / 4;
  const int threads = 256;
  long want = (n4 + threads - 1) / threads;
  int blocks = (int)(want > 1024 ? 1024 : (want < 1 ? 1 : want));
  count_vec_kernel<<<blocks, threads>>>(reinterpret_cast<const int4*>(input),
                                        output, n4, total, P);
  cudaDeviceSynchronize();
}
