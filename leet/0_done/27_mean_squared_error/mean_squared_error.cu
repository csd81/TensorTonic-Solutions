// Mean squared error (challenge 27).
//
// mse = (1/N) * sum_i (predictions[i] - targets[i])^2.  N <= 1e8;
// benchmark N = 50,000,000 -> 400 MB of reads, so the kernel is read-bound
// (the floor at this GPU's ~87 GB/s is ~4.6 ms).
//
// The starter measures 5.67 ms (70.6 GB/s).  It does all arithmetic in
// *double*: on sm_61 FP64 runs at 1/32 of the FP32 rate, so the 250 M
// double operations (two converts, a subtract, a multiply and an add per
// element) cost more than the memory stream.
//
// This version:
//
//  1. Loads 4 floats per input per iteration as float4 (a warp request is
//     four 512-byte transactions) and accumulates the squared differences in
//     FP32 with four independent accumulators.
//  2. Reduces each block with warp shuffles into one FP32 partial per block.
//  3. Sums the (<= 2048) block partials in double in the final kernel and
//     divides by N -- FP64 is used only 2048 times per call, where its rate
//     is irrelevant but its precision makes the final sum exact enough.
//
// Precision: each thread sums ~N/(blocks*TPB) = ~190 terms in FP32 and each
// block combines 256 of those, so the accumulated relative error stays around
// 1e-6 -- two orders below the 1e-4 relative tolerance used by the harness
// (measured error at the benchmark shape: 6.7e-3 absolute on 3.3e5).
//
// Measured: see README (about 4.7 ms = 85 GB/s, 1.2x).

#include <cuda_runtime.h>

namespace {

constexpr int TPB = 256;
constexpr int MAX_BLOCKS = 2048;

__device__ __forceinline__ float warp_reduce(float s) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
  return s;
}

__global__ void mse_kernel(const float4* __restrict__ p4,
                           const float4* __restrict__ t4,
                           const float* __restrict__ p,
                           const float* __restrict__ t,
                           float* __restrict__ block_sums, int n4, int N) {
  float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
    const float4 a = p4[i];
    const float4 b = t4[i];
    const float d0 = a.x - b.x;
    const float d1 = a.y - b.y;
    const float d2 = a.z - b.z;
    const float d3 = a.w - b.w;
    s0 = fmaf(d0, d0, s0);
    s1 = fmaf(d1, d1, s1);
    s2 = fmaf(d2, d2, s2);
    s3 = fmaf(d3, d3, s3);
  }
  // Tail (N mod 4 <= 3 elements).
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    for (int i = n4 * 4; i < N; ++i) {
      const float d = p[i] - t[i];
      s0 = fmaf(d, d, s0);
    }
  }

  float s = (s0 + s1) + (s2 + s3);
  __shared__ float warp_sums[32];
  s = warp_reduce(s);
  if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = s;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
    const int nw = (blockDim.x + 31) >> 5;
    for (int w = 0; w < nw; ++w) total += warp_sums[w];
    block_sums[blockIdx.x] = total;
  }
}

__global__ void final_kernel(const float* __restrict__ block_sums, int nb,
                             float* __restrict__ mse, int N) {
  double s = 0.0;
  for (int i = threadIdx.x; i < nb; i += blockDim.x) s += block_sums[i];
  __shared__ double shared[256];
  shared[threadIdx.x] = s;
  __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (threadIdx.x < off) shared[threadIdx.x] += shared[threadIdx.x + off];
    __syncthreads();
  }
  if (threadIdx.x == 0) *mse = (float)(shared[0] / (double)N);
}

// Cached block-sum buffer and accumulator.
float* g_block_sums = nullptr;
int g_cap = 0;

}  // namespace

// predictions, targets, mse are device pointers.
extern "C" void solve(const float* predictions, const float* targets, float* mse,
                      int N) {
  if (N <= 0) return;
  if (MAX_BLOCKS > g_cap) {
    if (g_block_sums) cudaFree(g_block_sums);
    cudaMalloc(&g_block_sums, (size_t)MAX_BLOCKS * sizeof(float));
    g_cap = MAX_BLOCKS;
  }

  const int n4 = N >> 2;
  int blocks = (n4 + TPB - 1) / TPB;
  if (blocks > MAX_BLOCKS) blocks = MAX_BLOCKS;
  if (blocks < 1) blocks = 1;

  mse_kernel<<<blocks, TPB>>>(reinterpret_cast<const float4*>(predictions),
                              reinterpret_cast<const float4*>(targets),
                              predictions, targets, g_block_sums, n4, N);
  final_kernel<<<1, 256>>>(g_block_sums, blocks, mse, N);
}
