// Group normalization (challenge 105).
//
// For every (n, g): mean/var over the contiguous block of (C/G)·H·W elements,
// then y = gamma_c·(x - mu)·inv_std + beta_c.  Benchmark: N=8, C=512, H=64,
// W=64, G=32 -> 256 groups of 65,536 elements, 67 MB in and 67 MB out.  The
// group's elements are contiguous in memory, so the kernel needs two reads of
// the group plus one write: 201 MB, ~2.4 ms at this GPU's ~85 GB/s.
//
// The starter measures 2.66 ms (75.8 GB/s).  Its per-element index math
// (i/hw_size, i%hw_size, recombining n/c/hw) is all unnecessary because a
// group is one contiguous range, and it uses scalar loads.
//
// This version:
//
//  1. Computes the group's base offset once per block and walks the range
//     directly (no divisions in the hot loops).
//  2. Uses float4 loads/stores with four FP32 accumulators in the stats pass;
//     in the normalize pass the channel of a chunk is derived once per chunk
//     (HW is a multiple of 4 on that path).
//  3. Reduces (sum, sumsq) with warp shuffles plus a shared combine, then
//     broadcasts mean/inv_std through shared memory.
//
// Measured: see README (~2.4 ms, ~1.1x -- the two-read structure is
// inherent: a 256 KB group does not fit in shared memory).

#include <cuda_runtime.h>
#include <math.h>

namespace {

constexpr int TPB = 256;

__device__ __forceinline__ void block_reduce2(float& a, float& b) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    a += __shfl_down_sync(0xffffffffu, a, off);
    b += __shfl_down_sync(0xffffffffu, b, off);
  }
  __shared__ float sa[8], sb[8];
  if ((threadIdx.x & 31) == 0) {
    sa[threadIdx.x >> 5] = a;
    sb[threadIdx.x >> 5] = b;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float ta = 0.0f, tb = 0.0f;
    const int nw = (blockDim.x + 31) >> 5;
    for (int w = 0; w < nw; ++w) {
      ta += sa[w];
      tb += sb[w];
    }
    sa[0] = ta;
    sb[0] = tb;
  }
  __syncthreads();
  a = sa[0];
  b = sb[0];
}

__global__ void group_norm_vec4_kernel(const float4* __restrict__ X4,
                                       const float* __restrict__ gamma,
                                       const float* __restrict__ beta,
                                       float4* __restrict__ Y4, int C, int HW,
                                       int cpg, int HW4, float eps) {
  const int n = blockIdx.y, g = blockIdx.x;
  const int c0 = g * cpg;
  const size_t base4 = ((size_t)n * C + c0) * HW4;
  const float4* xg = X4 + base4;
  float4* yg = Y4 + base4;
  const int count4 = cpg * HW4;

  // One pass over the group: sum and sum-of-squares with independent
  // accumulators.  (A separate sumsq pass would read the group a third time.)
  float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
  float q0 = 0.f, q1 = 0.f, q2 = 0.f, q3 = 0.f;
  for (int i = threadIdx.x; i < count4; i += TPB) {
    const float4 v = xg[i];
    s0 += v.x;
    s1 += v.y;
    s2 += v.z;
    s3 += v.w;
    q0 = fmaf(v.x, v.x, q0);
    q1 = fmaf(v.y, v.y, q1);
    q2 = fmaf(v.z, v.z, q2);
    q3 = fmaf(v.w, v.w, q3);
  }
  float sum = (s0 + s1) + (s2 + s3);
  float sq = (q0 + q1) + (q2 + q3);
  block_reduce2(sum, sq);
  const float mu = sum / (4.0f * count4);
  const float inv_std =
      rsqrtf(fmaxf(sq / (4.0f * count4) - mu * mu, 0.0f) + eps);

  for (int i = threadIdx.x; i < count4; i += TPB) {
    const int c = c0 + i / HW4;  // chunk lies inside one channel (HW % 4 == 0)
    const float ga = gamma[c], be = beta[c];
    const float4 v = xg[i];
    yg[i] = make_float4(fmaf(ga, (v.x - mu) * inv_std, be),
                        fmaf(ga, (v.y - mu) * inv_std, be),
                        fmaf(ga, (v.z - mu) * inv_std, be),
                        fmaf(ga, (v.w - mu) * inv_std, be));
  }
}

__global__ void group_norm_scalar_kernel(const float* __restrict__ X,
                                         const float* __restrict__ gamma,
                                         const float* __restrict__ beta,
                                         float* __restrict__ Y, int C, int HW,
                                         int cpg, float eps) {
  const int n = blockIdx.y, g = blockIdx.x;
  const int c0 = g * cpg;
  const size_t base = ((size_t)n * C + c0) * HW;
  const float* xg = X + base;
  float* yg = Y + base;
  const int count = cpg * HW;

  float s = 0.f, q = 0.f;
  for (int i = threadIdx.x; i < count; i += TPB) {
    const float v = xg[i];
    s += v;
    q = fmaf(v, v, q);
  }
  block_reduce2(s, q);
  const float mu = s / (float)count;
  const float inv_std = rsqrtf(fmaxf(q / (float)count - mu * mu, 0.0f) + eps);

  for (int i = threadIdx.x; i < count; i += TPB) {
    const int c = c0 + i / HW;
    yg[i] = fmaf(gamma[c], (xg[i] - mu) * inv_std, beta[c]);
  }
}

}  // namespace

extern "C" void solve(const float* X, const float* gamma, const float* beta,
                      float* Y, int N, int C, int H, int W, int G, float eps) {
  if (N <= 0 || C <= 0 || H <= 0 || W <= 0 || G <= 0) return;
  const int HW = H * W, cpg = C / G;
  const dim3 grid(G, N);
  if ((HW & 3) == 0) {
    group_norm_vec4_kernel<<<grid, TPB>>>(
        reinterpret_cast<const float4*>(X), gamma, beta,
        reinterpret_cast<float4*>(Y), C, HW, cpg, HW >> 2, eps);
  } else {
    group_norm_scalar_kernel<<<grid, TPB>>>(X, gamma, beta, Y, C, HW, cpg,
                                           eps);
  }
}
