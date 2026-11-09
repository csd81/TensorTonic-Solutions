// Layer normalization for a 2D [N, C] tensor (challenge 113).
//
// y[i,j] = weight[j] * (x[i,j] - mu_i) / sqrt(var_i + eps) + bias[j]
// with per-row statistics over the C features.  Benchmark: N = 65,536,
// C = 512 -> 134 MB in + 134 MB out; reading the row once and writing it once
// is 268 MB, ~3.2 ms at this GPU's ~85 GB/s.
//
// The starter measures 6.3 ms: it reads every row three times (mean pass,
// variance pass, normalize pass), i.e. 536 MB, with scalar loads.
//
// This version keeps the row in registers:
//
//  1. One block per row; blockDim is chosen so each thread owns exactly one
//     float4 chunk (C/4 <= 1024 for C <= 4096), loaded once into registers.
//  2. Two block reductions over those registers: sum -> mean, then the sum of
//     (x - mean)^2 -> variance.  Computing the variance from the register
//     values (rather than from sumsq - mean^2) keeps it accurate for rows
//     whose mean is large relative to their spread.
//  3. Normalize and write from the same registers (weight/bias as float4).
//
// A scalar path covers C not divisible by 4 using the same register scheme
// up to C = 1024 and re-reading (L1) beyond that.
//
// Measured: see README (~3.3 ms, close to the wall).

#include <cuda_runtime.h>
#include <math.h>

namespace {

// Reduces one float across the block and broadcasts it; `scratch` must be
// shared memory of at least 32 floats.
__device__ __forceinline__ float block_reduce_bcast(float v, float* scratch,
                                                    int nwarp) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  if ((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = v;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.0f;
    for (int w = 0; w < nwarp; ++w) t += scratch[w];
    scratch[0] = t;
  }
  __syncthreads();
  const float result = scratch[0];
  // Barrier after the read: the caller may immediately run the next
  // reduction, whose warp leaders would otherwise overwrite scratch[0] while
  // slower threads are still reading this result.
  __syncthreads();
  return result;
}

__global__ void ln_vec4_kernel(const float4* __restrict__ x4,
                               const float4* __restrict__ w4,
                               const float4* __restrict__ b4,
                               float4* __restrict__ y4, int C4, float C,
                               float eps) {
  __shared__ float scratch[32];
  const int i = blockIdx.x;
  const float4* row = x4 + (size_t)i * C4;
  const int nwarp = (blockDim.x + 31) >> 5;

  float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
  const bool active = threadIdx.x < C4;
  if (active) s = row[threadIdx.x];

  float sum = (s.x + s.y) + (s.z + s.w);
  const float mu = block_reduce_bcast(sum, scratch, nwarp) / C;

  float d0 = s.x - mu, d1 = s.y - mu, d2 = s.z - mu, d3 = s.w - mu;
  // Inactive threads (threadIdx.x >= C4) must contribute nothing: their zero
  // registers would otherwise add 4*mu^2 to the variance.
  float sq = active ? ((d0 * d0 + d1 * d1) + (d2 * d2 + d3 * d3)) : 0.0f;
  const float var = block_reduce_bcast(sq, scratch, nwarp) / C;
  const float inv_std = rsqrtf(var + eps);

  if (active) {
    const float4 g = w4[threadIdx.x];
    const float4 bb = b4[threadIdx.x];
    y4[(size_t)i * C4 + threadIdx.x] =
        make_float4(fmaf(g.x, d0 * inv_std, bb.x), fmaf(g.y, d1 * inv_std, bb.y),
                    fmaf(g.z, d2 * inv_std, bb.z), fmaf(g.w, d3 * inv_std, bb.w));
  }
}

// C % 4 != 0.  Registers hold one element per thread when C <= 1024 (the
// common case); larger C re-reads the row for the normalize pass.
__global__ void ln_scalar_kernel(const float* __restrict__ x,
                                 const float* __restrict__ w,
                                 const float* __restrict__ b,
                                 float* __restrict__ y, int C, float eps) {
  __shared__ float scratch[32];
  const int i = blockIdx.x;
  const float* row = x + (size_t)i * C;
  const int nwarp = (blockDim.x + 31) >> 5;

  const int j = threadIdx.x;
  const bool single = (blockDim.x >= C);  // one element per thread
  float v = (single && j < C) ? row[j] : 0.0f;

  float sum = 0.0f;
  if (single) {
    sum = (j < C) ? v : 0.0f;
  } else {
    for (int t = j; t < C; t += blockDim.x) sum += row[t];
  }
  const float mu = block_reduce_bcast(sum, scratch, nwarp) / (float)C;

  float sq = 0.0f;
  if (single) {
    if (j < C) {
      const float d = v - mu;
      sq = d * d;
    }
  } else {
    for (int t = j; t < C; t += blockDim.x) {
      const float d = row[t] - mu;
      sq = fmaf(d, d, sq);
    }
  }
  const float var = block_reduce_bcast(sq, scratch, nwarp) / (float)C;
  const float inv_std = rsqrtf(var + eps);

  if (single) {
    if (j < C) y[(size_t)i * C + j] = fmaf(w[j], (v - mu) * inv_std, b[j]);
  } else {
    for (int t = j; t < C; t += blockDim.x)
      y[(size_t)i * C + t] = fmaf(w[t], (row[t] - mu) * inv_std, b[t]);
  }
}

int warp_round(int n) {
  int t = ((n + 31) / 32) * 32;
  if (t < 32) t = 32;
  if (t > 1024) t = 1024;
  return t;
}

}  // namespace

extern "C" void solve(const float* input, const float* weight, const float* bias,
                      float* output, int N, int C, float eps) {
  if (N <= 0 || C <= 0) return;
  if ((C & 3) == 0) {
    const int threads = warp_round(C >> 2);
    ln_vec4_kernel<<<N, threads>>>(reinterpret_cast<const float4*>(input),
                                   reinterpret_cast<const float4*>(weight),
                                   reinterpret_cast<const float4*>(bias),
                                   reinterpret_cast<float4*>(output), C >> 2,
                                   (float)C, eps);
  } else {
    const int threads = warp_round(C);
    ln_scalar_kernel<<<N, threads>>>(input, weight, bias, output, C, eps);
  }
}
