// Token + position embedding lookup with layer normalization (challenge 106).
//
// For every (b, t): s = E_T[token_ids[b,t]] + E_P[position_ids[t]], then
// y = gamma * (s - mu) / sqrt(var + eps) + beta over the D dimensions.
// Benchmark: B=32, T=512, V=30000, P=2048, D=768 -> 16,384 rows; two row
// reads + one row write of D floats each = 150 MB of DRAM traffic, so the
// floor is ~1.8 ms at this GPU's ~85 GB/s.
//
// The starter measures 2.33 ms (64.8 GB/s): one block per (b, t) with 256
// threads, scalar loads, a shared-memory tree reduction, and a second pass
// over both rows for the normalize step (the second pass mostly hits L1, but
// the scalar loads and the tree reduction keep it off the wall).
//
// This version:
//
//  1. float4 rows: each thread loads one 16-byte chunk of the token row and
//     of the position row, forms s = token + position in registers, and
//     keeps s there -- the normalize pass never re-reads memory.
//  2. Stats reduction: warp shuffles for (sum, sumsq) plus a tiny
//     shared-memory combine of the warp results.
//  3. One pass writes the normalized row (gamma/beta loaded as float4), so
//     the kernel reads each row exactly once and writes exactly once.
//
// A scalar fallback covers D not divisible by 4 (row bases would not be
// 16-byte aligned) and re-reads the rows for the normalize step (they are in
// L1 at that point).
//
// Measured: see README (~1.9 ms, 1.2x).

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

__global__ void emb_ln_vec4_kernel(const int* __restrict__ token_ids,
                                   const int* __restrict__ position_ids,
                                   const float4* __restrict__ te,
                                   const float4* __restrict__ pe,
                                   const float4* __restrict__ gamma4,
                                   const float4* __restrict__ beta4,
                                   float4* __restrict__ out, int T, int D4,
                                   float eps) {
  const int b = blockIdx.y, t = blockIdx.x;
  const int ti = token_ids[b * T + t];
  const int pi = position_ids[t];
  const float4* tr = te + (size_t)ti * D4;
  const float4* pr = pe + (size_t)pi * D4;
  float4* orow = out + ((size_t)b * T + t) * D4;

  // D <= 1024 gives D4 <= 256 = TPB, so each thread owns exactly one float4
  // chunk and keeps s in registers across the reduction: the row is read once
  // and written once.
  const int i = threadIdx.x;
  float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
  if (i < D4) {
    const float4 a = tr[i];
    const float4 bq = pr[i];
    s.x = a.x + bq.x;
    s.y = a.y + bq.y;
    s.z = a.z + bq.z;
    s.w = a.w + bq.w;
  }
  float sum = (s.x + s.y) + (s.z + s.w);
  float sq = (s.x * s.x + s.y * s.y) + (s.z * s.z + s.w * s.w);
  block_reduce2(sum, sq);
  const float mu = sum / (4.0f * D4);
  const float inv_std = rsqrtf(fmaxf(sq / (4.0f * D4) - mu * mu, 0.0f) + eps);

  if (i < D4) {
    const float4 g = gamma4[i];
    const float4 bb = beta4[i];
    orow[i] = make_float4(fmaf(g.x, (s.x - mu) * inv_std, bb.x),
                          fmaf(g.y, (s.y - mu) * inv_std, bb.y),
                          fmaf(g.z, (s.z - mu) * inv_std, bb.z),
                          fmaf(g.w, (s.w - mu) * inv_std, bb.w));
  }
}

__global__ void emb_ln_scalar_kernel(const int* __restrict__ token_ids,
                                     const int* __restrict__ position_ids,
                                     const float* __restrict__ te,
                                     const float* __restrict__ pe,
                                     const float* __restrict__ gamma,
                                     const float* __restrict__ beta,
                                     float* __restrict__ out, int T, int D,
                                     float eps) {
  const int b = blockIdx.y, t = blockIdx.x;
  const int ti = token_ids[b * T + t];
  const int pi = position_ids[t];
  const float* tr = te + (size_t)ti * D;
  const float* pr = pe + (size_t)pi * D;
  float* orow = out + ((size_t)b * T + t) * D;

  float sum = 0.0f, sq = 0.0f;
  for (int d = threadIdx.x; d < D; d += TPB) {
    const float v = tr[d] + pr[d];
    sum += v;
    sq = fmaf(v, v, sq);
  }
  block_reduce2(sum, sq);
  const float mu = sum / (float)D;
  const float inv_std = rsqrtf(fmaxf(sq / (float)D - mu * mu, 0.0f) + eps);
  for (int d = threadIdx.x; d < D; d += TPB) {
    const float v = tr[d] + pr[d];
    orow[d] = fmaf(gamma[d], (v - mu) * inv_std, beta[d]);
  }
}

}  // namespace

extern "C" void solve(const int* token_ids, const int* position_ids,
                      const float* token_embeddings,
                      const float* position_embeddings, const float* gamma,
                      const float* beta, float* output, int B, int T, int V,
                      int P, int D, float eps) {
  if (B <= 0 || T <= 0 || D <= 0) return;
  const dim3 grid(T, B);
  if ((D & 3) == 0) {
    emb_ln_vec4_kernel<<<grid, TPB>>>(
        token_ids, position_ids,
        reinterpret_cast<const float4*>(token_embeddings),
        reinterpret_cast<const float4*>(position_embeddings),
        reinterpret_cast<const float4*>(gamma),
        reinterpret_cast<const float4*>(beta),
        reinterpret_cast<float4*>(output), T, D >> 2, eps);
  } else {
    emb_ln_scalar_kernel<<<grid, TPB>>>(token_ids, position_ids,
                                        token_embeddings, position_embeddings,
                                        gamma, beta, output, T, D, eps);
  }
}
