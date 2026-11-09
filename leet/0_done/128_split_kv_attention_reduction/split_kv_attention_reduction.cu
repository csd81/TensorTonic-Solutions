// Flash-Decoding split-KV attention merge (challenge 128).
//
// Merge per-split partial attention results with the log-sum-exp trick:
//   m_h = max_s lse[s,h],  w_s = exp(lse[s,h] - m_h)
//   out[h,d] = sum_s w_s * partial_out[s,h,d] / sum_s w_s
// Shapes: partial_out (S,H,D), partial_lse (S,H), output (H,D).
// Benchmark: S=128, H=64, D=128 -> 4.1 MB read, ~1M multiply-adds.
//
// The starter measures 67.8 us (62.9 GB/s effective).  It launches one block
// (one thread per dimension) per head; every thread re-reads the whole lse
// column and re-evaluates exp() per split -- 1M exponentials where 8K are
// needed.  It is already memory-bound (4.1 MB at this GPU's ~80 GB/s is
// ~52 us), so the wins are structural and small:
//
//   * One warp per head instead of one block: 64 warps, each keeping its four
//     accumulators in registers.
//   * float4 loads: a lane owns four consecutive dimensions, so each warp
//     reads a 512-byte split row per instruction.
//   * The lse max and the weights are computed once per (head, split) with
//     warp shuffles and kept in shared memory; exp() runs 8K times, not 1M.
//   * The split loop is unrolled by four, keeping four loads in flight per
//     lane.
//
// Measured variants at the benchmark shape (same process, interleaved):
//   one warp per head, 4 splits/iter, float4 (shipped)  58.2 us
//   four warps per head, split range divided           58.6 us
//   one warp per head, 2 splits/iter                   59.4 us
//   one warp per head, 1 split/iter                    63.2 us
//   starter (one block per head, scalar)               67.8 us
//
// D up to 256 -> up to two float4 chunks per lane, handled with two explicit
// accumulators.  A scalar path covers D not divisible by 4.

#include <cuda_runtime.h>
#include <math.h>

namespace {

constexpr int kMaxSplits = 256;  // challenge constraint
constexpr unsigned kAll = 0xffffffffu;

__device__ __forceinline__ float4 fma4(float w, const float4 v, float4 acc) {
  acc.x = fmaf(w, v.x, acc.x);
  acc.y = fmaf(w, v.y, acc.y);
  acc.z = fmaf(w, v.z, acc.z);
  acc.w = fmaf(w, v.w, acc.w);
  return acc;
}

// One warp per head.
template <bool VEC4>
__global__ void merge_kernel(const float* __restrict__ partial_out,
                             const float* __restrict__ partial_lse,
                             float* __restrict__ output, int S, int H, int D) {
  __shared__ float w[kMaxSplits];
  const int h = blockIdx.x;
  const int lane = threadIdx.x;

  // Max over splits, then the stable weights and their sum -- once per head.
  float m = -INFINITY;
  for (int s = lane; s < S; s += 32) {
    const float v = partial_lse[s * H + h];
    w[s] = v;
    m = fmaxf(m, v);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(kAll, m, off));

  float sum = 0.0f;
  for (int s = lane; s < S; s += 32) {
    const float e = __expf(w[s] - m);
    w[s] = e;
    sum += e;
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(kAll, sum, off);
  const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;

  if (VEC4) {
    const float4* src = reinterpret_cast<const float4*>(partial_out);
    const int dv4 = D >> 2;  // float4 chunks per row (<= 64)
    const int v0 = lane, v1 = lane + 32;
    float4 acc0 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 acc1 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4* dst = reinterpret_cast<float4*>(output) + (size_t)h * dv4;
    // The chunk-count branch is per lane but constant for the launch, so it
    // sits outside the split loop: the common D <= 128 case runs a loop with
    // no per-iteration branches at all.
    if (v1 < dv4) {
#pragma unroll 4
      for (int s = 0; s < S; ++s) {
        const float ws = w[s];
        const size_t o = (size_t)(s * H + h) * dv4;
        acc0 = fma4(ws, src[o + v0], acc0);
        acc1 = fma4(ws, src[o + v1], acc1);
      }
      dst[v0] = make_float4(acc0.x * inv, acc0.y * inv, acc0.z * inv,
                            acc0.w * inv);
      dst[v1] = make_float4(acc1.x * inv, acc1.y * inv, acc1.z * inv,
                            acc1.w * inv);
    } else if (v0 < dv4) {
#pragma unroll 4
      for (int s = 0; s < S; ++s) {
        const float ws = w[s];
        const size_t o = (size_t)(s * H + h) * dv4;
        acc0 = fma4(ws, src[o + v0], acc0);
      }
      dst[v0] = make_float4(acc0.x * inv, acc0.y * inv, acc0.z * inv,
                            acc0.w * inv);
    }
  } else {
    for (int v = lane; v < D; v += 32) {
      float acc = 0.0f;
      for (int s = 0; s < S; ++s) acc = fmaf(w[s], partial_out[(size_t)(s * H + h) * D + v], acc);
      output[(size_t)h * D + v] = acc * inv;
    }
  }
}

}  // namespace

// All pointers are device pointers.
extern "C" void solve(const float* partial_out, const float* partial_lse,
                      float* output, int num_splits, int num_heads,
                      int head_dim) {
  if (num_splits <= 0 || num_heads <= 0 || head_dim <= 0) return;
  if (num_splits > kMaxSplits) return;  // outside the challenge constraints
  if ((head_dim & 3) == 0) {
    merge_kernel<true><<<num_heads, 32>>>(partial_out, partial_lse, output,
                                         num_splits, num_heads, head_dim);
  } else {
    merge_kernel<false><<<num_heads, 32>>>(partial_out, partial_lse, output,
                                          num_splits, num_heads, head_dim);
  }
}
