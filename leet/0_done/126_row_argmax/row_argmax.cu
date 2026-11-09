// Row-wise argmax with smallest-index tie-breaking (challenge 126).
//
// output[b] = argmax_v logits[b, v], ties resolved to the smallest index.
// Benchmark: B = 256, V = 128,000 -> 131 MB of reads, so the kernel is
// read-bound (~1.6 ms at this GPU's ~82 GB/s).
//
// The starter measures 1.68 ms (78.0 GB/s) -- it is already essentially at
// the wall, with one block per row, a grid-stride scan and warp/block
// reductions of (value, index) pairs.  This version only tightens the load
// path:
//
//  * float4 loads: each thread scans 4 consecutive logits per load and
//    folds them into its local (value, index) pair in index order, so the
//    tie rule ("first index wins") is preserved within a chunk as well.
//  * The same merge rule in the warp/block reductions (strictly greater, or
//    equal with a smaller index).
//  * The scan uses a running pair in registers; V % 4 == 0 rows are aligned
//    for float4 (row starts are b*V floats), and a scalar fallback covers
//    other V.
//
// Measured: see README (about 1.6 ms -- the starter was already at the
// memory wall, so the gain is small).

#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int TPB = 512;

struct MaxPair {
  float val;
  int idx;
};

__device__ __forceinline__ MaxPair merge_pairs(MaxPair a, MaxPair b) {
  if (b.val > a.val) return b;
  if (b.val == a.val && b.idx < a.idx) return b;
  return a;
}

__device__ __forceinline__ MaxPair warp_reduce(MaxPair p) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    MaxPair o;
    o.val = __shfl_down_sync(0xffffffffu, p.val, off);
    o.idx = __shfl_down_sync(0xffffffffu, p.idx, off);
    p = merge_pairs(p, o);
  }
  return p;
}

__global__ void argmax_vec4_kernel(const float4* __restrict__ logits,
                                   int* __restrict__ output, int V4) {
  __shared__ MaxPair shared[TPB / 32];
  const int b = blockIdx.x;
  const float4* row = logits + (size_t)b * V4;

  MaxPair best = {-3.40282347e+38f, -1};
  for (int i = threadIdx.x; i < V4; i += TPB) {
    const float4 v = row[i];
    const int base = i * 4;
    // Fold in ascending index order so ties keep the first occurrence.
    if (v.x > best.val) { best.val = v.x; best.idx = base; }
    if (v.y > best.val) { best.val = v.y; best.idx = base + 1; }
    if (v.z > best.val) { best.val = v.z; best.idx = base + 2; }
    if (v.w > best.val) { best.val = v.w; best.idx = base + 3; }
  }
  best = warp_reduce(best);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) shared[warp] = best;
  __syncthreads();
  if (warp == 0) {
    MaxPair p = (lane < TPB / 32) ? shared[lane]
                                  : MaxPair{-3.40282347e+38f, -1};
    p = warp_reduce(p);
    if (lane == 0) output[b] = p.idx;
  }
}

__global__ void argmax_scalar_kernel(const float* __restrict__ logits,
                                     int* __restrict__ output, int V) {
  __shared__ MaxPair shared[TPB / 32];
  const int b = blockIdx.x;
  const float* row = logits + (size_t)b * V;

  MaxPair best = {-3.40282347e+38f, -1};
  for (int v = threadIdx.x; v < V; v += TPB) {
    const float x = row[v];
    if (x > best.val) {
      best.val = x;
      best.idx = v;
    }
  }
  best = warp_reduce(best);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) shared[warp] = best;
  __syncthreads();
  if (warp == 0) {
    MaxPair p = (lane < TPB / 32) ? shared[lane]
                                  : MaxPair{-3.40282347e+38f, -1};
    p = warp_reduce(p);
    if (lane == 0) output[b] = p.idx;
  }
}

}  // namespace

// logits, output are device pointers.
extern "C" void solve(const float* logits, int* output, int B, int V) {
  if (B <= 0 || V <= 0) return;
  if ((V & 3) == 0 && ((reinterpret_cast<uintptr_t>(logits) & 15) == 0)) {
    argmax_vec4_kernel<<<B, TPB>>>(reinterpret_cast<const float4*>(logits),
                                   output, V >> 2);
  } else {
    argmax_scalar_kernel<<<B, TPB>>>(logits, output, V);
  }
}
