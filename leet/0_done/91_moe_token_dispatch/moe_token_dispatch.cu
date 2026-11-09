// MoE token dispatch (challenge 91).
//
// Given x [T, D] and expert_idx [T], fill
//   dispatched_x[e, 0:token_counts[e], :] = the tokens routed to expert e,
//   in ascending token order,
// and token_counts[e] for every expert.  capacity = T.  Benchmark:
// T = 16,384, D = 512, E = 8 -> 64 MB of traffic (32 MB read + 32 MB
// written), so the copy floor is ~0.8 ms at this GPU's ~80 GB/s.
//
// The starter measures 8.39 ms (8.0 GB/s): its first kernel assigns slots
// from a single thread that walks all T tokens with a dependent global load
// per token (~7.5 ms of the total), then a second kernel copies one block per
// token.
//
// This version:
//
//  1. Assigns the stable slots with one warp per expert using the classic
//     ordered-compaction primitive: each iteration a warp reads 32
//     consecutive expert_idx values, `__ballot_sync(exp == e)` gives the mask
//     of lanes routed to the warp's expert, each matching lane's slot is
//     `running + popc(mask & lanes_before_it)`, and `running += popc(mask)`.
//     The mask is uniform across lanes, so the running count stays uniform
//     with no shuffles, atomics or synchronisation.  Token order within an
//     expert is preserved because iterations walk tokens in ascending order
//     and lanes are assigned ascending positions.  One kernel, no temporary
//     allocation beyond T ints (cached across calls).
//  2. Copies with one warp per token: each lane moves float4s (D = 512 is
//     four float4s per lane), so a token row is copied by four coalesced
//     load/store pairs.  A scalar path handles D not divisible by 4.
//
// Measured: see README (~1 ms, dominated by the 64 MB copy).

#include <cuda_runtime.h>

namespace {

constexpr int kWarp = 32;
constexpr unsigned kAll = 0xffffffffu;

// Stable slot assignment: one warp per expert, walking tokens in order.
__global__ void slot_kernel(const int* __restrict__ expert_idx,
                            int* __restrict__ token_slots,
                            int* __restrict__ token_counts, int T) {
  const int e = blockIdx.x;
  const int lane = threadIdx.x;  // blockDim.x == kWarp
  int running = 0;
  // The trip count must be uniform across the warp: __ballot_sync with
  // exited lanes is undefined.  Tail lanes of the last window load -1.
  const int windows = (T + kWarp - 1) / kWarp;
#pragma unroll 4
  for (int w = 0; w < windows; ++w) {
    const int i = w * kWarp + lane;
    const int exp = (i < T) ? expert_idx[i] : -1;
    const unsigned mask = __ballot_sync(kAll, exp == e);
    const int rank = __popc(mask & ((1u << lane) - 1u));
    if (exp == e) token_slots[i] = running + rank;
    running += __popc(mask);
  }
  if (lane == 0) token_counts[e] = running;
}

// Copy rows: one warp per token.
__global__ void dispatch_kernel(const float* __restrict__ x,
                                const int* __restrict__ expert_idx,
                                const int* __restrict__ token_slots,
                                float* __restrict__ dispatched_x, int T, int D,
                                int capacity) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= T) return;
  const int exp = expert_idx[warp];
  const int slot = token_slots[warp];
  const size_t src_base = (size_t)warp * D;
  const size_t dst_base = ((size_t)exp * capacity + slot) * D;
  if ((D & 3) == 0) {
    const int dv4 = D >> 2;
    const float4* src = reinterpret_cast<const float4*>(x + src_base);
    float4* dst = reinterpret_cast<float4*>(dispatched_x + dst_base);
    for (int i = lane; i < dv4; i += kWarp) dst[i] = src[i];
  } else {
    for (int i = lane; i < D; i += kWarp) {
      dispatched_x[dst_base + i] = x[src_base + i];
    }
  }
}

// Per-token slot workspace, allocated once and reused across calls (the
// benchmark calls solve in a loop; a malloc/free pair per call would cost
// more than the slot kernel itself).
int* g_slots = nullptr;
int g_slots_cap = 0;

}  // namespace

// x, expert_idx, dispatched_x, token_counts are device pointers.
extern "C" void solve(const float* x, const int* expert_idx,
                      float* dispatched_x, int* token_counts, int T, int D,
                      int E, int capacity) {
  if (T <= 0 || D <= 0 || E <= 0) return;
  if (T > g_slots_cap) {
    if (g_slots) cudaFree(g_slots);
    cudaMalloc(&g_slots, (size_t)T * sizeof(int));
    g_slots_cap = T;
  }

  slot_kernel<<<E, kWarp>>>(expert_idx, g_slots, token_counts, T);

  const int threads = 256;
  const int tokens_per_block = threads / kWarp;
  const int blocks = (T + tokens_per_block - 1) / tokens_per_block;
  dispatch_kernel<<<blocks, threads>>>(x, expert_idx, g_slots, dispatched_x,
                                       T, D, capacity);
}
