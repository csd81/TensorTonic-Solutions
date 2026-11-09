#include <cuda_runtime.h>

#include <cstdint>

// LSD radix sort for 32-bit unsigned integers: 8-bit digits, 4 passes.
//
// The starter ran 8 passes of a 4-bit radix with the histogram, a recursive
// Blelloch scan (three kernels plus cudaMalloc/cudaFree per recursion level)
// and the scatter as separate launches.  It also left the result in the wrong
// buffer: with 8 passes and its two-buffer swap the sorted data ends up in
// d_temp, so it is only correct when the last pass happens to be a no-op
// (anything below 2^28).
//
// This version uses 8-bit digits (4 passes instead of 8) and a flat,
// digit-major scan of the per-block histograms:
//
//   agg[d * nblocks + b] = number of elements of digit d in block b
//
// An exclusive scan over that array in memory order gives, for block b and
// digit d, exactly the destination base: the counts of every smaller digit
// over all blocks (later blocks included) plus the counts of digit d in the
// preceding blocks.  The radix bucket starts are a global quantity, so unlike
// a prefix sum this cannot be done with a cross-block lookback alone -- the
// pass is three stages:
//
//   1. hist_kernel    -- per-block digit histograms (coalesced read).
//   2. scan           -- exclusive scan of the (RADIX x nblocks) matrix.
//   3. scatter_kernel -- stable rank inside the block (per-warp histograms +
//                        a warp digit match built from RADIX_BITS ballots),
//                        staged through shared memory so the global writes
//                        leave in digit-run order and stay coalesced.
//
// The last pass writes straight into `output`, and all buffers are cached
// across calls.

#ifndef ITEMS
#define ITEMS 8
#endif

#define THREADS 256
#define BLOCK_ELEMS (THREADS * ITEMS)
#define RADIX_BITS 8
#define RADIX (1 << RADIX_BITS)
#define NWARPS (THREADS / 32)
#define WARP_MASK 0xFFFFFFFFu
#define SCAN_ITEMS 16
#define SCAN_CHUNK (THREADS * SCAN_ITEMS)

__device__ __forceinline__ unsigned lanemask_lt() {
  unsigned m;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(m));
  return m;
}

// Mask of lanes sharing this lane's digit (bitwise match over RADIX_BITS
// ballots -- the pre-Volta MatchAny trick).  Must be executed by the whole
// warp, so it is never called from inside a divergent branch.
__device__ __forceinline__ unsigned match_digit(unsigned d) {
  unsigned mask = WARP_MASK;
#pragma unroll
  for (int b = 0; b < RADIX_BITS; ++b) {
    const unsigned bit = __ballot_sync(WARP_MASK, (d >> b) & 1u);
    mask &= ((d >> b) & 1u) ? bit : ~bit;
  }
  return mask;
}

// ------------------------------------------------------- stage 1: histograms

__global__ void hist_kernel(const unsigned* __restrict__ src,
                            unsigned* __restrict__ agg, int N, int shift,
                            int nblocks) {
  __shared__ unsigned hist[RADIX];
  for (int i = threadIdx.x; i < RADIX; i += THREADS) hist[i] = 0;
  __syncthreads();

  const int base = blockIdx.x * BLOCK_ELEMS;
  const int nj = ITEMS / 4;
  const bool vec = ((reinterpret_cast<uintptr_t>(src) & 15u) == 0);
#pragma unroll
  for (int q = 0; q < nj; ++q) {
    const int idx = base + (q * 4) * THREADS + threadIdx.x * 4;
    if (vec && idx + 3 < N) {
      const uint4 v = *reinterpret_cast<const uint4*>(src + idx);
      atomicAdd(&hist[(v.x >> shift) & (RADIX - 1)], 1u);
      atomicAdd(&hist[(v.y >> shift) & (RADIX - 1)], 1u);
      atomicAdd(&hist[(v.z >> shift) & (RADIX - 1)], 1u);
      atomicAdd(&hist[(v.w >> shift) & (RADIX - 1)], 1u);
    } else {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int i2 = idx + j;
        if (i2 < N) {
          atomicAdd(&hist[(src[i2] >> shift) & (RADIX - 1)], 1u);
        }
      }
    }
  }
  __syncthreads();

  for (int i = threadIdx.x; i < RADIX; i += THREADS) {
    agg[i * nblocks + blockIdx.x] = hist[i];
  }
}

// ------------------------------------------------- stage 2: exclusive scan

__global__ void scan_blocks_kernel(unsigned* __restrict__ data,
                                   unsigned* __restrict__ sums, int n) {
  __shared__ unsigned s_warp[NWARPS];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int base = blockIdx.x * SCAN_CHUNK + tid * SCAN_ITEMS;

  unsigned run[SCAN_ITEMS];
  unsigned s = 0;
  const bool full = base + SCAN_ITEMS <= n;
  if (full) {
    uint4 v[SCAN_ITEMS / 4];
#pragma unroll
    for (int q = 0; q < SCAN_ITEMS / 4; ++q) {
      v[q] = reinterpret_cast<const uint4*>(data + base)[q];
    }
#pragma unroll
    for (int q = 0; q < SCAN_ITEMS / 4; ++q) {
      run[q * 4 + 0] = s;
      s += v[q].x;
      run[q * 4 + 1] = s;
      s += v[q].y;
      run[q * 4 + 2] = s;
      s += v[q].z;
      run[q * 4 + 3] = s;
      s += v[q].w;
    }
  } else {
#pragma unroll
    for (int j = 0; j < SCAN_ITEMS; ++j) {
      const int idx = base + j;
      const unsigned v = (idx < n) ? data[idx] : 0u;
      run[j] = s;
      s += v;
    }
  }

  unsigned x = s;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const unsigned y = __shfl_up_sync(WARP_MASK, x, off);
    if (lane >= off) x += y;
  }
  if (lane == 31) s_warp[warp] = x;
  __syncthreads();
  if (warp == 0) {
    unsigned w = (lane < NWARPS) ? s_warp[lane] : 0u;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const unsigned y = __shfl_up_sync(WARP_MASK, w, off);
      if (lane >= off) w += y;
    }
    if (lane < NWARPS) s_warp[lane] = w;
  }
  __syncthreads();

  const unsigned wbase = (warp > 0) ? s_warp[warp - 1] : 0u;
  const unsigned tbase = wbase + (x - s);
  if (full) {
#pragma unroll
    for (int q = 0; q < SCAN_ITEMS / 4; ++q) {
      uint4 o;
      o.x = run[q * 4 + 0] + tbase;
      o.y = run[q * 4 + 1] + tbase;
      o.z = run[q * 4 + 2] + tbase;
      o.w = run[q * 4 + 3] + tbase;
      reinterpret_cast<uint4*>(data + base)[q] = o;
    }
  } else {
#pragma unroll
    for (int j = 0; j < SCAN_ITEMS; ++j) {
      const int idx = base + j;
      if (idx < n) data[idx] = run[j] + tbase;
    }
  }
  if (tid == THREADS - 1) sums[blockIdx.x] = wbase + x;
}

__global__ void scan_sums_kernel(unsigned* __restrict__ sums, int m) {
  __shared__ unsigned s_warp[NWARPS];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int runs = (m + THREADS - 1) / THREADS;
  const int base = tid * runs;

  unsigned s = 0;
  for (int j = 0; j < runs; ++j) {
    if (base + j < m) s += sums[base + j];
  }

  unsigned x = s;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const unsigned y = __shfl_up_sync(WARP_MASK, x, off);
    if (lane >= off) x += y;
  }
  if (lane == 31) s_warp[warp] = x;
  __syncthreads();
  if (warp == 0) {
    unsigned w = (lane < NWARPS) ? s_warp[lane] : 0u;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const unsigned y = __shfl_up_sync(WARP_MASK, w, off);
      if (lane >= off) w += y;
    }
    if (lane < NWARPS) s_warp[lane] = w;
  }
  __syncthreads();

  const unsigned wbase = (warp > 0) ? s_warp[warp - 1] : 0u;
  unsigned acc = wbase + (x - s);
  for (int j = 0; j < runs; ++j) {
    if (base + j < m) {
      const unsigned v = sums[base + j];
      sums[base + j] = acc;
      acc += v;
    }
  }
}


// Produces intra-chunk exclusive prefixes in `data` plus the exclusive scan
// of the chunk totals in `sums`; the consumer adds the two.
static void scan_device(unsigned* data, unsigned* sums, int n) {
  const int nb = (n + SCAN_CHUNK - 1) / SCAN_CHUNK;
  scan_blocks_kernel<<<nb, THREADS>>>(data, sums, n);
  scan_sums_kernel<<<1, THREADS>>>(sums, nb);
}

// ---------------------------------------------------------- stage 3: scatter

__global__ void scatter_kernel(const unsigned* __restrict__ src,
                               unsigned* __restrict__ dst,
                               const unsigned* __restrict__ off,
                               const unsigned* __restrict__ sums, int N,
                               int shift, int nblocks) {
  __shared__ unsigned short warp_hist[NWARPS][RADIX];
  __shared__ unsigned block_hist[RADIX];
  __shared__ unsigned block_pref[RADIX];
  __shared__ unsigned block_off[RADIX];
  __shared__ unsigned s_scan[RADIX];
  __shared__ unsigned stage[BLOCK_ELEMS];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int block = blockIdx.x;

  for (int i = tid; i < NWARPS * RADIX; i += THREADS) {
    (&warp_hist[0][0])[i] = 0;
  }
  if (tid < RADIX) {
    const int idx = tid * nblocks + block;
    block_off[tid] = off[idx] + sums[idx / SCAN_CHUNK];
  }
  __syncthreads();

  // Warp w owns block-local positions [w*ITEMS*32, +ITEMS*32), ordered by
  // (item, lane); the loads are 32 consecutive uints per warp instruction.
  const int nvalid = min(BLOCK_ELEMS, N - block * BLOCK_ELEMS);

  unsigned vals[ITEMS];
  unsigned dig[ITEMS];
  bool valid[ITEMS];
#pragma unroll
  for (int i = 0; i < ITEMS; ++i) {
    const int pos = warp * (ITEMS * 32) + i * 32 + lane;  // block-local
    valid[i] = pos < nvalid;
    vals[i] = valid[i] ? src[block * BLOCK_ELEMS + pos] : 0u;
    dig[i] = (vals[i] >> shift) & (RADIX - 1);
  }

  // Stable rank: the running per-warp histogram supplies the count of the
  // same digit over the warp's earlier items, and the match mask supplies the
  // same-digit lanes of the current item.
  unsigned rank[ITEMS];
#pragma unroll
  for (int i = 0; i < ITEMS; ++i) {
    const unsigned vmask = __ballot_sync(WARP_MASK, valid[i]);
    const unsigned md = match_digit(dig[i]);
    const unsigned m = valid[i] ? (md & vmask) : 0u;
    const unsigned prev = warp_hist[warp][dig[i]];
    if (valid[i]) {
      rank[i] = prev + __popc(m & lanemask_lt());
      if (lane == __ffs(m) - 1)
        warp_hist[warp][dig[i]] = static_cast<unsigned short>(prev + __popc(m));
    }
    __syncwarp();
  }
  __syncthreads();

  // Block histogram, and turn each warp row into that warp's base (in place).
  if (tid < RADIX) {
    unsigned sum = 0;
#pragma unroll
    for (int w = 0; w < NWARPS; ++w) {
      const unsigned c = warp_hist[w][tid];
      warp_hist[w][tid] = static_cast<unsigned short>(sum);
      sum += c;
    }
    block_hist[tid] = sum;
  }
  __syncthreads();

  // Exclusive scan of the block histogram over the digits.
  if (tid < RADIX) s_scan[tid] = block_hist[tid];
  __syncthreads();
  for (int d = 1; d < RADIX; d <<= 1) {
    unsigned v = 0;
    if (tid < RADIX && tid >= d) v = s_scan[tid - d];
    __syncthreads();
    if (tid < RADIX) s_scan[tid] += v;
    __syncthreads();
  }
  if (tid < RADIX) block_pref[tid] = s_scan[tid] - block_hist[tid];
  __syncthreads();

  // Stage the block's elements at their block-local sorted positions.
#pragma unroll
  for (int i = 0; i < ITEMS; ++i) {
    if (valid[i]) {
      stage[block_pref[dig[i]] + warp_hist[warp][dig[i]] + rank[i]] = vals[i];
    }
  }
  __syncthreads();

  // Write out in sorted order: consecutive positions belong to the same digit
  // run, so the global stores stay coalesced.  `off` already contains the
  // block's own smaller-digit counts, so subtract the block prefix back out
  // of the staged position (p = block_pref + rank inside the digit run).
  for (int j = 0; j < ITEMS; ++j) {
    const int p = tid + j * THREADS;
    if (p < nvalid) {
      const unsigned v = stage[p];
      const unsigned d = (v >> shift) & (RADIX - 1);
      dst[block_off[d] + (p - static_cast<int>(block_pref[d]))] = v;
    }
  }
}

// ---------------------------------------------------------------- host side

static unsigned* g_tmp0 = nullptr;
static unsigned* g_tmp1 = nullptr;
static unsigned* g_agg = nullptr;
static unsigned* g_sums = nullptr;
static size_t g_capacity = 0;      // elements
static int g_blocks_capacity = 0;  // blocks

static void reserve(size_t n, int nblocks) {
  if (n <= g_capacity && nblocks <= g_blocks_capacity) return;
  cudaFree(g_tmp0);
  cudaFree(g_tmp1);
  cudaFree(g_agg);
  cudaFree(g_sums);
  const size_t cap = n + n / 4 + 1024;
  const int bcap = nblocks + nblocks / 4 + 64;
  cudaMalloc(&g_tmp0, cap * sizeof(unsigned));
  cudaMalloc(&g_tmp1, cap * sizeof(unsigned));
  cudaMalloc(&g_agg, static_cast<size_t>(bcap) * RADIX * sizeof(unsigned));
  cudaMalloc(&g_sums, static_cast<size_t>(bcap) * sizeof(unsigned));
  g_capacity = cap;
  g_blocks_capacity = bcap;
}

// input, output are device pointers.
extern "C" void solve(const unsigned int* input, unsigned int* output, int N) {
  if (N <= 0) return;

  const int nblocks = (N + BLOCK_ELEMS - 1) / BLOCK_ELEMS;
  reserve(static_cast<size_t>(N), nblocks);

  constexpr int PASSES = (32 + RADIX_BITS - 1) / RADIX_BITS;
  const int matrix = nblocks * RADIX;

  const unsigned* src = input;
  for (int pass = 0; pass < PASSES; ++pass) {
    const int shift = pass * RADIX_BITS;
    unsigned* dst = (pass == PASSES - 1)
                        ? output
                        : ((pass % 2 == 0) ? g_tmp0 : g_tmp1);
    hist_kernel<<<nblocks, THREADS>>>(src, g_agg, N, shift, nblocks);
    scan_device(g_agg, g_sums, matrix);
    scatter_kernel<<<nblocks, THREADS>>>(src, dst, g_agg, g_sums, N, shift,
                                        nblocks);
    src = dst;
  }
  cudaDeviceSynchronize();
}
