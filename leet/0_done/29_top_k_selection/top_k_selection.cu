// Top-k selection, k largest in descending order (challenge 29).
//
// Benchmark: N = 50,000,000, k = 100 -> 200 MB of reads, ~2.4 ms at this
// GPU's ~85 GB/s; the algorithm must therefore be a single pass over the
// input.
//
// The starter's fast path is broken at this benchmark: it launches
// BLOCK_SIZE * k floats of dynamic shared memory (256*100*4 = 100 KB) without
// the >48 KB opt-in, so the launch fails for k >= 49 (verified: at k = 100 the
// call returns "invalid argument" and leaves the output untouched).  Its
// k > 256 full-bitonic fallback is the only path that runs for k in [49, 256],
// and that path sorts all N values; at the benchmark shape the starter
// produces no output at all.
//
// This version uses a threshold-buffer selection, which is one pass over the
// data and correct for any value distribution:
//
//  * Every block keeps a shared candidate buffer of CAP = 2048 values plus a
//    threshold `thr` and a count, all in shared memory.  The buffer starts at
//    -inf.  While scanning, a thread appends any value > thr to the buffer
//    (atomic slot).  When the buffer approaches capacity the block sorts it
//    (descending bitonic sort), keeps the top k values as the retained
//    candidates and sets thr to the k-th largest.
//  * The scan advances CHUNK=4 grid strides at a time.  Each phase first
//    loads its CHUNK values into registers (independent loads issued
//    together, which is what keeps the kernel close to the memory wall),
//    then appends.  The capacity check uses __syncthreads_or, which is both
//    the phase barrier and a block-wide OR of the predicate: the compaction
//    decision is therefore uniform even though slower warps may still be
//    finishing the previous phase's appends, and the last appending thread
//    always observes the final count.  At most CHUNK*TPB = 1024 values are
//    appended between two checks, hence CAP-1024 = 1024 free slots at each
//    check are enough to keep every slot in range.
//  * Because thr is a valid lower bound at all times, no element that could
//    belong to the block's top-k is ever dropped; duplicates (all-equal
//    inputs) are handled naturally.  Values in [k, cnt) after a compaction
//    are all <= thr, so the final sort over the whole CAP buffer cannot
//    disturb the true top-k.
//  * Each block writes its k retained values to a candidate array; a second
//    pass of the same kernel over the blocks*k candidates produces the final
//    sorted top-k.
//
// The grid is one full occupancy wave: 8 blocks per SM (CAP = 8 KB of shared
// memory and 256 threads give 8 resident blocks on this GPU), capped by
// ceil(N/TPB).  Using one wave instead of thousands of blocks matters: every
// block compacts and sorts its own candidates, so fewer blocks means less
// sort work per element.
//
// k <= 1024 is handled this way.  For k > 1024 the kernel falls back to a
// full bitonic sort of the input (the same approach as the starter's large-k
// path; correctness only, it is not the benchmarked shape).
//
// Measured: see README.

#include <cuda_runtime.h>

namespace {

constexpr int TPB = 256;
constexpr int CHUNK = 4;          // grid strides per thread between checks
constexpr int CAP = 2048;         // shared candidate buffer, power of two
constexpr int MAX_FAST_K = 1024;  // k handled by the threshold scheme

static_assert(CAP - CHUNK * TPB >= MAX_FAST_K,
              "CAP must leave room for a full phase plus the k retained values");

__device__ __forceinline__ unsigned int float_to_ordered_uint(float f) {
  unsigned int u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Descending bitonic sort of CAP values in shared memory (all threads take
// part; every stage is followed by a barrier).
__device__ __forceinline__ void bitonic_sort_desc(float* buf, int tid) {
  for (int size = 2; size <= CAP; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int i = tid; i < CAP / 2; i += TPB) {
        const int partner = (i / stride) * (2 * stride) + (i % stride);
        const int lo = partner, hi = partner + stride;
        const bool dir_desc = ((lo & size) == 0);
        const float a = buf[lo], b = buf[hi];
        if ((dir_desc && a < b) || (!dir_desc && a > b)) {
          buf[lo] = b;
          buf[hi] = a;
        }
      }
      __syncthreads();
    }
  }
}

// One block selects the top k of the values at `input[0..n)`; `out` receives
// the k largest in descending order.
__global__ void topk_kernel(const float* __restrict__ input,
                            float* __restrict__ out, int n, int k) {
  __shared__ float buf[CAP];
  __shared__ float thr_s;
  __shared__ int cnt_s;

  const float neg_inf = __int_as_float(0xff800000u);
  const int tid = threadIdx.x;

  for (int i = tid; i < CAP; i += TPB) buf[i] = neg_inf;
  if (tid == 0) {
    thr_s = neg_inf;
    cnt_s = 0;
  }
  __syncthreads();

  const int stride = TPB * gridDim.x;
  for (int base = blockIdx.x * TPB; base < n; base += stride * CHUNK) {
    // __syncthreads_or is both the phase barrier and a block-wide OR of the
    // capacity predicate, so the compaction decision is uniform even though
    // the predicate is read while slower warps may still be finishing the
    // previous phase's appends.  The last appending thread observes the final
    // cnt_s, so the OR never misses an over-capacity buffer, and at most
    // CHUNK*TPB values are appended between two checks.
    if (__syncthreads_or(cnt_s > CAP - CHUNK * TPB)) {
      bitonic_sort_desc(buf, tid);
      // After sorting, buf[0..k) are the largest k, descending.
      if (tid == 0) {
        thr_s = buf[k - 1];
        cnt_s = k;
      }
      __syncthreads();
    }
    // Issue the phase's loads together (independent), then decide appends.
    float xs[CHUNK];
#pragma unroll
    for (int c = 0; c < CHUNK; ++c) {
      const int i = base + c * stride + tid;
      xs[c] = (i < n) ? input[i] : neg_inf;
    }
#pragma unroll
    for (int c = 0; c < CHUNK; ++c) {
      if (xs[c] > thr_s) {
        const int slot = atomicAdd(&cnt_s, 1);
        if (slot < CAP) buf[slot] = xs[c];
      }
    }
  }

  // Final sort so the block's retained values are ordered, then emit k.
  __syncthreads();
  bitonic_sort_desc(buf, tid);
  for (int i = tid; i < k; i += TPB) out[(size_t)blockIdx.x * k + i] = buf[i];
}

// Fallback for k > MAX_FAST_K: full bitonic sort of the (padded) input.
__global__ void pad_and_convert_kernel(const float* __restrict__ in,
                                       unsigned int* __restrict__ out, int N,
                                       int padded_N) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    out[idx] = float_to_ordered_uint(in[idx]);
  } else if (idx < padded_N) {
    out[idx] = 0;
  }
}

__global__ void bitonic_sort_step(unsigned int* data, int j, int kstep, int n) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (unsigned int)n) return;  // padded_N may not fill the last block
  const unsigned int ixj = i ^ j;
  if (ixj > i) {
    const bool descending = ((i & kstep) == 0);
    const unsigned int a = data[i], b = data[ixj];
    if ((descending && a < b) || (!descending && a > b)) {
      data[i] = b;
      data[ixj] = a;
    }
  }
}

__global__ void extract_kernel(const unsigned int* __restrict__ sorted,
                               float* __restrict__ output, int k) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < k) {
    unsigned int u = sorted[i];
    u = (u & 0x80000000u) ? (u & ~0x80000000u) : ~u;
    output[i] = __uint_as_float(u);
  }
}

// Candidate scratch, allocated once.
float* g_cand = nullptr;
int g_cand_cap = 0;

}  // namespace

extern "C" void solve(const float* input, float* output, int N, int k) {
  if (N <= 0 || k <= 0 || k > N) return;

  if (k <= MAX_FAST_K) {
    // One full occupancy wave: 8 resident blocks per SM at this block's
    // 8 KB of static shared memory and 256 threads.
    static int g_blocks_per_wave = 0;
    if (g_blocks_per_wave == 0) {
      int sms = 0;
      if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0) !=
              cudaSuccess ||
          sms <= 0) {
        sms = 6;
      }
      g_blocks_per_wave = sms * 8;
    }
    int blocks = (N + TPB - 1) / TPB;
    if (blocks > g_blocks_per_wave) blocks = g_blocks_per_wave;
    if (blocks < 1) blocks = 1;
    const int ncand = blocks * k;
    if (ncand > g_cand_cap) {
      if (g_cand) cudaFree(g_cand);
      cudaMalloc(&g_cand, (size_t)ncand * sizeof(float));
      g_cand_cap = ncand;
    }
    topk_kernel<<<blocks, TPB>>>(input, g_cand, N, k);
    // Stage 2: the same selection over the blocks' candidates.
    topk_kernel<<<1, TPB>>>(g_cand, output, ncand, k);
    return;
  }

  // Large k: full bitonic sort (not benchmarked; matches the starter's
  // approach for k > 256).
  int padded_N = 1;
  while (padded_N < N) padded_N <<= 1;
  unsigned int* d_data = nullptr;
  cudaMalloc(&d_data, (size_t)padded_N * sizeof(unsigned int));
  const int blocks = (padded_N + TPB - 1) / TPB;
  pad_and_convert_kernel<<<blocks, TPB>>>(input, d_data, N, padded_N);
  for (int kstep = 2; kstep <= padded_N; kstep <<= 1) {
    for (int j = kstep >> 1; j > 0; j >>= 1) {
      bitonic_sort_step<<<blocks, TPB>>>(d_data, j, kstep, padded_N);
    }
  }
  extract_kernel<<<(k + TPB - 1) / TPB, TPB>>>(d_data, output, k);
  cudaFree(d_data);
}
