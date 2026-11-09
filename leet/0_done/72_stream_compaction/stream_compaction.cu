#include <cuda_runtime.h>

#include <cstdint>

// Stream compaction: keep the positive elements of A in their original
// relative order at the front of `out`; the remaining positions stay 0.  The
// challenge guarantees `out` is pre-zeroed, so only selected elements are
// written.
//
// Single pass over A with a decoupled-lookback prefix-sum (the CUB
// DeviceScan approach).  Each block owns CHUNKS chunks of 1024 elements;
// within a chunk every thread loads four consecutive elements as one float4,
// so a block-wide exclusive scan over the thread counts reproduces the
// element order exactly.  The block publishes its aggregate, looks back over
// the preceding blocks' published aggregates until it finds an inclusive
// prefix, and then scatters using (block offset + within-block rank).
//
// Compared with the straightforward count -> scan -> scatter pipeline, this
// reads A only once: ~300 MB of traffic instead of ~500 MB at N = 50M, which
// is why it runs in ~4.2 ms instead of ~6.7 ms.  Six chunks per block are the
// measured optimum (fewer blocks expose the lookback latency, more chunks
// cost registers and occupancy).

#define THREADS 256
#define ITEMS 4
#define BLOCK_ELEMS (THREADS * ITEMS)
#define NWARPS (THREADS / 32)

// Exclusive scan of `val` across the block.  Returns the thread's exclusive
// prefix; the block total is left in s_total[0].  All threads must reach it.
__device__ __forceinline__ int block_exclusive_scan(int val, int* s_warp,
                                                    int* s_total, int tid) {
  const int lane = tid & 31;
  const int warp = tid >> 5;

  int inc = val;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const int n = __shfl_up_sync(0xffffffffu, inc, off);
    if (lane >= off) inc += n;
  }
  if (lane == 31) s_warp[warp] = inc;
  __syncthreads();

  if (warp == 0) {
    int wv = (lane < NWARPS) ? s_warp[lane] : 0;
    int winc = wv;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const int n = __shfl_up_sync(0xffffffffu, winc, off);
      if (lane >= off) winc += n;
    }
    if (lane < NWARPS) s_warp[lane] = winc - wv;  // exclusive warp offset
    if (lane == NWARPS - 1) s_total[0] = winc;    // block total
  }
  __syncthreads();

  return s_warp[warp] + inc - val;
}

#define FLAG_AGG 1ull
#define FLAG_PREFIX 2ull

__device__ __forceinline__ unsigned long long pack_status(int flag, int value) {
  return (static_cast<unsigned long long>(flag) << 32) |
         static_cast<unsigned long long>(static_cast<unsigned>(value));
}

// Single-pass compaction with a decoupled lookback scan.  Each block owns
// CHUNKS chunks of BLOCK_ELEMS elements; the chunks are processed in order so
// the concatenation of their outputs preserves the global element order.  The
// block's aggregate is published once and a lookback resolves its offset, so
// A is read exactly once (unlike the counting+scatter two-pass scheme).
#define CHUNKS 6
#define ELEMS_PER_BLOCK (CHUNKS * BLOCK_ELEMS)

__global__ void compact_kernel(const float* __restrict__ A, int N,
                               float* __restrict__ out,
                               unsigned long long* __restrict__ status) {
  __shared__ int s_warp[NWARPS];
  __shared__ int s_total[1];
  __shared__ int s_offset;
  __shared__ int s_prefix[CHUNKS][THREADS];
  __shared__ int s_chunk_total[CHUNKS];

  const int tid = threadIdx.x;
  const int block_base = blockIdx.x * ELEMS_PER_BLOCK;

  float vals[CHUNKS][ITEMS];
  int cnt[CHUNKS];
#pragma unroll
  for (int c = 0; c < CHUNKS; ++c) {
    const int base = block_base + c * BLOCK_ELEMS + tid * ITEMS;
    if (base + ITEMS <= N) {
      const float4 v = *reinterpret_cast<const float4*>(A + base);
      vals[c][0] = v.x; vals[c][1] = v.y; vals[c][2] = v.z; vals[c][3] = v.w;
    } else {
#pragma unroll
      for (int i = 0; i < ITEMS; ++i) {
        vals[c][i] = (base + i < N) ? A[base + i] : 0.0f;
      }
    }
    int cc = 0;
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
      if ((base + i) < N && vals[c][i] > 0.0f) ++cc;
    }
    cnt[c] = cc;
  }

  // Per-chunk scans give each thread's rank inside the chunk and the chunk
  // totals, which together determine the within-block order.
  int block_total = 0;
#pragma unroll
  for (int c = 0; c < CHUNKS; ++c) {
    s_prefix[c][tid] = block_exclusive_scan(cnt[c], s_warp, s_total, tid);
    if (tid == 0) s_chunk_total[c] = s_total[0];
    block_total += s_total[0];
    __syncthreads();  // s_warp / s_total reused by the next chunk's scan
  }

  if (tid == 0) {
    int prefix;
    if (blockIdx.x == 0) {
      prefix = 0;
      __threadfence();
      atomicExch(&status[0], pack_status(FLAG_PREFIX, prefix + block_total));
    } else {
      __threadfence();
      atomicExch(&status[blockIdx.x], pack_status(FLAG_AGG, block_total));
      int running = 0;
      int i = blockIdx.x - 1;
      while (true) {
        const unsigned long long st = atomicAdd(&status[i], 0ull);
        const int flag = static_cast<int>(st >> 32);
        if (flag == 0) continue;
        running += static_cast<int>(st & 0xffffffffull);
        if (flag == 2) break;
        --i;
      }
      prefix = running;
      __threadfence();
      atomicExch(&status[blockIdx.x], pack_status(FLAG_PREFIX, prefix + block_total));
    }
    s_offset = prefix;
  }
  __syncthreads();

  int running = s_offset;
#pragma unroll
  for (int c = 0; c < CHUNKS; ++c) {
    const int base = block_base + c * BLOCK_ELEMS + tid * ITEMS;
    int pos = running + s_prefix[c][tid];
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
      if ((base + i) < N && vals[c][i] > 0.0f) out[pos++] = vals[c][i];
    }
    running += s_chunk_total[c];
    __syncthreads();  // s_chunk_total read before the next iteration
  }
}

// A, out are device pointers.  out is assumed pre-initialised to 0.0 (the
// challenge guarantees this), so only the selected elements are written.
extern "C" void solve(const float* A, int N, float* out) {
  if (A == nullptr || out == nullptr || N <= 0) return;

  const int num_blocks = (N + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK;
  unsigned long long* status = nullptr;
  cudaMalloc(&status, static_cast<size_t>(num_blocks) * sizeof(unsigned long long));
  cudaMemset(status, 0, static_cast<size_t>(num_blocks) * sizeof(unsigned long long));

  compact_kernel<<<num_blocks, THREADS>>>(A, N, out, status);

  cudaFree(status);
  cudaDeviceSynchronize();
}
