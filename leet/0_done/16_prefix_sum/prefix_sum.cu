#include <cuda_runtime.h>

#include <cstdint>

// Inclusive prefix sum of a float array.
//
// At the benchmark size (N = 250,000) the array is only 1 MB, so the answer
// is decided by how many kernels have to run, not by bandwidth: the starter
// recurses over block totals, allocating with cudaMalloc at every level and
// launching three kernels per level.  For N = 250,000 that is two levels --
// block scan, recursive scan of 489 block totals, add block offsets.
//
// This version is a single launch: a decoupled-lookback scan (CUB's
// DeviceScan design).  Each block scans its own 2048 elements (per-thread
// sequential runs of ITEMS, a warp-level scan of the run totals, a warp-0
// scan of the per-warp totals), publishes its block total, and resolves the
// exclusive prefix of its predecessors by walking back over their published
// aggregates until it finds a block whose *inclusive* prefix has been
// published.  Nothing else is needed -- no second pass, no host recursion.
//
// The lookback scratch is cached across calls.

#ifndef ITEMS
#define ITEMS 16
#endif

#define THREADS 256
#define BLOCK_ELEMS (THREADS * ITEMS)
#define NWARPS (THREADS / 32)
#define WARP_MASK 0xFFFFFFFFu

__global__ void scan_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            float* __restrict__ agg, float* __restrict__ pre,
                            int* __restrict__ status, int N) {
  __shared__ float warp_tot[NWARPS];
  __shared__ float warp_pre[NWARPS];
  __shared__ float s_block_pre;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int block = blockIdx.x;
  const int base = block * BLOCK_ELEMS + tid * ITEMS;

  // Load the thread's ITEMS consecutive elements and scan them in registers.
  float v[ITEMS];
  const bool full = (base + ITEMS <= N);
  if (full) {
#pragma unroll
    for (int q = 0; q < ITEMS / 4; ++q) {
      const float4 f =
          *reinterpret_cast<const float4*>(input + base + q * 4);
      v[q * 4 + 0] = f.x;
      v[q * 4 + 1] = f.y;
      v[q * 4 + 2] = f.z;
      v[q * 4 + 3] = f.w;
    }
  } else {
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
      const int idx = base + i;
      v[i] = (idx < N) ? input[idx] : 0.0f;
    }
  }
  float run = 0.0f;
#pragma unroll
  for (int i = 0; i < ITEMS; ++i) {
    run += v[i];
    v[i] = run;
  }

  // Warp-inclusive scan of the run totals; keep the exclusive value taken
  // from the previous lane's inclusive result (no subtraction).
  float x = run;
  float prev = 0.0f;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const float y = __shfl_up_sync(WARP_MASK, x, off);
    if (lane >= off) x += y;
  }
  prev = __shfl_up_sync(WARP_MASK, x, 1);
  if (lane == 31) warp_tot[warp] = x;
  __syncthreads();

  if (warp == 0) {
    float w = (lane < NWARPS) ? warp_tot[lane] : 0.0f;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const float y = __shfl_up_sync(WARP_MASK, w, off);
      if (lane >= off) w += y;
    }
    if (lane < NWARPS) warp_pre[lane] = w;
  }
  __syncthreads();

  const float wpre = (warp > 0) ? warp_pre[warp - 1] : 0.0f;
  const float thread_pre = wpre + ((lane > 0) ? prev : 0.0f);
  const float block_total = warp_pre[NWARPS - 1];

  // Decoupled lookback: thread 0 resolves the block's exclusive prefix.
  if (tid == 0) {
    if (block == 0) {
      agg[0] = block_total;
      pre[0] = block_total;  // inclusive prefix of block 0
      __threadfence();
      status[0] = 2;
      s_block_pre = 0.0f;
    } else {
      agg[block] = block_total;
      __threadfence();
      status[block] = 1;

      float running = 0.0f;
      int i = block - 1;
      while (true) {
        const int st = atomicAdd(&status[i], 0);  // coherent read
        if (st == 0) continue;
        __threadfence();
        running = (st == 2) ? (pre[i] + running) : (agg[i] + running);
        if (st == 2) break;
        --i;
      }
      pre[block] = running + block_total;
      __threadfence();
      status[block] = 2;
      s_block_pre = running;
    }
  }
  __syncthreads();

  const float off = s_block_pre + thread_pre;
  if (full) {
#pragma unroll
    for (int q = 0; q < ITEMS / 4; ++q) {
      float4 f;
      f.x = off + v[q * 4 + 0];
      f.y = off + v[q * 4 + 1];
      f.z = off + v[q * 4 + 2];
      f.w = off + v[q * 4 + 3];
      *reinterpret_cast<float4*>(output + base + q * 4) = f;
    }
  } else {
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
      const int idx = base + i;
      if (idx < N) output[idx] = off + v[i];
    }
  }
}

// ---------------------------------------------------------------- host side

static float* g_agg = nullptr;
static float* g_pre = nullptr;
static int* g_status = nullptr;
static int g_capacity = 0;

static void reserve(int nblocks) {
  if (nblocks <= g_capacity) return;
  cudaFree(g_agg);
  cudaFree(g_pre);
  cudaFree(g_status);
  const int cap = nblocks + nblocks / 4 + 1024;
  cudaMalloc(&g_agg, static_cast<size_t>(cap) * sizeof(float));
  cudaMalloc(&g_pre, static_cast<size_t>(cap) * sizeof(float));
  cudaMalloc(&g_status, static_cast<size_t>(cap) * sizeof(int));
  g_capacity = cap;
}

// input, output are device pointers.
extern "C" void solve(const float* input, float* output, int N) {
  if (N <= 0) return;

  const int nblocks = (N + BLOCK_ELEMS - 1) / BLOCK_ELEMS;
  reserve(nblocks);
  cudaMemsetAsync(g_status, 0, static_cast<size_t>(nblocks) * sizeof(int));
  scan_kernel<<<nblocks, THREADS>>>(input, output, g_agg, g_pre, g_status, N);
  cudaDeviceSynchronize();
}
