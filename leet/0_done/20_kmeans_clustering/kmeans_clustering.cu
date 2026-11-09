#include <cuda_runtime.h>

#include <cooperative_groups.h>

#include <cfloat>
#include <cstdint>

namespace cg = cooperative_groups;

// k-means over 2D points: assign, re-mean, repeat until the centroids stop
// moving (or max_iterations).
//
// At the benchmark size (N = 10,000, k = 5, 30 iterations) the arithmetic is
// trivial -- about 1.5 M distance evaluations in total -- so the starter's
// 1.35 ms is almost entirely launch and sync overhead: 30 iterations of
// (assign kernel, 4 cudaMemset, accumulate kernel, update kernel, blocking
// 4-byte copy to the host) is ~150 launches, 120 memsets and 30 full device
// stalls.
//
// This version runs the whole clustering in a single cooperative launch:
//
//   for each iteration:
//     assign + accumulate
//     grid.sync()
//     re-mean the centroids, zero the accumulators, note whether anything moved
//     grid.sync()
//     stop when nothing moved
//   final assignment pass so the labels match the final centroids
//
// Accumulation is the part that needs care.  A per-element atomicAdd into the
// cluster sums serialises badly: with five clusters, a warp's 32 atomics
// collapse onto five shared addresses, and in the degenerate case (every point
// landing in one cluster) 10,000 atomics chase a single address -- measured at
// ~500 us per iteration.  So for k <= 8 each thread accumulates into eight
// register-resident (sum_x, sum_y, count) triples with predicated adds, and
// the block reduces them with warp shuffles; only `blocks * k` values reach
// global memory.  Larger k spreads the atomics over enough addresses to be
// harmless and takes the simpler shared-atomic path.
//
// The grid is sized from the workload: one block unless the per-iteration work
// can pay for a cross-block sync (and with a single block the grid sync
// degenerates to a block barrier).

#ifndef KM_THREADS
#define KM_THREADS 1024
#endif
#ifndef KM_BLOCK_TARGET
#define KM_BLOCK_TARGET 16  // distance evals per thread per iteration
#endif

#define KM_WARPS (KM_THREADS / 32)
#define MAX_SHARED_K 4000  // k * 12 bytes must fit in 48 KB of shared memory
#define WARP_MASK 0xFFFFFFFFu

// `*_x`/`*_y` are device pointers; the initial centroids are copied into the
// final arrays by the kernel itself.
__global__ void __launch_bounds__(KM_THREADS)
kmeans_kernel(const float* __restrict__ data_x,
                              const float* __restrict__ data_y,
                              int* __restrict__ labels,
                              const float* __restrict__ init_x,
                              const float* __restrict__ init_y,
                              float* __restrict__ cx, float* __restrict__ cy,
                              int n, int k, int max_iter,
                              float* __restrict__ gsum_x,
                              float* __restrict__ gsum_y,
                              int* __restrict__ gcount,
                              int* __restrict__ moved) {
  extern __shared__ float smem[];
  const bool use_shared = (k <= MAX_SHARED_K);
  float* s_sum_x = smem;
  float* s_sum_y = use_shared ? smem + k : nullptr;
  int* s_cnt = use_shared ? reinterpret_cast<int*>(smem + 2 * k) : nullptr;

  // Per-warp partial sums for the small-k path: [warp][cluster][x, y, count].
  __shared__ float wbuf[KM_WARPS][8][3];

  cg::grid_group grid = cg::this_grid();
  // With a single block the grid barrier is just a block barrier; skipping the
  // cooperative-groups path saves about 2 us per sync at the benchmark size.
  const bool one_block = (gridDim.x == 1);
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int gtid = blockIdx.x * blockDim.x + tid;
  const int gsize = gridDim.x * blockDim.x;

  // Prologue: zero the global accumulators, copy the initial centroids.
  for (int c = tid; c < k; c += blockDim.x) {
    gsum_x[c] = 0.0f;
    gsum_y[c] = 0.0f;
    gcount[c] = 0;
  }
  for (int c = gtid; c < k; c += gsize) {
    cx[c] = init_x[c];
    cy[c] = init_y[c];
  }
  if (one_block) __syncthreads(); else grid.sync();

  for (int it = 0; it < max_iter; ++it) {
    if (gtid == 0) *moved = 0;

    if (k <= 8) {
      // Register accumulation, no atomics in the hot loop.
      float ax[8], ay[8];
      int ac[8];
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        ax[c] = 0.0f;
        ay[c] = 0.0f;
        ac[c] = 0;
      }
      // Centroids in registers: reading them from global memory inside the
      // per-point loop costs ~16 us per iteration at the benchmark size.
      // Slots at or beyond k hold FLT_MAX so their distance never wins.
      float lcx[8], lcy[8];
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        lcx[c] = (c < k) ? cx[c] : FLT_MAX;
        lcy[c] = (c < k) ? cy[c] : FLT_MAX;
      }
      for (int i = gtid; i < n; i += gsize) {
        const float px = data_x[i];
        const float py = data_y[i];
        float best = FLT_MAX;
        int bc = 0;
#pragma unroll
        for (int c = 0; c < 8; ++c) {
          const float ddx = px - lcx[c];
          const float ddy = py - lcy[c];
          const float d = ddx * ddx + ddy * ddy;
          if (d < best) {
            best = d;
            bc = c;
          }
        }
        labels[i] = bc;
#pragma unroll
        for (int c = 0; c < 8; ++c) {
          if (bc == c) {
            ax[c] += px;
            ay[c] += py;
            ac[c] += 1;
          }
        }
      }
      // Warp-reduce each cluster, combine the warps, then one atomic per
      // (block, cluster).
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        if (c >= k) break;
        float vx = ax[c];
        float vy = ay[c];
        int vc = ac[c];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
          vx += __shfl_down_sync(WARP_MASK, vx, off);
          vy += __shfl_down_sync(WARP_MASK, vy, off);
          vc += __shfl_down_sync(WARP_MASK, vc, off);
        }
        if (lane == 0) {
          wbuf[warp][c][0] = vx;
          wbuf[warp][c][1] = vy;
          wbuf[warp][c][2] = static_cast<float>(vc);
        }
      }
      __syncthreads();
      if (tid < k) {
        float tx = 0.0f;
        float ty = 0.0f;
        int tc = 0;
#pragma unroll
        for (int w = 0; w < KM_WARPS; ++w) {
          tx += wbuf[w][tid][0];
          ty += wbuf[w][tid][1];
          tc += static_cast<int>(wbuf[w][tid][2]);
        }
        if (tc > 0) {
          atomicAdd(&gsum_x[tid], tx);
          atomicAdd(&gsum_y[tid], ty);
          atomicAdd(&gcount[tid], tc);
        }
      }
    } else {
      // Large k: the atomics spread over enough addresses to be cheap.
      if (use_shared) {
        for (int c = tid; c < k; c += blockDim.x) {
          s_sum_x[c] = 0.0f;
          s_sum_y[c] = 0.0f;
          s_cnt[c] = 0;
        }
        __syncthreads();
      }
      for (int i = gtid; i < n; i += gsize) {
        const float px = data_x[i];
        const float py = data_y[i];
        float best = FLT_MAX;
        int bc = 0;
        for (int c = 0; c < k; ++c) {
          const float ddx = px - cx[c];
          const float ddy = py - cy[c];
          const float d = ddx * ddx + ddy * ddy;
          if (d < best) {
            best = d;
            bc = c;
          }
        }
        labels[i] = bc;
        if (use_shared) {
          atomicAdd(&s_sum_x[bc], px);
          atomicAdd(&s_sum_y[bc], py);
          atomicAdd(&s_cnt[bc], 1);
        } else {
          atomicAdd(&gsum_x[bc], px);
          atomicAdd(&gsum_y[bc], py);
          atomicAdd(&gcount[bc], 1);
        }
      }
      if (use_shared) {
        __syncthreads();
        for (int c = tid; c < k; c += blockDim.x) {
          if (s_cnt[c] > 0) {
            atomicAdd(&gsum_x[c], s_sum_x[c]);
            atomicAdd(&gsum_y[c], s_sum_y[c]);
            atomicAdd(&gcount[c], s_cnt[c]);
          }
        }
      }
    }
    if (one_block) __syncthreads(); else grid.sync();

    // Re-mean, reset the accumulators, and record whether anything moved.
    for (int c = gtid; c < k; c += gsize) {
      const int cnt = gcount[c];
      if (cnt > 0) {
        const float nx = gsum_x[c] / static_cast<float>(cnt);
        const float ny = gsum_y[c] / static_cast<float>(cnt);
        if (fabsf(nx - cx[c]) > 1e-6f || fabsf(ny - cy[c]) > 1e-6f) *moved = 1;
        cx[c] = nx;
        cy[c] = ny;
      }
      gsum_x[c] = 0.0f;
      gsum_y[c] = 0.0f;
      gcount[c] = 0;
    }
    if (one_block) __syncthreads(); else grid.sync();
    if (*moved == 0) break;
  }

  // The loop updates after the last assignment, so classify once more.
  if (k <= 8) {
    float lcx[8], lcy[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      lcx[c] = (c < k) ? cx[c] : FLT_MAX;
      lcy[c] = (c < k) ? cy[c] : FLT_MAX;
    }
    for (int i = gtid; i < n; i += gsize) {
      const float px = data_x[i];
      const float py = data_y[i];
      float best = FLT_MAX;
      int bc = 0;
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        const float ddx = px - lcx[c];
        const float ddy = py - lcy[c];
        const float d = ddx * ddx + ddy * ddy;
        if (d < best) {
          best = d;
          bc = c;
        }
      }
      labels[i] = bc;
    }
  } else {
    for (int i = gtid; i < n; i += gsize) {
      const float px = data_x[i];
      const float py = data_y[i];
      float best = FLT_MAX;
      int bc = 0;
      for (int c = 0; c < k; ++c) {
        const float ddx = px - cx[c];
        const float ddy = py - cy[c];
        const float d = ddx * ddx + ddy * ddy;
        if (d < best) {
          best = d;
          bc = c;
        }
      }
      labels[i] = bc;
    }
  }
}

// ---------------------------------------------------------------- host side

static float* g_sum_x = nullptr;
static float* g_sum_y = nullptr;
static int* g_count = nullptr;
static int* g_moved = nullptr;
static int g_cap_k = 0;
static int g_sm_count = 0;

static void reserve(int k) {
  if (k <= g_cap_k) return;
  cudaFree(g_sum_x);
  cudaFree(g_sum_y);
  cudaFree(g_count);
  cudaFree(g_moved);
  const int cap = k + k / 4 + 64;
  cudaMalloc(&g_sum_x, cap * sizeof(float));
  cudaMalloc(&g_sum_y, cap * sizeof(float));
  cudaMalloc(&g_count, cap * sizeof(int));
  cudaMalloc(&g_moved, sizeof(int));
  g_cap_k = cap;
}

static void query_device() {
  if (g_sm_count != 0) return;
  cudaDeviceGetAttribute(&g_sm_count, cudaDevAttrMultiProcessorCount, 0);
  if (g_sm_count < 1) g_sm_count = 1;
}

// data_x, data_y, labels, initial_centroid_x/y, final_centroid_x/y are device
// pointers.
extern "C" void solve(const float* data_x, const float* data_y, int* labels,
                      float* initial_centroid_x, float* initial_centroid_y,
                      float* final_centroid_x, float* final_centroid_y,
                      int sample_size, int k, int max_iterations) {
  if (sample_size <= 0 || k <= 0 || max_iterations < 0) return;

  reserve(k);
  query_device();

  const int shmem = (k <= MAX_SHARED_K)
                        ? k * (2 * (int)sizeof(float) + (int)sizeof(int))
                        : 0;

  // Cooperative launches must fit co-resident; size the grid from the actual
  // shared-memory footprint rather than assuming the occupancy.
  int per_sm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kmeans_kernel,
                                                KM_THREADS, shmem);
  int max_blocks = per_sm * g_sm_count;
  if (max_blocks < 1) max_blocks = 1;

  // One block unless the per-iteration work can pay for the cross-block sync.
  const long long work = static_cast<long long>(sample_size) * k;
  long long want = (work + KM_THREADS * KM_BLOCK_TARGET - 1) /
                   (KM_THREADS * KM_BLOCK_TARGET);
  if (want < 1) want = 1;
  if (want > max_blocks) want = max_blocks;
  const int blocks = static_cast<int>(want);

  void* args[] = {(void*)&data_x,
                  (void*)&data_y,
                  (void*)&labels,
                  (void*)&initial_centroid_x,
                  (void*)&initial_centroid_y,
                  (void*)&final_centroid_x,
                  (void*)&final_centroid_y,
                  (void*)&sample_size,
                  (void*)&k,
                  (void*)&max_iterations,
                  (void*)&g_sum_x,
                  (void*)&g_sum_y,
                  (void*)&g_count,
                  (void*)&g_moved};
  cudaError_t err = cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(kmeans_kernel), dim3(blocks), dim3(KM_THREADS),
      args, shmem, 0);
  if (err != cudaSuccess) {
    // Fall back to the smallest grid that always fits.
    cudaGetLastError();
    err = cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(kmeans_kernel), dim3(1), dim3(KM_THREADS),
        args, shmem, 0);
  }
  cudaDeviceSynchronize();
  (void)err;
}
