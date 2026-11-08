#include <cuda_runtime.h>

#include <cstdint>

// Exclusive segmented prefix sum:
//   output[i] = sum of values[j] for j < i inside i's segment (0 at a flag),
//   where flags[i] == 1 starts a new segment.
//
// The starter ran three passes plus a recursive scan over block aggregates and
// materialised the whole inclusive scan as a double array (400 MB at
// N = 50 M), so it moved well over a gigabyte.  This version is a single pass
// with a decoupled-lookback scan (CUB DeviceScan style) generalised to the
// segmented operator:
//
//   seg_op(a, b) = ( b.flag ? b.val : a.val + b.val , a.flag | b.flag )
//
// Each block owns CHUNKS chunks of THREADS * ITEMS elements, publishes its
// aggregate, looks back over preceding blocks until it finds a published
// inclusive prefix, and then writes the float output directly -- values and
// flags are read once and no intermediate array is allocated.
//
// Accumulation is double (float would lose too much over a segment of tens of
// millions of elements); the output is float as specified.

#ifndef THREADS
#define THREADS 256
#endif
#ifndef ITEMS
#define ITEMS 16
#endif
#ifndef CHUNKS
#define CHUNKS 2
#endif
#define BLOCK_ELEMS (CHUNKS * THREADS * ITEMS)
#define NWARPS (THREADS / 32)

struct Pair {
  double val;
  int flag;
};

__device__ __forceinline__ Pair seg_op(Pair a, Pair b) {
  Pair r;
  r.val = b.flag ? b.val : (a.val + b.val);
  r.flag = a.flag | b.flag;
  return r;
}

__device__ __forceinline__ Pair warp_scan(Pair x, int lane) {
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const double ov = __shfl_up_sync(0xffffffffu, x.val, off);
    const int of = __shfl_up_sync(0xffffffffu, x.flag, off);
    if (lane >= off) x = seg_op(Pair{ov, of}, x);
  }
  return x;
}

// ALIGNED: both input arrays and the output are 16-byte aligned (always true
// for cudaMalloc'ed memory), so each ITEMS/4 consecutive groups move as
// float4/int4 instructions.  Without it the warp's scalar loads are a
// stride-ITEMS gather that costs several times the L1 wavefronts.
template <bool ALIGNED>
__global__ void segscan_kernel(const float* __restrict__ values,
                               const int* __restrict__ flags,
                               float* __restrict__ output,
                               double* __restrict__ agg_val,
                               int* __restrict__ agg_flag,
                               double* __restrict__ pre_val,
                               int* __restrict__ pre_flag,
                               int* __restrict__ status, int N) {
  static_assert(ITEMS % 4 == 0, "vectorised path assumes ITEMS multiple of 4");
  __shared__ double s_val[CHUNKS][NWARPS];
  __shared__ int s_flag[CHUNKS][NWARPS];
  __shared__ double s_block_pre_val;
  __shared__ int s_block_pre_flag;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int block_base = blockIdx.x * BLOCK_ELEMS;

  double vals[CHUNKS][ITEMS];
  int flg[CHUNKS][ITEMS];
#pragma unroll
  for (int c = 0; c < CHUNKS; ++c) {
    const int base = block_base + c * (THREADS * ITEMS) + tid * ITEMS;
    if (ALIGNED && base + ITEMS <= N) {
#pragma unroll
      for (int i = 0; i < ITEMS; i += 4) {
        const float4 v4 = *reinterpret_cast<const float4*>(values + base + i);
        const int4 f4 = *reinterpret_cast<const int4*>(flags + base + i);
        vals[c][i + 0] = static_cast<double>(v4.x);
        vals[c][i + 1] = static_cast<double>(v4.y);
        vals[c][i + 2] = static_cast<double>(v4.z);
        vals[c][i + 3] = static_cast<double>(v4.w);
        flg[c][i + 0] = f4.x;
        flg[c][i + 1] = f4.y;
        flg[c][i + 2] = f4.z;
        flg[c][i + 3] = f4.w;
      }
    } else {
#pragma unroll
      for (int i = 0; i < ITEMS; ++i) {
        const int idx = base + i;
        if (idx < N) {
          vals[c][i] = static_cast<double>(values[idx]);
          flg[c][i] = flags[idx];
        } else {
          vals[c][i] = 0.0;
          flg[c][i] = 0;
        }
      }
    }
  }

  // Per-thread aggregate of each chunk (the segmented operator's pair for the
  // thread's ITEMS contiguous elements).
  Pair agg[CHUNKS];
#pragma unroll
  for (int c = 0; c < CHUNKS; ++c) {
    double s = 0.0;
    int anyf = 0;
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
      if (flg[c][i]) {
        s = vals[c][i];
        anyf = 1;
      } else {
        s += vals[c][i];
      }
    }
    agg[c] = Pair{s, anyf};
  }

  // Warp scan of the aggregates of every chunk, then a single warp scans the
  // per-warp aggregates -- all chunks share the same two barriers.
  Pair local_inc[CHUNKS];
#pragma unroll
  for (int c = 0; c < CHUNKS; ++c) {
    local_inc[c] = warp_scan(agg[c], lane);
    if (lane == 31) {
      s_val[c][warp] = local_inc[c].val;
      s_flag[c][warp] = local_inc[c].flag;
    }
  }
  __syncthreads();

  if (warp == 0) {
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
      const Pair w = (lane < NWARPS) ? Pair{s_val[c][lane], s_flag[c][lane]}
                                     : Pair{0.0, 0};
      const Pair wi = warp_scan(w, lane);
      if (lane < NWARPS) {
        s_val[c][lane] = wi.val;
        s_flag[c][lane] = wi.flag;
      }
    }
  }
  __syncthreads();

  // s_val[c][NWARPS-1] now holds chunk c's aggregate; every thread can build
  // the block aggregate from it.
  Pair block_agg = Pair{s_val[0][NWARPS - 1], s_flag[0][NWARPS - 1]};
#pragma unroll
  for (int c = 1; c < CHUNKS; ++c) {
    block_agg = seg_op(
        block_agg, Pair{s_val[c][NWARPS - 1], s_flag[c][NWARPS - 1]});
  }

  if (tid == 0) {
    // Decoupled lookback: publish the aggregate, then walk backwards until a
    // block with a published inclusive prefix is found (status 2 entries hold
    // prefix + own aggregate, so one lookup resolves a predecessor).
    Pair prefix;  // exclusive: everything before this block
    if (blockIdx.x == 0) {
      prefix = Pair{0.0, 0};
      agg_val[0] = block_agg.val;
      agg_flag[0] = block_agg.flag;
      const Pair incl = seg_op(prefix, block_agg);
      pre_val[0] = incl.val;
      pre_flag[0] = incl.flag;
      __threadfence();
      status[0] = 2;  // published last: readers see prefix and aggregate
    } else {
      agg_val[blockIdx.x] = block_agg.val;
      agg_flag[blockIdx.x] = block_agg.flag;
      __threadfence();
      status[blockIdx.x] = 1;

      Pair running = Pair{0.0, 0};
      int i = blockIdx.x - 1;
      while (true) {
        const int st = atomicAdd(&status[i], 0);  // coherent read
        if (st == 0) continue;
        __threadfence();
        const Pair item = (st == 2)
                              ? Pair{pre_val[i], pre_flag[i]}
                              : Pair{agg_val[i], agg_flag[i]};
        running = seg_op(item, running);
        if (st == 2) break;
        --i;
      }
      prefix = running;
      const Pair incl = seg_op(prefix, block_agg);
      pre_val[blockIdx.x] = incl.val;
      pre_flag[blockIdx.x] = incl.flag;
      __threadfence();
      status[blockIdx.x] = 2;
    }
    s_block_pre_val = prefix.val;
    s_block_pre_flag = prefix.flag;
  }
  __syncthreads();

  // Chunk prefixes: the block prefix combined with the earlier chunks'
  // aggregates.
  Pair chunk_pre = Pair{s_block_pre_val, s_block_pre_flag};
#pragma unroll
  for (int c = 0; c < CHUNKS; ++c) {
    const Pair wp = (warp > 0) ? Pair{s_val[c][warp - 1], s_flag[c][warp - 1]}
                               : Pair{0.0, 0};
    const double pv = __shfl_up_sync(0xffffffffu, local_inc[c].val, 1);
    const int pf = __shfl_up_sync(0xffffffffu, local_inc[c].flag, 1);
    const Pair excl = (lane == 0) ? wp : seg_op(wp, Pair{pv, pf});
    double run = seg_op(chunk_pre, excl).val;

    const int base = block_base + c * (THREADS * ITEMS) + tid * ITEMS;
    float o[ITEMS];
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
      if (flg[c][i]) {
        o[i] = 0.0f;
        run = vals[c][i];
      } else {
        o[i] = static_cast<float>(run);
        run += vals[c][i];
      }
    }
    if (ALIGNED && base + ITEMS <= N) {
#pragma unroll
      for (int i = 0; i < ITEMS; i += 4) {
        *reinterpret_cast<float4*>(output + base + i) =
            make_float4(o[i], o[i + 1], o[i + 2], o[i + 3]);
      }
    } else {
#pragma unroll
      for (int i = 0; i < ITEMS; ++i) {
        if (base + i < N) output[base + i] = o[i];
      }
    }
    chunk_pre = seg_op(chunk_pre,
                       Pair{s_val[c][NWARPS - 1], s_flag[c][NWARPS - 1]});
  }
}

// values, flags, output are device pointers.
//
// Scratch buffers for the lookback (aggregate, inclusive prefix and status per
// block) are cached across calls: cudaMalloc/cudaFree are synchronous and
// would otherwise dominate a short call.
static double* g_agg_val = nullptr;
static int* g_agg_flag = nullptr;
static double* g_pre_val = nullptr;
static int* g_pre_flag = nullptr;
static int* g_status = nullptr;
static int g_capacity = 0;

static void reserve_scratch(int num_blocks) {
  if (num_blocks <= g_capacity) return;
  cudaFree(g_agg_val);
  cudaFree(g_agg_flag);
  cudaFree(g_pre_val);
  cudaFree(g_pre_flag);
  cudaFree(g_status);
  const int cap = num_blocks + num_blocks / 4 + 1024;
  cudaMalloc(&g_agg_val, cap * sizeof(double));
  cudaMalloc(&g_agg_flag, cap * sizeof(int));
  cudaMalloc(&g_pre_val, cap * sizeof(double));
  cudaMalloc(&g_pre_flag, cap * sizeof(int));
  cudaMalloc(&g_status, cap * sizeof(int));
  g_capacity = cap;
}

extern "C" void solve(const float* values, const int* flags, float* output,
                      int N) {
  if (values == nullptr || flags == nullptr || output == nullptr || N <= 0) {
    return;
  }

  const int num_blocks = (N + BLOCK_ELEMS - 1) / BLOCK_ELEMS;
  reserve_scratch(num_blocks);
  double* agg_val = g_agg_val;
  int* agg_flag = g_agg_flag;
  double* pre_val = g_pre_val;
  int* pre_flag = g_pre_flag;
  int* status = g_status;
  cudaMemsetAsync(status, 0, num_blocks * sizeof(int));

  const bool aligned =
      ((reinterpret_cast<uintptr_t>(values) | reinterpret_cast<uintptr_t>(flags) |
        reinterpret_cast<uintptr_t>(output)) &
       15u) == 0;
  if (aligned) {
    segscan_kernel<true><<<num_blocks, THREADS>>>(values, flags, output,
                                                  agg_val, agg_flag, pre_val,
                                                  pre_flag, status, N);
  } else {
    segscan_kernel<false><<<num_blocks, THREADS>>>(values, flags, output,
                                                   agg_val, agg_flag, pre_val,
                                                   pre_flag, status, N);
  }
  cudaDeviceSynchronize();
}
