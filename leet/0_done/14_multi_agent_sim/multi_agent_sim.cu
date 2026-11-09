#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

// Boids neighbourhood step: for every agent, average the velocities of the
// agents within radius r = 5 and move it by alpha = 0.05 of the difference.
//
// The starter is a straight O(N^2) sweep: one thread per agent, every agent
// compared against every other, tiled through shared memory.  At N = 10,000
// that is 10^8 distance tests, and it measures 1.34-1.51 ms regardless of how
// the agents are spread out.
//
// Two changes:
//
// 1. Cheaper comparisons (always).  The shared tile keeps positions and
//    velocities in separate float2 arrays, so the inner loop loads 8 bytes
//    per candidate instead of the full 16-byte float4 and only touches the
//    velocity when the radius test passes.  Every agent also counts itself
//    (distance 0 passes) and subtracts once at the end, which removes the
//    per-candidate `i != j` branch.  That alone is 1.55-1.75x.
//
// 2. A spatial hash (the big win).  With a fixed radius the O(N^2) sweep is
//    wasted work: agents are binned into cells of side r, and each agent only
//    tests the agents in its own cell and the eight neighbours -- any agent
//    within r must be in one of those, because a cell is exactly r wide.  The
//    binning is a counting sort over a hashed cell index (no bounding box
//    needed, so it works for any coordinate range), and the query kernel
//    de-duplicates hashed buckets so a collision never counts an agent twice.
//    At N = 10,000 the whole step becomes a handful of tiny kernels.
//
// Small N keeps the direct sweep, where the binning would not pay for its
// launch overhead.

#define R2 25.0f          // r = 5, squared
#define INV_R 0.2f        // 1 / r
#define BS 256
#define SCAN_THREADS 256
#define DIRECT_LIMIT 4096
#define MAX_BUCKETS (1 << 20)

__device__ __forceinline__ unsigned cell_hash(int cx, int cy, unsigned mask) {
  return (static_cast<unsigned>(cx) * 73856093u ^
          static_cast<unsigned>(cy) * 19349663u) &
         mask;
}

// ---------------------------------------------------------- direct O(N^2)

__global__ void boids_direct_kernel(const float4* __restrict__ ag,
                                    float4* __restrict__ out, int N) {
  __shared__ float2 tp[BS];
  __shared__ float2 tv[BS];
  const int i = blockIdx.x * BS + threadIdx.x;
  float2 me = make_float2(0.0f, 0.0f);
  float2 mv = make_float2(0.0f, 0.0f);
  if (i < N) {
    const float4 a = ag[i];
    me = make_float2(a.x, a.y);
    mv = make_float2(a.z, a.w);
  }
  float sx = 0.0f, sy = 0.0f;
  int cnt = 0;
  const int nt = (N + BS - 1) / BS;
  for (int t = 0; t < nt; ++t) {
    const int jl = t * BS + threadIdx.x;
    if (jl < N) {
      const float4 a = ag[jl];
      tp[threadIdx.x] = make_float2(a.x, a.y);
      tv[threadIdx.x] = make_float2(a.z, a.w);
    }
    __syncthreads();
    const int lim = (N - t * BS < BS) ? (N - t * BS) : BS;
#pragma unroll 16
    for (int k = 0; k < lim; ++k) {
      const float dx = me.x - tp[k].x;
      const float dy = me.y - tp[k].y;
      if (dx * dx + dy * dy < R2) {
        sx += tv[k].x;
        sy += tv[k].y;
        ++cnt;
      }
    }
    __syncthreads();
  }
  if (i < N) {
    sx -= mv.x;  // self always passes the radius test
    sy -= mv.y;
    --cnt;
    float avx = mv.x, avy = mv.y;
    if (cnt > 0) {
      avx = sx / static_cast<float>(cnt);
      avy = sy / static_cast<float>(cnt);
    }
    const float nvx = mv.x + 0.05f * (avx - mv.x);
    const float nvy = mv.y + 0.05f * (avy - mv.y);
    out[i] = make_float4(me.x + nvx, me.y + nvy, nvx, nvy);
  }
}

// ------------------------------------------------------- spatial hash path

__global__ void count_kernel(const float4* __restrict__ ag, int N,
                             unsigned mask, int* __restrict__ counts) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    const float4 a = ag[i];
    const int cx = static_cast<int>(floorf(a.x * INV_R));
    const int cy = static_cast<int>(floorf(a.y * INV_R));
    atomicAdd(&counts[cell_hash(cx, cy, mask)], 1);
  }
}

// Two-level exclusive scan of the bucket counts: the per-chunk prefixes are
// written to `starts`/`cursor`, the chunk totals are scanned in one block, and
// the consumers add the chunk prefix themselves (a single block over the whole
// bucket array used to cost 0.22 ms at 65k buckets).
#define SCAN_CHUNK 4096
#define SCAN_ITEMS (SCAN_CHUNK / SCAN_THREADS)

__global__ void scan_blocks_kernel(const int* __restrict__ counts,
                                   int* __restrict__ starts,
                                   int* __restrict__ cursor,
                                   int* __restrict__ chunk_sums, int nb) {
  __shared__ int wsum[SCAN_THREADS / 32];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int base = blockIdx.x * SCAN_CHUNK + tid * SCAN_ITEMS;

  int run[SCAN_ITEMS];
  int s = 0;
#pragma unroll
  for (int j = 0; j < SCAN_ITEMS; ++j) {
    const int idx = base + j;
    const int v = (idx < nb) ? counts[idx] : 0;
    run[j] = s;
    s += v;
  }
  int x = s;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const int y = __shfl_up_sync(0xFFFFFFFFu, x, off);
    if (lane >= off) x += y;
  }
  if (lane == 31) wsum[warp] = x;
  __syncthreads();
  if (warp == 0) {
    int w = (lane < SCAN_THREADS / 32) ? wsum[lane] : 0;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const int y = __shfl_up_sync(0xFFFFFFFFu, w, off);
      if (lane >= off) w += y;
    }
    if (lane < SCAN_THREADS / 32) wsum[lane] = w;
  }
  __syncthreads();
  const int wbase = (warp > 0) ? wsum[warp - 1] : 0;
  const int tbase = wbase + (x - s);
#pragma unroll
  for (int j = 0; j < SCAN_ITEMS; ++j) {
    const int idx = base + j;
    if (idx < nb) {
      starts[idx] = run[j] + tbase;
      cursor[idx] = run[j] + tbase;
    }
  }
  if (tid == SCAN_THREADS - 1) chunk_sums[blockIdx.x] = wbase + x;
}

__global__ void scan_sums_kernel(int* __restrict__ chunk_sums, int m) {
  __shared__ int wsum[SCAN_THREADS / 32];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int runs = (m + SCAN_THREADS - 1) / SCAN_THREADS;
  const int base = tid * runs;

  int s = 0;
  for (int j = 0; j < runs; ++j) {
    if (base + j < m) s += chunk_sums[base + j];
  }
  int x = s;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const int y = __shfl_up_sync(0xFFFFFFFFu, x, off);
    if (lane >= off) x += y;
  }
  if (lane == 31) wsum[warp] = x;
  __syncthreads();
  if (warp == 0) {
    int w = (lane < SCAN_THREADS / 32) ? wsum[lane] : 0;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const int y = __shfl_up_sync(0xFFFFFFFFu, w, off);
      if (lane >= off) w += y;
    }
    if (lane < SCAN_THREADS / 32) wsum[lane] = w;
  }
  __syncthreads();
  const int wbase = (warp > 0) ? wsum[warp - 1] : 0;
  int acc = wbase + (x - s);
  for (int j = 0; j < runs; ++j) {
    if (base + j < m) {
      const int v = chunk_sums[base + j];
      chunk_sums[base + j] = acc;
      acc += v;
    }
  }
}

__global__ void scatter_kernel(const float4* __restrict__ ag, int N,
                               unsigned mask, int* __restrict__ cursor,
                               const int* __restrict__ chunk_sums,
                               float2* __restrict__ ps,
                               float2* __restrict__ vs) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    const float4 a = ag[i];
    const int cx = static_cast<int>(floorf(a.x * INV_R));
    const int cy = static_cast<int>(floorf(a.y * INV_R));
    const unsigned h = cell_hash(cx, cy, mask);
    const int p = atomicAdd(&cursor[h], 1) + chunk_sums[h >> 12];
    ps[p] = make_float2(a.x, a.y);
    vs[p] = make_float2(a.z, a.w);
  }
}

__global__ void query_kernel(const float4* __restrict__ ag,
                             float4* __restrict__ out,
                             const int* __restrict__ starts,
                             const int* __restrict__ counts,
                             const float2* __restrict__ ps,
                             const float2* __restrict__ vs,
                             const int* __restrict__ chunk_sums, int N,
                             unsigned mask) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  const float4 a = ag[i];
  const float2 me = make_float2(a.x, a.y);
  const float2 mv = make_float2(a.z, a.w);
  const int cx = static_cast<int>(floorf(me.x * INV_R));
  const int cy = static_cast<int>(floorf(me.y * INV_R));

  // The nine neighbouring cells, with duplicate hashed buckets removed so a
  // collision can never count an agent twice.
  unsigned hs[9];
  int nh = 0;
#pragma unroll
  for (int oy = -1; oy <= 1; ++oy) {
#pragma unroll
    for (int ox = -1; ox <= 1; ++ox) {
      const unsigned h = cell_hash(cx + ox, cy + oy, mask);
      bool dup = false;
      for (int q = 0; q < nh; ++q) {
        if (hs[q] == h) dup = true;
      }
      if (!dup) hs[nh++] = h;
    }
  }

  float sx = 0.0f, sy = 0.0f;
  int cnt = 0;
  for (int q = 0; q < nh; ++q) {
    const int s = starts[hs[q]] + chunk_sums[hs[q] >> 12];
    const int e = s + counts[hs[q]];
    for (int p = s; p < e; ++p) {
      const float dx = me.x - ps[p].x;
      const float dy = me.y - ps[p].y;
      if (dx * dx + dy * dy < R2) {
        sx += vs[p].x;
        sy += vs[p].y;
        ++cnt;
      }
    }
  }

  sx -= mv.x;  // self
  sy -= mv.y;
  --cnt;
  float avx = mv.x, avy = mv.y;
  if (cnt > 0) {
    avx = sx / static_cast<float>(cnt);
    avy = sy / static_cast<float>(cnt);
  }
  const float nvx = mv.x + 0.05f * (avx - mv.x);
  const float nvy = mv.y + 0.05f * (avy - mv.y);
  out[i] = make_float4(me.x + nvx, me.y + nvy, nvx, nvy);
}

// ---------------------------------------------------------------- host side

static int* g_counts = nullptr;
static int* g_starts = nullptr;
static int* g_cursor = nullptr;
static float2* g_ps = nullptr;
static float2* g_vs = nullptr;
static int* g_chunk_sums = nullptr;
static int g_cap_buckets = 0;
static int g_cap_n = 0;

static void reserve(int nb, int n) {
  if (nb <= g_cap_buckets && n <= g_cap_n) return;
  if (nb > g_cap_buckets) {
    cudaFree(g_counts);
    cudaFree(g_starts);
    cudaFree(g_cursor);
    cudaFree(g_chunk_sums);
    const int cap = nb + nb / 4;
    cudaMalloc(&g_counts, static_cast<size_t>(cap) * sizeof(int));
    cudaMalloc(&g_starts, static_cast<size_t>(cap) * sizeof(int));
    cudaMalloc(&g_cursor, static_cast<size_t>(cap) * sizeof(int));
    cudaMalloc(&g_chunk_sums, static_cast<size_t>(cap / SCAN_CHUNK + 2) *
                                  sizeof(int));
    g_cap_buckets = cap;
  }
  if (n > g_cap_n) {
    cudaFree(g_ps);
    cudaFree(g_vs);
    const int cap = n + n / 4;
    cudaMalloc(&g_ps, static_cast<size_t>(cap) * sizeof(float2));
    cudaMalloc(&g_vs, static_cast<size_t>(cap) * sizeof(float2));
    g_cap_n = cap;
  }
}

// agents, agents_next are device pointers, 4 floats per agent.
extern "C" void solve(const float* agents, float* agents_next, int N) {
  if (N <= 0) return;
  const float4* ag = reinterpret_cast<const float4*>(agents);
  float4* out = reinterpret_cast<float4*>(agents_next);
  const int blocks = (N + BS - 1) / BS;

  if (N < DIRECT_LIMIT) {
    boids_direct_kernel<<<blocks, BS>>>(ag, out, N);
    return;
  }

  int nb = 1024;
  while (nb < 4 * N && nb < MAX_BUCKETS) nb <<= 1;
  reserve(nb, N);
  const unsigned mask = static_cast<unsigned>(nb - 1);

  cudaMemsetAsync(g_counts, 0, static_cast<size_t>(nb) * sizeof(int));
  const int nchunks = (nb + SCAN_CHUNK - 1) / SCAN_CHUNK;
  count_kernel<<<blocks, BS>>>(ag, N, mask, g_counts);
  scan_blocks_kernel<<<nchunks, SCAN_THREADS>>>(g_counts, g_starts, g_cursor,
                                                g_chunk_sums, nb);
  scan_sums_kernel<<<1, SCAN_THREADS>>>(g_chunk_sums, nchunks);
  scatter_kernel<<<blocks, BS>>>(ag, N, mask, g_cursor, g_chunk_sums, g_ps,
                                 g_vs);
  query_kernel<<<blocks, BS>>>(ag, out, g_starts, g_counts, g_ps, g_vs,
                               g_chunk_sums, N, mask);
}
