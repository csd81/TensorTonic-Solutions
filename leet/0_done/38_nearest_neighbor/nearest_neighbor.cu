// Nearest-neighbor search (challenge 38).
//
// For each point q, find the index j != q with the smallest squared distance.
// N <= 100,000; benchmark N = 10,000.
//
// Strategy
// --------
// The brute-force test is O(N^2) = 1e8 distance evaluations at N = 10,000.
// Every lane of a warp reads the *same* candidate each iteration, so the
// shared-memory loads are broadcasts: bank layout is irrelevant here, and an
// AoS tile measures within ~1% of the SoA tile kept below (1.317 vs 1.306 ms
// in a same-process interleaved run).  What actually matters is the
// instruction count per distance evaluation, so the kernel is
// instruction-throughput bound, not bandwidth bound:
//
//  1. Two queries per thread (TQ = 2): each candidate coordinates pair
//     loaded from shared memory is reused for two distance evaluations,
//     halving the shared-load instructions per pair and raising the ILP of
//     the inner loop.  This is the main gain.
//  2. Hoist the self-index test: a thread's queries occupy the tiny range
//     [q0, q0+1].  For tiles that do not intersect it (39 of 40 tiles at
//     N = 10,000) the inner loop runs without the `j != q` comparison,
//     worth ~6%.
//  3. 128-thread blocks: more (smaller) blocks per SM fill the 6 SMs
//     better than the starter's 256-thread blocks; 256 threads with TQ = 2
//     costs 12%.
//
// Measured on the Quadro P2000 (N = 10,000, 128 threads/block, same-process
// interleaved median of 40 runs): 1.28-1.31 ms = 77-78 Gpairs/s, versus
// 1.54-1.57 ms for the starter's one-query AoS-tile kernel (1.20x).
// At N = 100,000: 122.5 ms (81.6 Gpairs/s) versus 147.8 ms (1.21x).

#include <cuda_runtime.h>
#include <float.h>

namespace {

constexpr int kTile = 256;             // points staged in shared memory
constexpr int kQueriesPerThread = 2;   // TQ
constexpr int kThreads = 128;          // -> 256 queries per block

__global__ void nearest_neighbor_kernel(const float* __restrict__ points,
                                        int* __restrict__ indices, int N) {
  __shared__ float sx[kTile], sy[kTile], sz[kTile];

  const int q0 = (blockIdx.x * blockDim.x + threadIdx.x) * kQueriesPerThread;

  // Two queries per thread, held in registers for the whole kernel.
  float qx[kQueriesPerThread], qy[kQueriesPerThread], qz[kQueriesPerThread];
  float best[kQueriesPerThread];
  int best_idx[kQueriesPerThread];
#pragma unroll
  for (int u = 0; u < kQueriesPerThread; ++u) {
    const int q = q0 + u;
    qx[u] = qy[u] = qz[u] = 0.0f;
    best[u] = FLT_MAX;
    best_idx[u] = -1;
    if (q < N) {
      qx[u] = points[3 * q];
      qy[u] = points[3 * q + 1];
      qz[u] = points[3 * q + 2];
    }
  }

  for (int tile_start = 0; tile_start < N; tile_start += kTile) {
    // Cooperative AoS -> SoA staging; works for any blockDim.x <= kTile.
    for (int t = threadIdx.x; t < kTile; t += blockDim.x) {
      const int li = tile_start + t;
      if (li < N) {
        sx[t] = points[3 * li];
        sy[t] = points[3 * li + 1];
        sz[t] = points[3 * li + 2];
      }
    }
    __syncthreads();

    const int tile_end = min(kTile, N - tile_start);
    const bool tile_has_self =
        (q0 + kQueriesPerThread - 1 >= tile_start) &&
        (q0 < tile_start + tile_end);

    if (!tile_has_self) {
      // Fast path: no candidate can be one of this thread's own queries.
      for (int k = 0; k < tile_end; ++k) {
        const float px = sx[k], py = sy[k], pz = sz[k];
        const int j = tile_start + k;
#pragma unroll
        for (int u = 0; u < kQueriesPerThread; ++u) {
          if (q0 + u < N) {
            const float dx = qx[u] - px;
            const float dy = qy[u] - py;
            const float dz = qz[u] - pz;
            const float d = dx * dx + dy * dy + dz * dz;
            if (d < best[u]) {
              best[u] = d;
              best_idx[u] = j;
            }
          }
        }
      }
    } else {
      // Slow path: this tile contains this thread's own index/queries.
      for (int k = 0; k < tile_end; ++k) {
        const float px = sx[k], py = sy[k], pz = sz[k];
        const int j = tile_start + k;
#pragma unroll
        for (int u = 0; u < kQueriesPerThread; ++u) {
          if (q0 + u < N) {
            const float dx = qx[u] - px;
            const float dy = qy[u] - py;
            const float dz = qz[u] - pz;
            const float d = dx * dx + dy * dy + dz * dz;
            if (j != q0 + u && d < best[u]) {
              best[u] = d;
              best_idx[u] = j;
            }
          }
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int u = 0; u < kQueriesPerThread; ++u) {
    const int q = q0 + u;
    if (q < N) indices[q] = best_idx[u];
  }
}

}  // namespace

// points and indices are device pointers.
extern "C" void solve(const float* points, int* indices, int N) {
  if (N <= 0) return;
  const int per_block = kThreads * kQueriesPerThread;
  const int blocks = (N + per_block - 1) / per_block;
  nearest_neighbor_kernel<<<blocks, kThreads>>>(points, indices, N);
}
