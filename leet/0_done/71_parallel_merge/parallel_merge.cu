#include <cuda_runtime.h>

#include <cstdint>

// Merge two sorted arrays A [M] and B [N] into C [M+N].
//
// At the benchmark (M = N = 25,000,000) the kernel reads 200 MB and writes
// 200 MB, so the floor is ~4.9 ms at this GPU's ~82 GB/s read+write rate.
// The starter runs a binary search per output element (50 M searches, each a
// ~25-probe random walk through the 100 MB arrays) and measures 45.8 ms
// (8.7 GB/s of the 400 MB).
//
// This version is the classical merge-path partition:
//
//   1. partition_kernel computes, for every block boundary (one per TILE
//      outputs), the merge-path split (i, j) with i+j = diagonal via a binary
//      search.  This is one kernel of ~16 K independent searches, so the
//      latency is hidden across threads.  (Computing the splits inside the
//      merge blocks instead serialises a 25-probe dependent chain in front of
//      every tile: that variant measured 22 ms.)
//   2. merge_tile_kernel gives each block one output tile of TILE = 3072
//      elements.  The block loads its A-range and B-range into shared memory
//      with coalesced global reads, each thread finds the split of its own
//      12-output sub-range with a binary search inside the shared tile, then
//      merges its 12 values into registers and writes them as three float4
//      stores (coalesced).
//
// Measured 6.47 ms (61.8 GB/s effective), 7.1x faster than the starter.
// TILE sweep: 512 -> 10.7 ms, 1024 -> 8.0, 2048 -> 6.6, 3072 -> 6.47,
// 4096 -> 8.7.

namespace {

constexpr int TILE = 3072;   // outputs per block
constexpr int TPB = 256;
constexpr int ITEMS = TILE / TPB;  // 12, must be a multiple of 4

// Merge-path split: the number of A-elements among the first d outputs.
__device__ __forceinline__ int merge_path_split(const float* A, int M,
                                                const float* B, int N, int d) {
  int low = max(0, d - N), high = min(d, M);
  while (low < high) {
    const int mid = low + ((high - low) >> 1);
    const int jb = d - 1 - mid;
    if (__ldg(&A[mid]) <= __ldg(&B[jb])) low = mid + 1; else high = mid;
  }
  return low;
}

__global__ void partition_kernel(const float* __restrict__ A, int M,
                                 const float* __restrict__ B, int N,
                                 int* __restrict__ parts, int nparts, int total) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= nparts) return;
  parts[b] = merge_path_split(A, M, B, N, min(b * TILE, total));
}

__global__ void __launch_bounds__(TPB) merge_tile_kernel(
    const float* __restrict__ A, int M, const float* __restrict__ B, int N,
    float* __restrict__ C, const int* __restrict__ parts, int total) {
  __shared__ float sA[TILE], sB[TILE];

  const int b = blockIdx.x;
  const int d0 = b * TILE;
  const int count = min(TILE, total - d0);
  const int i0 = __ldg(&parts[b]);
  const int i1 = __ldg(&parts[b + 1]);
  const int j0 = d0 - i0;
  const int j1 = d0 + count - i1;
  const int tid = threadIdx.x;

  const int a_len = i1 - i0, b_len = j1 - j0;
  for (int idx = tid; idx < a_len; idx += TPB) sA[idx] = __ldg(&A[i0 + idx]);
  for (int idx = tid; idx < b_len; idx += TPB) sB[idx] = __ldg(&B[j0 + idx]);
  __syncthreads();

  const int ld = tid * ITEMS;
  if (ld >= count) return;

  int low = max(0, ld - b_len), high = min(ld, a_len);
  while (low < high) {
    const int mid = low + ((high - low) >> 1);
    const int jb = ld - 1 - mid;
    if (sA[mid] <= sB[jb]) low = mid + 1; else high = mid;
  }
  int i = low, j = ld - low;

  float res[ITEMS];
#pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    const float va = (i < a_len) ? sA[i] : INFINITY;
    const float vb = (j < b_len) ? sB[j] : INFINITY;
    if (va <= vb) {
      res[k] = va;
      ++i;
    } else {
      res[k] = vb;
      ++j;
    }
  }

  float* dst = C + d0 + ld;
  if (ld + ITEMS <= count) {
#pragma unroll
    for (int v = 0; v < ITEMS; v += 4) {
      *reinterpret_cast<float4*>(dst + v) =
          make_float4(res[v], res[v + 1], res[v + 2], res[v + 3]);
    }
  } else {
    const int end = count - ld;
    for (int k = 0; k < end; ++k) dst[k] = res[k];
  }
}

int* g_parts = nullptr;
int g_parts_cap = 0;

}  // namespace

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
  if (M < 0 || N < 0) return;
  const int total = M + N;
  if (total == 0) return;
  if (M == 0) {
    cudaMemcpy(C, B, (size_t)N * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    return;
  }
  if (N == 0) {
    cudaMemcpy(C, A, (size_t)M * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    return;
  }

  const int nblocks = (total + TILE - 1) / TILE;
  const int nparts = nblocks + 1;
  if (nparts > g_parts_cap) {
    if (g_parts) cudaFree(g_parts);
    cudaMalloc(&g_parts, (size_t)nparts * sizeof(int));
    g_parts_cap = nparts;
  }

  partition_kernel<<<(nparts + TPB - 1) / TPB, TPB>>>(A, M, B, N, g_parts,
                                                      nparts, total);
  merge_tile_kernel<<<nblocks, TPB>>>(A, M, B, N, C, g_parts, total);
  cudaDeviceSynchronize();
}
