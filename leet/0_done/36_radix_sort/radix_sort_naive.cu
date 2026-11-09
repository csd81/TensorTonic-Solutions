#include <cuda_runtime.h>

// Naive LSD radix sort for 32-bit unsigned integers: 4-bit digits, 8 passes.
//
// Each pass is the plain textbook counting sort, all through global memory:
//
//   1. hist_kernel   -- one thread per element, global atomicAdd into the
//                       per-block digit histogram agg[block][digit];
//   2. scan_kernel   -- one thread per digit, sequentially accumulates the
//                       per-block counts to get the destination base of every
//                       (block, digit) bucket (global histogram + naive scan);
//   3. rank_kernel   -- one thread per block, walks the block in order and
//                       writes each element's stable rank inside its digit;
//   4. scatter_kernel-- one thread per element, writes
//                       dst[off[block][digit] + rank[i]] = src[i].
//
// No vector loads, shared-memory tiling, warp ballots, lookback or other
// tricks; the scan is a plain sequential prefix and the rank is a plain
// running counter per block.

#define RADIX_BITS 4
#define RADIX (1 << RADIX_BITS)
#define ELEMS_PER_BLOCK 1024
#define THREADS 256

__global__ void hist_kernel(const unsigned* __restrict__ src,
                            unsigned* __restrict__ agg, int N, int shift) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  const int b = i / ELEMS_PER_BLOCK;
  const int d = (src[i] >> shift) & (RADIX - 1);
  atomicAdd(&agg[b * RADIX + d], 1u);
}

// Exclusive scan of the (block, digit) matrix: the destination base of digit
// d in block b is the number of elements of every smaller digit in every
// block, plus the number of elements of digit d in the preceding blocks.
// One thread per digit; the block dimension is scanned sequentially.
__global__ void scan_kernel(const unsigned* __restrict__ agg,
                            unsigned* __restrict__ off, int nblocks) {
  if (blockIdx.x != 0) return;
  const int d = threadIdx.x;
  if (d >= RADIX) return;

  unsigned base = 0;
  for (int e = 0; e < d; ++e) {
    unsigned total = 0;
    for (int b = 0; b < nblocks; ++b) total += agg[b * RADIX + e];
    base += total;
  }
  unsigned running = base;
  for (int b = 0; b < nblocks; ++b) {
    const unsigned v = agg[b * RADIX + d];
    off[b * RADIX + d] = running;
    running += v;
  }
}

// Stable rank of every element inside its block: one thread per block walks
// the block's elements in order with a 16-entry running counter.
__global__ void rank_kernel(const unsigned* __restrict__ src,
                            int* __restrict__ rank, int N, int shift) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  const int base = b * ELEMS_PER_BLOCK;
  if (base >= N) return;
  int end = base + ELEMS_PER_BLOCK;
  if (end > N) end = N;

  int cnt[RADIX];
#pragma unroll
  for (int d = 0; d < RADIX; ++d) cnt[d] = 0;
  for (int i = base; i < end; ++i) {
    const int d = (src[i] >> shift) & (RADIX - 1);
    rank[i] = cnt[d];
    cnt[d] = cnt[d] + 1;
  }
}

__global__ void scatter_kernel(const unsigned* __restrict__ src,
                               unsigned* __restrict__ dst,
                               const unsigned* __restrict__ off,
                               const int* __restrict__ rank, int N,
                               int shift) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  const int b = i / ELEMS_PER_BLOCK;
  const int d = (src[i] >> shift) & (RADIX - 1);
  dst[off[b * RADIX + d] + rank[i]] = src[i];
}

// input, output are device pointers.
extern "C" void solve(const unsigned int* input, unsigned int* output, int N) {
  if (input == nullptr || output == nullptr || N <= 0) return;

  const int nblocks = (N + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK;
  const size_t nb = static_cast<size_t>(nblocks);

  unsigned* tmp0 = nullptr;
  unsigned* tmp1 = nullptr;
  unsigned* agg = nullptr;
  unsigned* off = nullptr;
  int* rank = nullptr;
  cudaMalloc(&tmp0, static_cast<size_t>(N) * sizeof(unsigned));
  cudaMalloc(&tmp1, static_cast<size_t>(N) * sizeof(unsigned));
  cudaMalloc(&agg, nb * RADIX * sizeof(unsigned));
  cudaMalloc(&off, nb * RADIX * sizeof(unsigned));
  cudaMalloc(&rank, static_cast<size_t>(N) * sizeof(int));

  const int elem_blocks = (N + THREADS - 1) / THREADS;
  const int block_blocks = (nblocks + THREADS - 1) / THREADS;

  constexpr int PASSES = 32 / RADIX_BITS;
  const unsigned* src = input;
  for (int pass = 0; pass < PASSES; ++pass) {
    const int shift = pass * RADIX_BITS;
    unsigned* dst = (pass == PASSES - 1)
                        ? output
                        : ((pass & 1) ? tmp1 : tmp0);

    cudaMemsetAsync(agg, 0, nb * RADIX * sizeof(unsigned));
    hist_kernel<<<elem_blocks, THREADS>>>(src, agg, N, shift);
    scan_kernel<<<1, RADIX>>>(agg, off, nblocks);
    rank_kernel<<<block_blocks, THREADS>>>(src, rank, N, shift);
    scatter_kernel<<<elem_blocks, THREADS>>>(src, dst, off, rank, N, shift);
    src = dst;
  }

  cudaFree(tmp0);
  cudaFree(tmp1);
  cudaFree(agg);
  cudaFree(off);
  cudaFree(rank);
  cudaDeviceSynchronize();
}
