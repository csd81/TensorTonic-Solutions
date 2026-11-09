#include <cuda_runtime.h>

#include <cstdint>

// Count the occurrences of K in an array of N int32 values.
//
// This is a pure streaming problem: 400 MB of reads at N = 100,000,000, so
// the kernel is built around vectorized int4 (16-byte) loads and a
// grid-stride loop, with one atomicAdd per block for the final total.
// The result lands in output[0], which solve() zeroes first.

#define THREADS 256
#define MAX_BLOCKS 4096

__global__ void count_vec4_kernel(const int4* __restrict__ input,
                                  int* __restrict__ output, int n4, int tail,
                                  const int* __restrict__ tail_ptr, int K) {
  __shared__ int s_warp[THREADS / 32];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int stride = gridDim.x * blockDim.x;

  int count = 0;
  for (int i = blockIdx.x * blockDim.x + tid; i < n4; i += stride) {
    const int4 v = input[i];
    count += (v.x == K) + (v.y == K) + (v.z == K) + (v.w == K);
  }
  // Leftover elements (N % 4) are handled by the first threads of block 0.
  if (blockIdx.x == 0 && tid < tail) {
    count += (tail_ptr[tid] == K);
  }

#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    count += __shfl_down_sync(0xffffffffu, count, off);
  }
  if (lane == 0) s_warp[warp] = count;
  __syncthreads();

  if (warp == 0) {
    int v = (lane < THREADS / 32) ? s_warp[lane] : 0;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if (lane == 0) atomicAdd(output, v);
  }
}

__global__ void count_scalar_kernel(const int* __restrict__ input,
                                    int* __restrict__ output, int N, int K) {
  __shared__ int s_warp[THREADS / 32];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int stride = gridDim.x * blockDim.x;

  int count = 0;
  for (int i = blockIdx.x * blockDim.x + tid; i < N; i += stride) {
    count += (input[i] == K);
  }

#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    count += __shfl_down_sync(0xffffffffu, count, off);
  }
  if (lane == 0) s_warp[warp] = count;
  __syncthreads();

  if (warp == 0) {
    int v = (lane < THREADS / 32) ? s_warp[lane] : 0;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if (lane == 0) atomicAdd(output, v);
  }
}

// input, output are device pointers; the result is written to output[0].
extern "C" void solve(const int* input, int* output, int N, int K) {
  if (input == nullptr || output == nullptr || N <= 0) return;

  cudaMemsetAsync(output, 0, sizeof(int));

  const int n4 = N >> 2;
  int blocks = (n4 + THREADS - 1) / THREADS;
  if (blocks > MAX_BLOCKS) blocks = MAX_BLOCKS;
  if (blocks < 1) blocks = 1;

  if ((reinterpret_cast<uintptr_t>(input) & 15u) == 0) {
    const int tail = N & 3;
    count_vec4_kernel<<<blocks, THREADS>>>(
        reinterpret_cast<const int4*>(input), output, n4, tail,
        input + (n4 << 2), K);
  } else {
    count_scalar_kernel<<<blocks, THREADS>>>(input, output, N, K);
  }
  cudaDeviceSynchronize();
}
