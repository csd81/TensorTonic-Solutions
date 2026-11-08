#include <cuda_runtime.h>

// Reduces `val` across the whole block; the block sum ends up in thread 0.
__device__ __forceinline__ float block_reduce_sum(float val) {
    // 1) Intra-warp tree: 32 lanes -> lane 0, done entirely in registers.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffffu, val, offset);

    // 2) Stage one partial per warp through shared memory.
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = val;
    __syncthreads();

    // 3) First warp reduces the per-warp partials.
    const int num_warps = (blockDim.x + 31) >> 5;
    val = (lane < num_warps) ? warp_sums[lane] : 0.0f;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffffu, val, offset);

    return val;  // valid in lane 0
}

__global__ void dot_kernel(const float* A, const float* B, float* result, int N) {
    // Fused map + reduce: each product stays in a register, never in memory.
    // Grid-stride form so the kernel is also correct if the launch is ever
    // sized to fewer blocks than ceil(N / blockDim.x).
    float local = 0.0f;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N;
         idx += blockDim.x * gridDim.x) {
        local = fmaf(A[idx], B[idx], local);
    }

    // Block-level tree reduction, then one atomic per block (not per thread).
    float block_sum = block_reduce_sum(local);
    if (threadIdx.x == 0) {
        atomicAdd(result, block_sum);
    }
}

extern "C" void solve(const float* A, const float* B, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    // Cap the grid: there is one atomicAdd per block, so launching ceil(N/256)
    // blocks (39k+ at N = 1e7) serializes on the single accumulator. With the
    // grid-stride loop above, a few thousand blocks already saturate DRAM.
    if (blocks > 2048) blocks = 2048;
    cudaMemset(result, 0, sizeof(float));
    dot_kernel<<<blocks, threads>>>(A, B, result, N);
    cudaDeviceSynchronize();
}
