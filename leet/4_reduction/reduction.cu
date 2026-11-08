#include <cuda_runtime.h>

__device__ __forceinline__ float warpReduceSum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val) {
    // 1024 max threads per block / 32 threads per warp = 32 warps max
    __shared__ float shared[32]; 
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    // Reduce within each warp using registers
    val = warpReduceSum(val);

    // Write reduced warp values to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Read from shared memory only if that warp existed
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0f;

    // Final reduce within the first warp
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

__global__ void reduceKernel(const float* __restrict__ input, float* __restrict__ output, int N) {
    float sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop to handle arbitrary N up to 100,000,000
    for (int i = idx; i < N; i += stride) {
        sum += input[i];
    }

    // Reduce across the entire thread block
    sum = blockReduceSum(sum);

    // Atomically add each block's partial sum to the final output
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    // Initialize output memory to 0.0f before atomic accumulation
    cudaMemset(output, 0, sizeof(float));

    const int threadsPerBlock = 256;
    // Cap blocks to balance occupancy and minimize atomicAdd contention
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > 1024) {
        blocksPerGrid = 1024;
    }

    reduceKernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
