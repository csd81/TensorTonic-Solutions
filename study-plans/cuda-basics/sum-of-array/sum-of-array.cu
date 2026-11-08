#include <cuda_runtime.h>

// Helper function to perform a sum reduction across a 32-thread warp
__inline__ __device__ float warpReduceSum(float val) {
    // Each thread pulls the value from the thread 'offset' lanes ahead
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

__global__ void sum_kernel(const float* input, float* result, int N) {
    float sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 1. Each thread loads a single element (if within bounds)
    if (idx < N) {
        sum = input[idx];
    }

    // 2. Reduce sums within each warp using fast register shuffles
    sum = warpReduceSum(sum);

    // 3. Allocate shared memory for partial warp sums
    // A block can have max 1024 threads, meaning max 32 warps.
    __shared__ float warpSums[32];

    int laneIdx = threadIdx.x % 32;
    int warpIdx = threadIdx.x / 32;

    // 4. The first thread of each warp writes its partial sum to shared memory
    if (laneIdx == 0) {
        warpSums[warpIdx] = sum;
    }

    // Ensure all warp partials are fully written before reading
    __syncthreads();

    // 5. Let the first warp reduce the partial sums collected from shared memory
    if (warpIdx == 0) {
        int numWarps = (blockDim.x + 31) / 32; // Active warps in this block
        
        // Read valid warp sums, pad the rest of the lanes with 0.0f
        sum = (laneIdx < numWarps) ? warpSums[laneIdx] : 0.0f;
        
        // Final shuffle reduction for the block sum
        sum = warpReduceSum(sum);

        // 6. Thread 0 adds the entire block's sum atomically to the global result
        if (threadIdx.x == 0) {
            atomicAdd(result, sum);
        }
    }
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    cudaMemset(result, 0, sizeof(float));
    sum_kernel<<<blocks, threads>>>(input, result, N);
    cudaDeviceSynchronize();
}