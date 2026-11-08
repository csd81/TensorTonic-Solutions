#include <cuda_runtime.h>

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float blockReduceSum(float val) {
    static __shared__ float shared[32]; 
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;

    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

__global__ void dotProductKernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ result, int N) {
    float sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop to accumulate element-wise products per thread
    for (int i = idx; i < N; i += stride) {
        sum += A[i] * B[i];
    }

    // Reduce within the thread block
    sum = blockReduceSum(sum);

    // First thread in each block adds its block's partial sum to the global result
    if (threadIdx.x == 0) {
        atomicAdd(result, sum);
    }
}

// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) {
    // Initialize the device output pointer to 0.0f before atomic accumulation
    cudaMemset(result, 0, sizeof(float));

    int threadsPerBlock = 256;
    // Cap grid size to maximize SM occupancy without excessive atomic contention
    int maxBlocks = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > maxBlocks) {
        blocksPerGrid = maxBlocks;
    }

    dotProductKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, result, N);
}
