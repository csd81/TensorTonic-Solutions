#include <cuda_runtime.h>

__inline__ __device__ double warpReduceSum(double val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void mseKernel(const float* __restrict__ predictions,
                          const float* __restrict__ targets,
                          double* __restrict__ blockSums,
                          int N) {
    double localSum = 0.0;

    // Grid-stride loop to handle large N efficiently
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        double diff = static_cast<double>(predictions[i]) - static_cast<double>(targets[i]);
        localSum += diff * diff;
    }

    // Intra-warp reduction
    localSum = warpReduceSum(localSum);

    // Shared memory to store sums of each warp in the block
    __shared__ double shared[32]; 
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    if (lane == 0) {
        shared[wid] = localSum;
    }
    __syncthreads();

    // First warp reduces the warp sums
    localSum = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0;
    if (wid == 0) {
        localSum = warpReduceSum(localSum);
        if (threadIdx.x == 0) {
            blockSums[blockIdx.x] = localSum;
        }
    }
}

__global__ void finalReduceKernel(const double* __restrict__ blockSums,
                                  float* __restrict__ mse,
                                  int numBlocks,
                                  int N) {
    double totalSum = 0.0;
    for (int i = threadIdx.x; i < numBlocks; i += blockDim.x) {
        totalSum += blockSums[i];
    }

    totalSum = warpReduceSum(totalSum);

    __shared__ double shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    if (lane == 0) {
        shared[wid] = totalSum;
    }
    __syncthreads();

    totalSum = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0;
    if (wid == 0) {
        totalSum = warpReduceSum(totalSum);
        if (threadIdx.x == 0) {
            *mse = static_cast<float>(totalSum / static_cast<double>(N));
        }
    }
}

// predictions, targets, mse are device pointers
extern "C" void solve(const float* predictions, const float* targets, float* mse, int N) {
    if (N <= 0) return;

    const int threadsPerBlock = 256;
    // Cap grid size to ensure optimal occupancy and fast second-stage reduction
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > 1024) {
        blocksPerGrid = 1024;
    }

    double* d_blockSums = nullptr;
    cudaMalloc(&d_blockSums, blocksPerGrid * sizeof(double));

    // Stage 1: Compute squared differences and reduce per block
    mseKernel<<<blocksPerGrid, threadsPerBlock>>>(predictions, targets, d_blockSums, N);

    // Stage 2: Final reduction across block sums and divide by N
    finalReduceKernel<<<1, 256>>>(d_blockSums, mse, blocksPerGrid, N);

    cudaFree(d_blockSums);
}
