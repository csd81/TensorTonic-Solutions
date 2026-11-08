#include <cuda_runtime.h>

__inline__ __device__ double warpReduceSum(double val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void monteCarloReduceKernel(const float* __restrict__ y_samples, double* __restrict__ partial_sums, int n_samples) {
    double sum = 0.0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop to accumulate in double precision
    for (int i = idx; i < n_samples; i += stride) {
        sum += static_cast<double>(y_samples[i]);
    }

    // Intra-warp reduction
    sum = warpReduceSum(sum);

    // Store warp sums into shared memory
    __shared__ double shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    if (lane == 0) {
        shared[wid] = sum;
    }
    __syncthreads();

    // Final reduction by the first warp
    sum = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0;
    if (wid == 0) {
        sum = warpReduceSum(sum);
    }

    // Write block's partial sum to global memory
    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = sum;
    }
}

__global__ void finalFinalizeKernel(const double* __restrict__ partial_sums, float* __restrict__ result, int num_blocks, float a, float b, int n_samples) {
    double total_sum = 0.0;

    // Grid has 1 block; sum all block-level partial sums
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        total_sum += partial_sums[i];
    }

    total_sum = warpReduceSum(total_sum);

    __shared__ double shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    if (lane == 0) {
        shared[wid] = total_sum;
    }
    __syncthreads();

    total_sum = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0;
    if (wid == 0) {
        total_sum = warpReduceSum(total_sum);
    }

    if (threadIdx.x == 0) {
        double scale = (static_cast<double>(b) - static_cast<double>(a)) / static_cast<double>(n_samples);
        *result = static_cast<float>(total_sum * scale);
    }
}

// y_samples, result are device pointers
extern "C" void solve(const float* y_samples, float* result, float a, float b, int n_samples) {
    const int blockSize = 256;
    int numBlocks = (n_samples + blockSize - 1) / blockSize;
    if (numBlocks > 1024) {
        numBlocks = 1024;
    }

    double* d_partial_sums = nullptr;
    cudaMalloc(&d_partial_sums, numBlocks * sizeof(double));

    monteCarloReduceKernel<<<numBlocks, blockSize>>>(y_samples, d_partial_sums, n_samples);
    finalFinalizeKernel<<<1, blockSize>>>(d_partial_sums, result, numBlocks, a, b, n_samples);

    cudaFree(d_partial_sums);
}
