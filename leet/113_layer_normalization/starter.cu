#include <cuda_runtime.h>
#include <math.h>

// Warp-level reduction helper for sum
__device__ inline float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction helper for sum using shared memory
__device__ inline float blockReduceSum(float val) {
    static __shared__ float shared[32]; // Maximum 32 warps per block (1024 threads)
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // Read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;

    if (wid == 0) val = warpReduceSum(val);
    return val;
}

__global__ void layerNormKernel(const float* input, const float* weight, const float* bias, float* output,
                                int N, int C, float eps) {
    // Each block handles exactly one row (sample i)
    int i = blockIdx.x;
    if (i >= N) return;

    // Phase 1: Compute the Mean
    float sum = 0.0f;
    for (int j = threadIdx.x; j < C; j += blockDim.x) {
        sum += input[i * C + j];
    }
    float mean = blockReduceSum(sum) / C;

    // Broadcast the mean to all threads via shared memory
    __shared__ float s_mean;
    if (threadIdx.x == 0) {
        s_mean = mean;
    }
    __syncthreads();
    mean = s_mean;

    // Phase 2: Compute the Variance
    float sq_sum = 0.0f;
    for (int j = threadIdx.x; j < C; j += blockDim.x) {
        float diff = input[i * C + j] - mean;
        sq_sum += diff * diff;
    }
    float var = blockReduceSum(sq_sum) / C;

    // Broadcast the reciprocal standard deviation to all threads
    __shared__ float s_rstd;
    if (threadIdx.x == 0) {
        s_rstd = rsqrtf(var + eps);
    }
    __syncthreads();
    float rstd = s_rstd;

    // Phase 3: Normalize and scale/shift
    for (int j = threadIdx.x; j < C; j += blockDim.x) {
        int idx = i * C + j;
        output[idx] = weight[j] * ((input[idx] - mean) * rstd) + bias[j];
    }
}

extern "C" void solve(const float* input, const float* weight, const float* bias, float* output,
                      int N, int C, float eps) {
    // 256 threads per block balances occupancy and shared memory requirements
    int threadsPerBlock = 256;
    int blocksPerGrid = N;

    layerNormKernel<<<blocksPerGrid, threadsPerBlock>>>(input, weight, bias, output, N, C, eps);
}
