#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// Helper: Warp-level max reduction
__device__ __forceinline__ float warpReduceMax(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Helper: Block-level max reduction
__device__ __forceinline__ float blockReduceMax(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceMax(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -FLT_MAX;
    if (wid == 0) val = warpReduceMax(val);
    return val;
}

// Helper: Warp-level sum reduction
__device__ __forceinline__ float warpReduceSum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Helper: Block-level sum reduction
__device__ __forceinline__ float blockReduceSum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

// Kernel 1: Find the maximum value in the array
__global__ void reduce_max_kernel(const float* input, float* block_maxes, int N) {
    float max_val = -FLT_MAX;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        max_val = fmaxf(max_val, input[i]);
    }

    max_val = blockReduceMax(max_val);
    if (threadIdx.x == 0) {
        block_maxes[blockIdx.x] = max_val;
    }
}

// Kernel 2: Compute exp(x_i - max) and find block sums
__global__ void exp_and_sum_kernel(const float* input, float* output, const float* d_max, float* block_sums, int N) {
    float max_val = *d_max;
    float sum_val = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        float val = expf(input[i] - max_val);
        output[i] = val; // Store intermediate numerator in output array
        sum_val += val;
    }

    sum_val = blockReduceSum(sum_val);
    if (threadIdx.x == 0) {
        block_sums[blockIdx.x] = sum_val;
    }
}

// Helper Kernel: Finalize reduction across block results (launched with 1 block)
__global__ void final_reduce_max_kernel(float* block_maxes, int count) {
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        max_val = fmaxf(max_val, block_maxes[i]);
    }
    max_val = blockReduceMax(max_val);
    if (threadIdx.x == 0) {
        block_maxes[0] = max_val;
    }
}

__global__ void final_reduce_sum_kernel(float* block_sums, int count) {
    float sum_val = 0.0f;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        sum_val += block_sums[i];
    }
    sum_val = blockReduceSum(sum_val);
    if (threadIdx.x == 0) {
        block_sums[0] = sum_val;
    }
}

// Kernel 3: Normalize each element by dividing by the total sum
__global__ void normalize_kernel(float* output, const float* d_sum, int N) {
    float inv_sum = 1.0f / (*d_sum);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        output[i] *= inv_sum;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // Cap grid size to ensure efficient grid-stride loops and small intermediate buffers
    if (blocksPerGrid > 1024) blocksPerGrid = 1024;

    // Separate max and sum scratch buffers: exp_and_sum_kernel reads the
    // global max while block 0 writes its partial sum. Sharing one buffer
    // would race and corrupt the max used by other blocks.
    float* d_max = nullptr;
    float* d_sums = nullptr;
    cudaMalloc(&d_max, blocksPerGrid * sizeof(float));
    cudaMalloc(&d_sums, blocksPerGrid * sizeof(float));

    // Step 1: Global Max
    reduce_max_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, d_max, N);
    final_reduce_max_kernel<<<1, threadsPerBlock>>>(d_max, blocksPerGrid);

    // Step 2: Numerators & Global Sum
    exp_and_sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, d_max, d_sums, N);
    final_reduce_sum_kernel<<<1, threadsPerBlock>>>(d_sums, blocksPerGrid);

    // Step 3: Normalize
    normalize_kernel<<<blocksPerGrid, threadsPerBlock>>>(output, d_sums, N);

    cudaDeviceSynchronize();
    cudaFree(d_max);
    cudaFree(d_sums);
}
