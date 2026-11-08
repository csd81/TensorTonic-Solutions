#include <cuda_runtime.h>
#include <math.h>

// Helper function for warp-level sum reduction
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel 1: Compute the sum of squares across all elements
__global__ void sumSquaresKernel(const float* __restrict__ input, float* __restrict__ sum_sq, int N) {
    float local_sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop to accumulate squares
    for (int i = idx; i < N; i += stride) {
        float val = input[i];
        local_sum += val * val;
    }

    // Reduce within each warp
    local_sum = warpReduceSum(local_sum);

    // Store warp sums into shared memory
    __shared__ float shared[32]; // Max 32 warps per block (1024 threads)
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    if (lane == 0) {
        shared[wid] = local_sum;
    }
    __syncthreads();

    // First warp reduces the block's warp sums
    local_sum = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) {
        local_sum = warpReduceSum(local_sum);
        if (threadIdx.x == 0) {
            atomicAdd(sum_sq, local_sum);
        }
    }
}

// Kernel 2: Apply RMS scaling and shift
__global__ void rmsNormKernel(const float* __restrict__ input, float* __restrict__ output,
                              const float* __restrict__ sum_sq, float gamma, float beta, int N, float eps) {
    // Load the precomputed inverse RMS once per block into shared memory
    __shared__ float inv_rms_gamma;
    if (threadIdx.x == 0) {
        float mean_sq = (*sum_sq) / static_cast<float>(N);
        float inv_rms = rsqrtf(mean_sq + eps);
        inv_rms_gamma = inv_rms * gamma;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Compute y_i = (x_i * inv_rms * gamma) + beta
    for (int i = idx; i < N; i += stride) {
        output[i] = fmaf(input[i], inv_rms_gamma, beta);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, float* output, int N, float eps) {
    if (N <= 0) return;

    float* d_sum_sq = nullptr;
    cudaMalloc(&d_sum_sq, sizeof(float));
    cudaMemset(d_sum_sq, 0, sizeof(float));

    const int threads = 256;
    int blocks = (N + threads - 1) / threads;
    // Cap blocks to avoid excessive atomic contention while keeping SMs saturated
    if (blocks > 1024) blocks = 1024;

    sumSquaresKernel<<<blocks, threads>>>(input, d_sum_sq, N);
    rmsNormKernel<<<blocks, threads>>>(input, output, d_sum_sq, gamma, beta, N, eps);

    cudaFree(d_sum_sq);
}
