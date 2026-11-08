#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// Warp-level reduction for maximum value
__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Block-level reduction for maximum value
__device__ float blockReduceMax(float val) {
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

// Warp-level reduction for sum
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction for sum
__device__ float blockReduceSum(float val) {
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

// Kernel: One thread block processes one sample (row)
__global__ void cross_entropy_kernel(
    const float* __restrict__ logits,
    const int* __restrict__ true_labels,
    float* __restrict__ loss,
    int N,
    int C
) {
    int sample_idx = blockIdx.x;
    if (sample_idx >= N) return;

    const float* row = logits + sample_idx * C;

    // 1. Find the maximum logit in the row for numerical stability
    float thread_max = -FLT_MAX;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        thread_max = fmaxf(thread_max, row[c]);
    }
    
    __shared__ float s_max;
    float max_val = blockReduceMax(thread_max);
    if (threadIdx.x == 0) {
        s_max = max_val;
    }
    __syncthreads();
    max_val = s_max;

    // 2. Compute sum of exp(z_{jk} - max_val)
    float thread_sum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        thread_sum += expf(row[c] - max_val);
    }

    float sum_exp = blockReduceSum(thread_sum);

    // 3. Thread 0 calculates final sample loss and accumulates into global average
    if (threadIdx.x == 0) {
        int label = true_labels[sample_idx];
        float sample_loss = (max_val + logf(sum_exp)) - row[label];
        atomicAdd(loss, sample_loss / static_cast<float>(N));
    }
}

// logits, true_labels, loss are device pointers
extern "C" void solve(const float* logits, const int* true_labels, float* loss, int N, int C) {
    // Initialize output device memory to 0 before atomic accumulation
    cudaMemset(loss, 0, sizeof(float));

    // Launch one block per sample; 256 threads efficiently covers C <= 1000
    int threads_per_block = 256;
    int blocks_per_grid = N;

    cross_entropy_kernel<<<blocks_per_grid, threads_per_block>>>(
        logits, true_labels, loss, N, C
    );
}
