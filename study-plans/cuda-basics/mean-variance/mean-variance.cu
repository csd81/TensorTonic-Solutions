#include <cuda_runtime.h>

__global__ void mean_variance_kernel(const float* input, float* mean_out, float* var_out, int N) {
    float sum = 0.0f;
    float sum_sq = 0.0f;

    // 1. Grid-stride loop: Handles arrays larger than the grid size safely
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        float x = input[i];
        sum += x;
        sum_sq += x * x;
    }

    // 2. Warp-level reduction using fast register shuffles
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    // 3. Shared memory to hold the partial sum of each warp (max 32 warps per block)
    __shared__ float s_sum[32];
    __shared__ float s_sum_sq[32];

    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    // The first thread of each warp writes its results to shared memory
    if (lane == 0) {
        s_sum[warp_id] = sum;
        s_sum_sq[warp_id] = sum_sq;
    }

    __syncthreads();

    // 4. The first warp reduces the per-warp results across the entire block
    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        // Threads out of bounds take 0.0f to avoid garbage accumulation
        float b_sum = (lane < num_warps) ? s_sum[lane] : 0.0f;
        float b_sum_sq = (lane < num_warps) ? s_sum_sq[lane] : 0.0f;

        for (int offset = 16; offset > 0; offset /= 2) {
            b_sum += __shfl_down_sync(0xffffffff, b_sum, offset);
            b_sum_sq += __shfl_down_sync(0xffffffff, b_sum_sq, offset);
        }

        // 5. Thread 0 adds the block's total to the global accumulators
        if (lane == 0) {
            atomicAdd(mean_out, b_sum);
            atomicAdd(var_out, b_sum_sq);
        }
    }
}

__global__ void finalize_kernel(float* mean_out, float* var_out, int N) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float sum_x = *mean_out;
        float sum_x2 = *var_out;
        
        float mean = sum_x / (float)N;
        
        // Var(X) = E[X^2] - E[X]^2
        float var = (sum_x2 / (float)N) - (mean * mean);
        
        // Clamp to 0.0f to handle catastrophic cancellation floating point errors
        if (var < 0.0f) {
            var = 0.0f;
        }
        
        *mean_out = mean;
        *var_out = var;
    }
}

extern "C" void solve(const float* input, float* mean_out, float* var_out, int N) {
    int threads = 256;
    
    // Cap blocks to 1024 to minimize atomic collisions while easily saturating the GPU
    int max_blocks = 1024;
    int blocks = (N + threads - 1) / threads;
    if (blocks > max_blocks) {
        blocks = max_blocks;
    }
    
    // Variables initialized to zero by harness pointer memset
    cudaMemset(mean_out, 0, sizeof(float));
    cudaMemset(var_out, 0, sizeof(float));
    
    // Accumulate sum and sum_of_squares
    mean_variance_kernel<<<blocks, threads>>>(input, mean_out, var_out, N);
    
    // Output final mu and sigma^2 calculations
    finalize_kernel<<<1, 1>>>(mean_out, var_out, N);
    
    cudaDeviceSynchronize();
}