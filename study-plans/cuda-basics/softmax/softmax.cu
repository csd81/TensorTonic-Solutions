#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    __shared__ float s_max[32];
    __shared__ float s_sum[32];
    __shared__ float global_max;
    __shared__ float global_sum;

    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;

    // ==========================================
    // PASS 1: Find Global Maximum (for stability)
    // ==========================================
    // Initialize with input[0] safely since constraint N >= 1
    float local_max = input[0]; 
    for (int i = tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, input[i]);
    }

    // Warp-level reduction for max
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }

    // Store warp maximums in shared memory
    if (lane == 0) {
        s_max[wid] = local_max;
    }
    __syncthreads();

    // Final block-level reduction by warp 0
    if (wid == 0) {
        int num_warps = blockDim.x / 32;
        // Pad out-of-bounds lanes with a safe existing value (s_max[0])
        float val = (lane < num_warps) ? s_max[lane] : s_max[0];
        for (int offset = 16; offset > 0; offset /= 2) {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
        if (lane == 0) global_max = val;
    }
    __syncthreads();
    
    float m = global_max;

    // ==========================================
    // PASS 2: Find Sum of Exponentials
    // ==========================================
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        local_sum += expf(input[i] - m);
    }

    // Warp-level reduction for sum
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    // Store warp sums in shared memory
    if (lane == 0) {
        s_sum[wid] = local_sum;
    }
    __syncthreads();

    // Final block-level reduction by warp 0
    if (wid == 0) {
        int num_warps = blockDim.x / 32;
        // Pad out-of-bounds lanes with 0.0f (neutral for sum)
        float val = (lane < num_warps) ? s_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) global_sum = val;
    }
    __syncthreads();
    
    float s = global_sum;

    // ==========================================
    // PASS 3: Normalize and Write Outputs
    // ==========================================
    for (int i = tid; i < N; i += blockDim.x) {
        output[i] = expf(input[i] - m) / s;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    // Launch a single block mapped to the entire row/array.
    // We use 1024 threads to maximize hardware occupancy and work distribution 
    // for vectors length > 1024.
    int threads = 1024;
    int blocks = 1;
    
    softmax_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}