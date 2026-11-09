#include <cuda_runtime.h>

// CUDA Kernel for fused residual addition and RMS normalization
__global__ void fused_residual_rms_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ residual,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int N,
    int C,
    float eps
) {
    // Each block handles one or more rows (tokens)
    // For optimal load balancing up to C=65,536, threads cooperatively loop over the columns
    int row_idx = blockIdx.x;
    if (row_idx >= N) return;

    // Offset pointers to the current row
    const float* row_x = x + row_idx * C;
    const float* row_res = residual + row_idx * C;
    float* row_out = out + row_idx * C;

    float sum_sq = 0.0f;

    // Step 1: Thread-local summation of squares
    for (int col_idx = threadIdx.x; col_idx < C; col_idx += blockDim.x) {
        float z = row_x[col_idx] + row_res[col_idx];
        sum_sq += z * z;
    }

    // Step 2: Intra-warp reduction using shuffle instructions
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }

    // Step 3: Inter-warp reduction using shared memory (if block size > 32)
    __shared__ float shared_sum[32]; 
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (lane_id == 0) {
        shared_sum[warp_id] = sum_sq;
    }
    __syncthreads();

    // The first warp reduces the values from all other warps
    if (warp_id == 0) {
        float block_sum = (threadIdx.x < (blockDim.x / 32)) ? shared_sum[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xFFFFFFFF, block_sum, offset);
        }
        if (lane_id == 0) {
            // Compute the reciprocal of RMS to replace costly division with fast multiplication
            shared_sum[0] = rsqrtf((block_sum / C) + eps);
        }
    }
    __syncthreads();

    float inv_rms = shared_sum[0];

    // Step 4: Compute out[i, j] and write back to global memory
    for (int col_idx = threadIdx.x; col_idx < C; col_idx += blockDim.x) {
        float z = row_x[col_idx] + row_res[col_idx];
        row_out[col_idx] = z * inv_rms * weight[col_idx];
    }
}

extern "C" void solve(
    const float* x, 
    const float* residual, 
    const float* weight, 
    float* out, 
    int N,
    int C, 
    float eps
) {
    // 256 threads per block balance shared memory usage and occupancy cleanly
    int threads_per_block = 256;
    int blocks_per_grid = N;

    fused_residual_rms_norm_kernel<<<blocks_per_grid, threads_per_block>>>(
        x, residual, weight, out, N, C, eps
    );
}
