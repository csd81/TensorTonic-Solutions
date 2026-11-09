#include <cuda_runtime.h>

// CUDA Kernel for computing RoPE with half-split layout
__global__ void rope_kernel(const float* __restrict__ Q, 
                            const float* __restrict__ cos_vals, 
                            const float* __restrict__ sin_vals, 
                            float* __restrict__ output, 
                            int M, int D) 
{
    int total_elements = M * D;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total_elements; i += stride) {
        int col = i % D;
        int half_D = D / 2;
        
        float q_val = Q[i];
        float cos_val = cos_vals[i];
        float sin_val = sin_vals[i];
        float rotated_q;

        // Perform half-split layout rotation logic
        if (col < half_D) {
            // rotate_half maps second half to first half with negation
            rotated_q = -Q[i + half_D];
        } else {
            // rotate_half maps first half to second half without negation
            rotated_q = Q[i - half_D];
        }

        // Element-wise fused multiply-add operations
        output[i] = q_val * cos_val + rotated_q * sin_val;
    }
}

// Host function setup
extern "C" void solve(float* Q, float* cos, float* sin, float* output, int M, int D) {
    int total_elements = M * D;
    
    // Choose an optimal block size (typically 256 or 512 for memory-bound tasks)
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    // Cap the grid size to prevent unnecessary overhead if the dataset is massive
    if (grid_size > 4096) {
        grid_size = 4096;
    }

    rope_kernel<<<grid_size, block_size>>>(Q, cos, sin, output, M, D);
}
