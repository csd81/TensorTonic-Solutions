#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define BLOCK_SIZE_X 256

// Shared memory configuration: 
// - Block reduction for row statistics (dx calculation)
// - Column reductions for dweight (using shared memory or atomic operations depending on performance requirements)
__global__ void rms_norm_backward_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ weight, 
    const float* __restrict__ dy, 
    float* __restrict__ dx,
    float* __restrict__ dweight, 
    int N, int C, float eps) 
{
    // 1. Compute dx and intermediate localized dweight contributions
    // Each block manages a set of columns for a specific row, or maps blocks to rows.
    // Given the constraints (C up to 16,384), we map 1 block to process exactly 1 row.
    int row = blockIdx.x;
    if (row >= N) return;

    int tid = threadIdx.x;

    // First, compute the row sum of x^2 to get r_i
    float row_sq_sum = 0.0f;
    for (int col = tid; col < C; col += blockDim.x) {
        float val = x[row * C + col];
        row_sq_sum += val * val;
    }

    // Shared memory reduction for the row sum
    __shared__ float s_mem[BLOCK_SIZE_X];
    s_mem[tid] = row_sq_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_mem[tid] += s_mem[tid + stride];
        }
        __syncthreads();
    }

    float r_i = 1.0f / sqrtf((s_mem[0] / C) + eps);

    // Compute g_ij * x_hat_ij row sum for the dx formula
    float dot_g_xhat = 0.0f;
    for (int col = tid; col < C; col += blockDim.x) {
        int idx = row * C + col;
        float x_hat = x[idx] * r_i;
        float g = dy[idx] * weight[col];
        dot_g_xhat += g * x_hat;
    }

    s_mem[tid] = dot_g_xhat;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_mem[tid] += s_mem[tid + stride];
        }
        __syncthreads();
    }

    float mean_g_xhat = s_mem[0] / C;

    // Compute and write dx, and perform atomic adds to global memory for dweight
    for (int col = tid; col < C; col += blockDim.x) {
        int idx = row * C + col;
        float x_hat = x[idx] * r_i;
        float g = dy[idx] * weight[col];
        
        // dx write-out
        dx[idx] = r_i * (g - x_hat * mean_g_xhat);

        // Accumulate dweight safely across rows
        float dw_val = dy[idx] * x_hat;
        atomicAdd(&dweight[col], dw_val);
    }
}

// Initialization kernel because dweight and dx are not zero-initialized
__global__ void zero_init_dweight(float* dweight, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < C) {
        dweight[idx] = 0.0f;
    }
}

extern "C" void solve(const float* x, const float* weight, const float* dy, float* dx,
                      float* dweight, int N, int C, float eps) 
{
    // Initialize dweight to zero
    int init_threads = 256;
    int init_blocks = (C + init_threads - 1) / init_threads;
    zero_init_dweight<<<init_blocks, init_threads>>>(dweight, C);

    // Launch main backward pass kernel
    // 1 Block per Row (N blocks total)
    dim3 grid(N);
    dim3 block(BLOCK_SIZE_X);

    rms_norm_backward_kernel<<<grid, block>>>(x, weight, dy, dx, dweight, N, C, eps);
    cudaDeviceSynchronize();
}
