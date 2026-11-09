#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Block size for the reduction kernel
#define BLOCK_SIZE 256

__global__ void fused_prenorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ residual,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    float* __restrict__ residual_out,
    float* __restrict__ mean_out,
    float* __restrict__ rstd_out,
    int N, int C, float eps)
{
    // Allocate shared memory for the two-pass block reductions
    // We need space to reduce sums and squared differences across the block
    __shared__ float s_mem[BLOCK_SIZE];

    int row = blockIdx.x;
    if (row >= N) return;

    int tid = threadIdx.x;

    // Step 1: Compute the row mean (First Pass)
    float thread_sum = 0.0f;
    for (int col = tid; col < C; col += BLOCK_SIZE) {
        int idx = row * C + col;
        float h_val = x[idx] + residual[idx];
        residual_out[idx] = h_val; // Write h to residual_out immediately (one-pass optimization)
        thread_sum += h_val;
    }

    s_mem[tid] = thread_sum;
    __syncthreads();

    // In-block tree reduction for the sum
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_mem[tid] += s_mem[tid + stride];
        }
        __syncthreads();
    }

    float mean = s_mem[0] / C;
    if (tid == 0) {
        mean_out[row] = mean;
    }

    // Step 2: Compute the row variance (Second Pass)
    // Using a stable two-pass approach to handle potential large means safely
    float thread_var_sum = 0.0f;
    for (int col = tid; col < C; col += BLOCK_SIZE) {
        int idx = row * C + col;
        float diff = residual_out[idx] - mean;
        thread_var_sum += diff * diff;
    }

    s_mem[tid] = thread_var_sum;
    __syncthreads();

    // In-block tree reduction for the variance sum
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_mem[tid] += s_mem[tid + stride];
        }
        __syncthreads();
    }

    float rstd = rsqrtf((s_mem[0] / C) + eps);
    if (tid == 0) {
        rstd_out[row] = rstd;
    }
    __syncthreads(); // Ensure rstd and mean are uniformly available if needed, though they are kept in registers

    // Step 3: Compute the final normalized output y
    for (int col = tid; col < C; col += BLOCK_SIZE) {
        int idx = row * C + col;
        float h_val = residual_out[idx];
        y[idx] = (h_val - mean) * rstd * gamma[col] + beta[col];
    }
}

extern "C" void solve(const float* x, const float* residual, const float* gamma, const float* beta,
                      float* y, float* residual_out, float* mean, float* rstd, int N, int C,
                      float eps) 
{
    // Launch one block per row to keep row elements inside fast shared memory/registers
    dim3 grid(N);
    dim3 block(BLOCK_SIZE);

    fused_prenorm_kernel<<<grid, block>>>(
        x, residual, gamma, beta, 
        y, residual_out, mean, rstd, 
        N, C, eps
    );
}
