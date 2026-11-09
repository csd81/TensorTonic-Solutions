#include <cuda_runtime.h>

// Cooperative block-level reduction helper for sum
__device__ __forceinline__ float block_reduce_sum(float val) {
    static __shared__ float shared[32]; 
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // Read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;

    if (wid == 0) {
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }
    return val;
}

// Kernel to compute input gradient dx
__global__ void layernorm_backward_dx_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    const float* __restrict__ dy,
    float* __restrict__ dx,
    int N, int C) 
{
    // Each block processes a single row
    int i = blockIdx.x; 
    if (i >= N) return;

    float mu = mean[i];
    float r = rstd[i];

    float sum_g = 0.0f;
    float sum_g_xhat = 0.0f;

    // First pass loop: compute local sums for sum_g and sum_g_xhat across the row
    for (int j = threadIdx.x; j < C; j += blockDim.x) {
        int idx = i * C + j;
        float x_ij = x[idx];
        float dy_ij = dy[idx];
        float gamma_j = gamma[j];

        float xhat_ij = (x_ij - mu) * r;
        float g_ij = dy_ij * gamma_j;

        sum_g += g_ij;
        sum_g_xhat += g_ij * xhat_ij;
    }

    // Shared block reductions
    sum_g = block_reduce_sum(sum_g);
    sum_g_xhat = block_reduce_sum(sum_g_xhat);

    // Broadcast reduced sums to all threads in the block using shared memory
    __shared__ float s_sum_g, s_sum_g_xhat;
    if (threadIdx.x == 0) {
        s_sum_g = sum_g;
        s_sum_g_xhat = sum_g_xhat;
    }
    __syncthreads();

    float inv_C = 1.0f / (float)C;

    // Second pass loop: compute final dx values
    for (int j = threadIdx.x; j < C; j += blockDim.x) {
        int idx = i * C + j;
        float x_ij = x[idx];
        float dy_ij = dy[idx];
        float gamma_j = gamma[j];

        float xhat_ij = (x_ij - mu) * r;
        float g_ij = dy_ij * gamma_j;

        dx[idx] = r * (g_ij - inv_C * s_sum_g - xhat_ij * inv_C * s_sum_g_xhat);
    }
}

// Kernel to compute parameter gradients dgamma and dbeta
__global__ void layernorm_backward_params_kernel(
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    const float* __restrict__ dy,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta,
    int N, int C) 
{
    // Each thread handles a distinct column index
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= C) return;

    float sum_dgamma = 0.0f;
    float sum_dbeta = 0.0f;

    // Reduce over all rows vertically for the assigned column
    for (int i = 0; i < N; ++i) {
        int idx = i * C + j;
        float dy_ij = dy[idx];
        float x_ij = x[idx];
        float mu = mean[i];
        float r = rstd[i];

        float xhat_ij = (x_ij - mu) * r;

        sum_dgamma += dy_ij * xhat_ij;
        sum_dbeta += dy_ij;
    }

    // Write final output values directly
    dgamma[j] = sum_dgamma;
    dbeta[j] = sum_dbeta;
}

extern "C" void solve(const float* x, const float* gamma, const float* mean, const float* rstd,
                      const float* dy, float* dx, float* dgamma, float* dbeta, int N, int C) 
{
    // 1. Launch Input Gradient Kernel (dx)
    // Map each row to a unique block to handle large C effectively
    int threads_dx = 256;
    int blocks_dx = N;
    layernorm_backward_dx_kernel<<<blocks_dx, threads_dx>>>(x, gamma, mean, rstd, dy, dx, N, C);

    // 2. Launch Parameter Gradients Kernel (dgamma, dbeta)
    // Distribute columns evenly across 1D threads-per-block configuration
    int threads_params = 256;
    int blocks_params = (C + threads_params - 1) / threads_params;
    layernorm_backward_params_kernel<<<blocks_params, threads_params>>>(x, mean, rstd, dy, dgamma, dbeta, N, C);
    
    // Ensure all processing streams conclude asynchronously safe
    cudaDeviceSynchronize();
}
