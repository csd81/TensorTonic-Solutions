#include <cuda_runtime.h>
#include <math.h>

__global__ void batch_norm_2d_kernel(const float* __restrict__ input,
                                     const float* __restrict__ gamma,
                                     const float* __restrict__ beta,
                                     float* __restrict__ output,
                                     int N, int C, float eps) {
    // Each block processes one feature channel j
    int j = blockIdx.x;
    if (j >= C) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Shared memory for parallel reductions (size defined at launch)
    extern __shared__ float sdata[];

    // 1. Compute sum for mean (mu_j)
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += block_size) {
        local_sum += input[i * C + j];
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float mu = sdata[0] / static_cast<float>(N);
    __syncthreads(); // Ensure all threads read mu before reusing sdata

    // 2. Compute sum of squared differences for variance (sigma_j^2)
    float local_sq_diff = 0.0f;
    for (int i = tid; i < N; i += block_size) {
        float diff = input[i * C + j] - mu;
        local_sq_diff += diff * diff;
    }
    sdata[tid] = local_sq_diff;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float var = sdata[0] / static_cast<float>(N);
    float inv_std = rsqrtf(var + eps);
    float g = gamma[j];
    float b = beta[j];

    // 3. Normalize, scale, and shift
    for (int i = tid; i < N; i += block_size) {
        int idx = i * C + j;
        float x_hat = (input[idx] - mu) * inv_std;
        output[idx] = g * x_hat + b;
    }
}

// input, gamma, beta, output are device pointers
extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output,
                      int N, int C, float eps) {
    if (N <= 0 || C <= 0) return;

    // 256 threads per block is a good balance for N up to 10,000
    int threads_per_block = 256;
    int blocks_per_grid = C;
    size_t shared_mem_size = threads_per_block * sizeof(float);

    batch_norm_2d_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
        input, gamma, beta, output, N, C, eps
    );
}
