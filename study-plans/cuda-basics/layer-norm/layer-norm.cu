#include <cuda_runtime.h>
#include <math.h>

__global__ void layer_norm_kernel(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    int tid = threadIdx.x;
    int row = blockIdx.x;
    
    // Safety check in case grid exceeds M
    if (row >= M) return;

    int row_offset = row * N;
    
    // Step 1: Thread-local accumulation of sum and squared sum
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    // A strided loop enables handling rows larger than the block size (N > 256)
    for (int j = tid; j < N; j += blockDim.x) {
        float val = input[row_offset + j];
        local_sum += val;
        local_sum_sq += val * val;
    }

    // Step 2: Block-wide tree reduction using shared memory
    __shared__ float s_sum[256];
    __shared__ float s_sum_sq[256];

    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    __syncthreads();

    // Tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sum_sq[tid] += s_sum_sq[tid + stride];
        }
        __syncthreads();
    }

    // Step 3: Compute mean and inverse standard deviation in thread 0
    __shared__ float s_mean;
    __shared__ float s_inv_std;

    if (tid == 0) {
        s_mean = s_sum[0] / N;
        // E[x^2] - (E[x])^2. 
        // fmaxf prevents negative variance from floating point inaccuracies
        float var = fmaxf((s_sum_sq[0] / N) - (s_mean * s_mean), 0.0f);
        s_inv_std = rsqrtf(var + eps);
    }
    // Ensure all threads can see the computed mean and inverse std dev
    __syncthreads();

    // Step 4: Fused normalize and affine transform write
    for (int j = tid; j < N; j += blockDim.x) {
        float val = input[row_offset + j];
        float norm_val = (val - s_mean) * s_inv_std;
        output[row_offset + j] = norm_val * gamma[j] + beta[j];
    }
}

extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    int threads = 256;
    dim3 blocks(M);
    
    // Launch one block per row, so each row reduces entirely independently
    layer_norm_kernel<<<blocks, threads>>>(input, gamma, beta, output, M, N, eps);
    cudaDeviceSynchronize();
}