#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256

__global__ void attention_sinks_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int M, int d, int num_sinks, int window_size) 
{
    // Each block handles exactly one query row i
    int i = blockIdx.x;
    if (i >= M) return;

    int tid = threadIdx.x;
    float scale = 1.0f / sqrtf((float)d);

    // Dynamic window bounds for query row i
    int window_start = i - window_size + 1;
    if (window_start < 0) window_start = 0;

    // Shared memory for finding maximums and summing exponential scores
    __shared__ float s_max[BLOCK_SIZE];
    __shared__ float s_sum[BLOCK_SIZE];

    // --- Pass 1: Local Maximum Score Discovery ---
    float local_max = -1e20f;

    // 1a. Process Sinks
    for (int j = tid; j < num_sinks && j <= i; j += BLOCK_SIZE) {
        float dot = 0.0f;
        for (int k = 0; k < d; ++k) {
            dot += Q[i * d + k] * K[j * d + k];
        }
        float score = dot * scale;
        if (score > local_max) local_max = score;
    }

    // 1b. Process Window Tokens
    int start_j = (window_start < num_sinks) ? num_sinks : window_start;
    for (int j = start_j + tid; j <= i; j += BLOCK_SIZE) {
        float dot = 0.0f;
        for (int k = 0; k < d; ++k) {
            dot += Q[i * d + k] * K[j * d + k];
        }
        float score = dot * scale;
        if (score > local_max) local_max = score;
    }

    // Block reduction to find absolute global max score for row i
    s_max[tid] = local_max;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            if (s_max[tid + stride] > s_max[tid]) {
                s_max[tid] = s_max[tid + stride];
            }
        }
        __syncthreads();
    }
    float max_score = s_max[0];

    // --- Pass 2: Softmax Denominator Accumulation ---
    float local_sum = 0.0f;

    // 2a. Exp-sum over Sinks
    for (int j = tid; j < num_sinks && j <= i; j += BLOCK_SIZE) {
        float dot = 0.0f;
        for (int k = 0; k < d; ++k) {
            dot += Q[i * d + k] * K[j * d + k];
        }
        local_sum += expf(dot * scale - max_score);
    }

    // 2b. Exp-sum over Window
    for (int j = start_j + tid; j <= i; j += BLOCK_SIZE) {
        float dot = 0.0f;
        for (int k = 0; k < d; ++k) {
            dot += Q[i * d + k] * K[j * d + k];
        }
        local_sum += expf(dot * scale - max_score);
    }

    // Block reduction to find absolute global sum for row i
    s_sum[tid] = local_sum;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }
    float sum_exp = s_sum[0];

    // --- Pass 3: Value Weighting and Output Generation ---
    // Accumulate the final vector output row in groups of d dimensions
    for (int k = tid; k < d; k += BLOCK_SIZE) {
        float out_val = 0.0f;

        // 3a. Weighted sum over Sinks
        for (int j = 0; j < num_sinks && j <= i; ++j) {
            float dot = 0.0f;
            for (int col = 0; col < d; ++col) {
                dot += Q[i * d + col] * K[j * d + col];
            }
            float weight = expf(dot * scale - max_score) / sum_exp;
            out_val += weight * V[j * d + k];
        }

        // 3b. Weighted sum over Window
        for (int j = start_j; j <= i; ++j) {
            float dot = 0.0f;
            for (int col = 0; col < d; ++col) {
                dot += Q[i * d + col] * K[j * d + col];
            }
            float weight = expf(dot * scale - max_score) / sum_exp;
            out_val += weight * V[j * d + k];
        }

        output[i * d + k] = out_val;
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d,
                      int num_sinks, int window_size) 
{
    // Launch M blocks (one per query vector row) using 256 threads per block
    dim3 blocks(M);
    dim3 threads(BLOCK_SIZE);

    attention_sinks_kernel<<<blocks, threads>>>(Q, K, V, output, M, d, num_sinks, window_size);
}
