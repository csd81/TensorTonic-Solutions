#include <cuda_runtime.h>
#include <math.h>

// CUDA kernel to compute Attention with ALiBi
__global__ void alibi_attention_kernel(
    const float* __restrict__ Q, 
    const float* __restrict__ K, 
    const float* __restrict__ V, 
    float* __restrict__ output, 
    int M, int N, int d, float alpha) 
{
    // Each thread handles one cell (i, k) of the final output matrix M x d
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= M || k >= d) return;

    // Scale factor for the dot product attention
    float inv_sqrt_d = 1.0f / sqrtf((float)d);

    // Step 1: Find the maximum attention score in row i to guarantee numerical stability for Softmax
    float max_score = -1e20f;
    for (int j = 0; j < N; ++j) {
        float qk = 0.0f;
        for (int dot_idx = 0; dot_idx < d; ++dot_idx) {
            qk += Q[i * d + dot_idx] * K[j * d + dot_idx];
        }
        float score = qk * inv_sqrt_d + alpha * (float)(i - j);
        if (score > max_score) {
            max_score = score;
        }
    }

    // Step 2: Compute the sum of exponentials for the denominator of the Softmax function
    float sum_exp = 0.0f;
    for (int j = 0; j < N; ++j) {
        float qk = 0.0f;
        for (int dot_idx = 0; dot_idx < d; ++dot_idx) {
            qk += Q[i * d + dot_idx] * K[j * d + dot_idx];
        }
        float score = qk * inv_sqrt_d + alpha * (float)(i - j);
        sum_exp += expf(score - max_score);
    }

    // Step 3: Compute the final weighted sum using the Softmax distribution over the Value matrix V
    float out_val = 0.0f;
    for (int j = 0; j < N; ++j) {
        float qk = 0.0f;
        for (int dot_idx = 0; dot_idx < d; ++dot_idx) {
            qk += Q[i * d + dot_idx] * K[j * d + dot_idx];
        }
        float score = qk * inv_sqrt_d + alpha * (float)(i - j);
        float weight = expf(score - max_score) / sum_exp;
        
        out_val += weight * V[j * d + k];
    }

    // Store the finalized result inside the output matrix
    output[i * d + k] = out_val;
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d, float alpha) 
{
    // Define block and grid configurations
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((d + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel execution
    alibi_attention_kernel<<<numBlocks, threadsPerBlock>>>(Q, K, V, output, M, N, d, alpha);

    // Sync to ensure work is completed on the stream before returning
    cudaDeviceSynchronize();
}
