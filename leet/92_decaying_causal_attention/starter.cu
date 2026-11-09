#include <cuda_runtime.h>
#include <cmath>

// CUDA kernel to compute decaying causal attention
__global__ void decaying_causal_attention_kernel(
    const float* __restrict__ Q, 
    const float* __restrict__ K, 
    const float* __restrict__ V, 
    float* __restrict__ output, 
    int seq_len, 
    int d_model, 
    float gamma,
    float scale) 
{
    // Each block handles a set of rows (n), each thread handles a column index (d)
    int n = blockIdx.x * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x; // if treating block as 2D, or simpler:
    
    // Better Mapping: 
    // blockIdx.x is the row 'n' being computed.
    // threadIdx.x is the feature dimension 'd' (from 0 to d_model - 1).
    int row_n = blockIdx.x;
    int col_d = threadIdx.x;

    if (row_n >= seq_len || col_d >= d_model) {
        return;
    }

    // Load Q[row_n] into shared memory or registers to avoid redundant global reads
    // Since all threads in this block share row_n, let's load Q[row_n, k] dynamically
    // or just read from global memory if cached in L1/L2.
    
    float accum = 0.0f;

    // Loop over all past positions m <= n
    for (int m = 0; m <= row_n; ++m) {
        // Compute dot product Q[row_n] * K[m]
        float qk_dot = 0.0f;
        for (int k = 0; k < d_model; ++k) {
            qk_dot += Q[row_n * d_model + k] * K[m * d_model + k];
        }
        
        // Apply scaling factor 1 / sqrt(d_model)
        float score = qk_dot * scale;
        
        // Apply decay factor gamma^(row_n - m)
        float decay = powf(gamma, (float)(row_n - m));
        
        // Accumulate (score * decay) * V[m, col_d]
        accum += score * decay * V[m * d_model + col_d];
    }

    // Write result to output
    output[row_n * d_model + col_d] = accum;
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int seq_len,
                      int d_model, float gamma) 
{
    // Scaling factor: 1.0 / sqrt(d_model)
    float scale = 1.0f / sqrtf((float)d_model);

    // Each block processes one row 'n' of the sequence.
    // Threads in the block process the columns 'd_model'.
    // If d_model > 256, we can increase block size, but constraint says d_model <= 256.
    dim3 threadsPerBlock(d_model); 
    dim3 numBlocks(seq_len);

    // Launch kernel
    decaying_causal_attention_kernel<<<numBlocks, threadsPerBlock>>>(
        Q, K, V, output, seq_len, d_model, gamma, scale
    );
    
    // Synchronize to ensure completion
    cudaDeviceSynchronize();
}
