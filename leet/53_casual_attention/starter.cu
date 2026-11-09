#include <cuda_runtime.h>
#include <math.h>

// Shared memory size can be dynamically or statically allocated.
// Since d <= 128, a single row of K or V fits well within registers/shared memory per block.
__global__ void causal_attention_kernel(const float* __restrict__ Q, 
                                         const float* __restrict__ K, 
                                         const float* __restrict__ V, 
                                         float* __restrict__ output, 
                                         int M, int d) {
    // Each thread handles one query row i
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;

    // Scale factor: 1.0 / sqrt(d)
    float scale = 1.0f / sqrtf((float)d);

    // Load Query row i into registers or local arrays
    // Since d <= 128, we can use a fixed-size local buffer
    float q_row[128];
    #pragma unroll
    for (int k = 0; k < d; ++k) {
        q_row[k] = Q[i * d + k];
    }

    // First pass: Compute dot products, apply causal mask, and find max for numerical stability
    float scores[10000]; // Allocation for the maximum possible causal keys (j <= i)
    float max_score = -INFINITY;

    for (int j = 0; j <= i; ++j) {
        float score = 0.0f;
        #pragma unroll
        for (int k = 0; k < d; ++k) {
            score += q_row[k] * K[j * d + k];
        }
        score *= scale;
        scores[j] = score;
        if (score > max_score) {
            max_score = score;
        }
    }

    // Second pass: Compute sum of exponentials (denominator of softmax)
    float exp_sum = 0.0f;
    for (int j = 0; j <= i; ++j) {
        scores[j] = expf(scores[j] - max_score);
        exp_sum += scores[j];
    }

    // Third pass: Multiply softmax probabilities by Value matrix V to get output row i
    // Initialize output row accumulates to 0
    float out_row[128] = {0.0f};
    for (int j = 0; j <= i; ++j) {
        float weight = scores[j] / exp_sum;
        #pragma unroll
        for (int k = 0; k < d; ++k) {
            out_row[k] += weight * V[j * d + k];
        }
    }

    // Write the accumulated results back to global memory
    #pragma unroll
    for (int k = 0; k < d; ++k) {
        output[i * d + k] = out_row[k];
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d) {
    // Define execution configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the causal attention kernel
    causal_attention_kernel<<<blocksPerGrid, threadsPerBlock>>>(Q, K, V, output, M, d);
    
    // Synchronize to ensure completion before returning
    cudaDeviceSynchronize();
}
