#include <cuda_runtime.h>
#include <math.h>

// CUDA kernel for variable-length causal self-attention over packed sequences
__global__ void causal_attention_kernel(const float* __restrict__ Q, 
                                        const float* __restrict__ K, 
                                        const float* __restrict__ V, 
                                        const int* __restrict__ cu_seqlens,
                                        float* __restrict__ output, 
                                        int T, int d) {
    // Each block processes exactly one sequence
    int s = blockIdx.x;
    int start_idx = cu_seqlens[s];
    int end_idx = cu_seqlens[s + 1];
    int seq_len = end_idx - start_idx;

    if (seq_len <= 0) return;

    float scale = 1.0f / sqrtf((float)d);

    // Dynamic shared memory allocated at launch to fit the maximum sequence length's exp sums
    // Each thread needs memory to store intermediate results if necessary, 
    // but here we allocate shared memory per sequence token row for safe reduction.
    extern __shared__ float shared_mem[];
    float* row_max = shared_mem;              // size: seq_len
    float* row_sum = shared_mem + seq_len;      // size: seq_len

    // Step 1: Compute max score for each row (token t) in the sequence for numerical stability
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        int t = start_idx + i;
        float max_val = -INFINITY;

        // Causal restriction: u only goes from 0 to i (inclusive)
        for (int j = 0; j <= i; ++j) {
            int u = start_idx + j;
            float score = 0.0f;
            for (int k = 0; k < d; ++k) {
                score += Q[t * d + k] * K[u * d + k];
            }
            score *= scale;
            if (score > max_val) {
                max_val = score;
            }
        }
        row_max[i] = max_val;
    }
    __syncthreads();

    // Step 2: Compute denominator (sum of exps) for each row
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        int t = start_idx + i;
        float sum_val = 0.0f;
        float m = row_max[i];

        for (int j = 0; j <= i; ++j) {
            int u = start_idx + j;
            float score = 0.0f;
            for (int k = 0; k < d; ++k) {
                score += Q[t * d + k] * K[u * d + k];
            }
            score *= scale;
            sum_val += expf(score - m);
        }
        row_sum[i] = sum_val;
    }
    __syncthreads();

    // Step 3: Compute Softmax * V and write to the output matrix
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        int t = start_idx + i;
        float m = row_max[i];
        float s_sum = row_sum[i];

        // Loop over dimensions of V to compute the output token representation
        for (int dim = 0; dim < d; ++dim) {
            float out_val = 0.0f;

            for (int j = 0; j <= i; ++j) {
                int u = start_idx + j;
                float score = 0.0f;
                for (int k = 0; k < d; ++k) {
                    score += Q[t * d + k] * K[u * d + k];
                }
                score *= scale;
                
                float weight = expf(score - m) / s_sum;
                out_val += weight * V[u * d + dim];
            }
            output[t * d + dim] = out_val;
        }
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, const int* cu_seqlens,
                      float* output, int T, int d, int S) {
    // Determine a reasonable block size
    int threads_per_block = 256;
    
    // Each sequence can have up to 2048 tokens. 
    // We dynamically allocate shared memory for row_max and row_sum arrays.
    // Max requirement per block: 2048 * sizeof(float) * 2 = 16384 bytes (16 KB), well within typical limits.
    size_t shared_mem_size = 2 * 2048 * sizeof(float);

    // Launch S blocks (one block per sequence)
    causal_attention_kernel<<<S, threads_per_block, shared_mem_size>>>(Q, K, V, cu_seqlens, output, T, d);
    
    // Synchronize to ensure completion before returning to host
    cudaDeviceSynchronize();
}
