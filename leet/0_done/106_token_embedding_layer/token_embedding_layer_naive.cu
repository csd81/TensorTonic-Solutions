#include <cuda_runtime.h>

// Naive baseline for challenge 106, kept as the starter's shape: one block
// per (b, t) with 256 threads, scalar loads, a shared-memory tree reduction,
// and a second pass over both embedding rows for the normalize step.
// Measures 2.33 ms at the benchmark shape (64.8 GB/s effective; the wall is
// ~81 GB/s and the shipped kernel reaches it).

// Coalesced and parallelized kernel for embedding lookup and layer normalization
__global__ void embedding_ln_naive_kernel(
    const int* __restrict__ token_ids, 
    const int* __restrict__ position_ids, 
    const float* __restrict__ token_embeddings,
    const float* __restrict__ position_embeddings, 
    const float* __restrict__ gamma, 
    const float* __restrict__ beta,
    float* __restrict__ output, 
    int B, int T, int V, int P, int D, float eps) 
{
    // Each block processes a single (batch_idx, time_step) pair -> sequence element
    int b = blockIdx.y;
    int t = blockIdx.x;

    if (b >= B || t >= T) return;

    // Fetch the token ID and position ID for the current sequence element
    int token_id = token_ids[b * T + t];
    int pos_id = position_ids[t];

    // Identify pointers to the exact rows in the embedding tables
    const float* token_row = token_embeddings + (token_id * D);
    const float* pos_row = position_embeddings + (pos_id * D);

    // Step 1: Compute sum and sum of squares for mean and variance
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;

    // Strided loop through the embedding dimensions to leverage block threads efficiently
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float val = token_row[d] + pos_row[d];
        local_sum += val;
        local_sq_sum += val * val;
    }

    // Parallel reduction in shared memory to combine thread aggregates
    extern __shared__ float shared_mem[];
    float* s_sum = shared_mem;
    float* s_sq_sum = shared_mem + blockDim.x;

    s_sum[threadIdx.x] = local_sum;
    s_sq_sum[threadIdx.x] = local_sq_sum;
    __syncthreads();

    // Reduce within the block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
            s_sq_sum[threadIdx.x] += s_sq_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Read the mean and variance components (thread 0 holds the total sums)
    float mean = s_sum[0] / D;
    float var = (s_sq_sum[0] / D) - (mean * mean);
    float inv_std = rsqrtf(var + eps);

    // Step 2: Apply Layer Normalization scale, shift, and save to output
    float* out_row = output + (b * T * D + t * D);
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float val = token_row[d] + pos_row[d];
        out_row[d] = gamma[d] * ((val - mean) * inv_std) + beta[d];
    }
}

extern "C" void solve(const int* token_ids, const int* position_ids, const float* token_embeddings,
                      const float* position_embeddings, const float* gamma, const float* beta,
                      float* output, int B, int T, int V, int P, int D, float eps) 
{
    // Allocate 256 threads per block (standard optimal balance for shared memory & occupancy)
    int threads = 256;
    
    // Each (B, T) combination maps neatly onto a 2D grid structure
    dim3 grid(T, B);
    
    // Shared memory allocated dynamically: 2 arrays of size `threads` (one for sum, one for sq_sum)
    size_t shared_mem_size = 2 * threads * sizeof(float);

    embedding_ln_naive_kernel<<<grid, threads, shared_mem_size>>>(
        token_ids, position_ids, token_embeddings, position_embeddings, gamma, beta, output, B, T, V, P, D, eps
    );
}
