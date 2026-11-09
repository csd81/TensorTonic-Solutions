#include <cuda_runtime.h>
#include <math.h>

// Coalesced/shared-memory reduction helper for max finding within a warp
__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float tmp = __shfl_down_sync(0xffffffff, val, offset);
        if (tmp > val) val = tmp;
    }
    return val;
}

// Coalesced/shared-memory reduction helper for sum within a warp
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void gated_causal_attn_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ X,
    const float* __restrict__ W_g,
    float* __restrict__ output,
    int M, int d_model, int h, int d_k, float scale) 
{
    // Map each thread block to a specific token (row i) and attention head (head j)
    int i = blockIdx.x; // token index (row)
    int j = blockIdx.y; // head index
    
    if (i >= M || j >= h) return;

    int tid = threadIdx.x;
    int bdim = blockDim.x;

    // Dynamically allocated shared memory to store softmax probabilities during reduction
    // Needs to hold up to M elements. Allocated based on M in execution configuration.
    extern __shared__ float s_mem[]; 
    
    // --- STEP 1: Compute Q_j * K_j^T for row i with Causal Masking ---
    // Thread block collaboratively computes scores for K_j[col] where col <= i
    // For col > i, the causal mask sets the logit value to -infinity.
    float max_val = -1e20f; 

    // Locate the query vector for this specific head and row
    const float* q_ptr = Q + i * d_model + j * d_k;

    for (int col = tid; col <= i; col += bdim) {
        const float* k_ptr = K + col * d_model + j * d_k;
        float score = 0.0f;
        for (int d = 0; d < d_k; ++d) {
            score += q_ptr[d] * k_ptr[d];
        }
        score *= scale;
        s_mem[col] = score;
        if (score > max_val) max_val = score;
    }

    // Initialize remaining elements past the causal limit to -inf
    for (int col = i + 1 + tid; col < M; col += bdim) {
        s_mem[col] = -1e20f;
    }
    __syncthreads();

    // Block-wide parallel reduction to find the maximum logit value (for numerical stability)
    float block_max = max_val;
    if (bdim > 32) {
        __shared__ float shm_max[32];
        int lane = tid % 32;
        int wid = tid / 32;
        float w_max = warpReduceMax(max_val);
        if (lane == 0) shm_max[wid] = w_max;
        __syncthreads();
        
        float final_max = (tid < (bdim / 32)) ? shm_max[lane] : -1e20f;
        if (wid == 0) {
            final_max = warpReduceMax(final_max);
            if (lane == 0) shm_max[0] = final_max;
        }
        __syncthreads();
        block_max = shm_max[0];
    } else {
        __shared__ float shm_max[1];
        float w_max = warpReduceMax(max_val);
        if (tid == 0) shm_max[0] = w_max;
        __syncthreads();
        block_max = shm_max[0];
    }

    // --- STEP 2: Exponentiate and Compute Softmax Sum ---
    float sum_val = 0.0f;
    for (int col = tid; col <= i; col += bdim) {
        float exp_score = expf(s_mem[col] - block_max);
        s_mem[col] = exp_score;
        sum_val += exp_score;
    }
    for (int col = i + 1 + tid; col < M; col += bdim) {
        s_mem[col] = 0.0f;
    }
    __syncthreads();

    // Block-wide reduction to sum up the exponentials
    float block_sum = sum_val;
    if (bdim > 32) {
        __shared__ float shm_sum[32];
        int lane = tid % 32;
        int wid = tid / 32;
        float w_sum = warpReduceSum(sum_val);
        if (lane == 0) shm_sum[wid] = w_sum;
        __syncthreads();
        
        float final_sum = (tid < (bdim / 32)) ? shm_sum[lane] : 0.0f;
        if (wid == 0) {
            final_sum = warpReduceSum(final_sum);
            if (lane == 0) shm_sum[0] = final_sum;
        }
        __syncthreads();
        block_sum = shm_sum[0];
    } else {
        __shared__ float shm_sum[1];
        float w_sum = warpReduceSum(sum_val);
        if (tid == 0) shm_sum[0] = w_sum;
        __syncthreads();
        block_sum = shm_sum[0];
    }

    // Normalize exponentiated values into valid Softmax probabilities
    float inv_sum = 1.0f / (block_sum + 1e-9f);
    for (int col = tid; col <= i; col += bdim) {
        s_mem[col] *= inv_sum;
    }
    __syncthreads();

    // --- STEP 3: Compute Head Attention Matrix Product & Fused Gating ---
    // Every thread calculates a specific subset of dimension index 'd' in head_j
    int h_offset = j * d_k;
    const float* x_row = X + i * d_model;

    for (int d = tid; d < d_k; d += bdim) {
        // Multi-head Attention calculation: head_j = Softmax(scores) * V_j
        float head_val = 0.0f;
        for (int col = 0; col <= i; ++col) {
            head_val += s_mem[col] * V[col * d_model + h_offset + d];
        }

        // Fused Gate generation: Compute G_j = X * W_g inline for column (h_offset + d)
        float g_logit = 0.0f;
        int target_col = h_offset + d;
        for (int k_idx = 0; k_idx < d_model; ++k_idx) {
            g_logit += x_row[k_idx] * W_g[k_idx * d_model + target_col];
        }
        float gate = 1.0f / (1.0f + expf(-g_logit)); // Sigmoid element-wise gating

        // Store the final element-wise multiplied value directly into global memory
        output[i * d_model + h_offset + d] = head_val * gate;
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, const float* X,
                      const float* W_g, float* output, int M, int d_model, int h) 
{
    int d_k = d_model / h;
    float scale = 1.0f / sqrtf((float)d_k);

    // Block dimensions tailored dynamically to stay within limits and preserve execution occupancy
    int threads = 256;
    if (M < 256) {
        threads = ((M + 31) / 32) * 32; // Round up to nearest warp size
        if (threads < 32) threads = 32;
    }

    // Map Block Grid to process each Token across every Head independently
    dim3 grid(M, h);
    
    // Allocate dynamic shared memory space dynamically configured to holding M float elements
    size_t shared_mem_size = M * sizeof(float);

    gated_causal_attn_kernel<<<grid, threads, shared_mem_size>>>(
        Q, K, V, X, W_g, output, M, d_model, h, d_k, scale
    );
    
    // Synchronize to guarantee execution completes within the calling scope
    cudaDeviceSynchronize();
}
