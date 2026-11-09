#include <cuda_runtime.h>

// Vectorized loading can be utilized if layouts align, but a direct coalesced approach 
// guarantees correctness across arbitrary D, H_q, H_kv constraints.
__global__ void fused_qkv_rope_cache_kernel(
    const float* __restrict__ x,
    const float* __restrict__ W_qkv,
    const float* __restrict__ cos_sin_cache,
    const int* __restrict__ positions,
    float* __restrict__ K_cache,
    float* __restrict__ V_cache,
    float* __restrict__ Q_out,
    int B, int d_model, int H_q, int H_kv, int D, int S_max) 
{
    // Thread coordinates
    // blockIdx.x -> sequence index b
    // blockIdx.y -> head group or flat head index
    // threadIdx.x -> channel/dimension index within the head (0 to D-1)
    
    int b = blockIdx.x;
    if (b >= B) return;

    int pos = positions[b];
    int half_D = D / 2;

    // Total head stride allocations
    int q_elements = H_q * D;
    int k_elements = H_kv * D;
    int total_qkv_cols = (H_q + 2 * H_kv) * D;

    // --- 1. Compute Q_out elements ---
    // Every thread can compute a specific channel component of Q
    for (int hq = blockIdx.y; hq < H_q; hq += gridDim.y) {
        for (int j_half = threadIdx.x; j_half < half_D; j_half += blockDim.x) {
            int j1 = j_half;
            int j2 = j_half + half_D;

            // Global column index in W_qkv for Query
            int col1 = hq * D + j1;
            int col2 = hq * D + j2;

            // Matrix multiplication: x[b, :] * W_qkv[:, col]
            float q1 = 0.0f;
            float q2 = 0.0f;
            for (int d = 0; d < d_model; ++d) {
                float x_val = x[b * d_model + d];
                q1 += x_val * W_qkv[d * total_qkv_cols + col1];
                q2 += x_val * W_qkv[d * total_qkv_cols + col2];
            }

            // Fetch sine and cosine values
            float c = cos_sin_cache[pos * D + j1];
            float s = cos_sin_cache[pos * D + j2];

            // Apply RoPE
            float rope_q1 = q1 * c - q2 * s;
            float rope_q2 = q2 * c + q1 * s;

            // Write out to Q_out
            Q_out[(b * H_q + hq) * D + j1] = rope_q1;
            Q_out[(b * H_q + hq) * D + j2] = rope_q2;
        }
    }

    // --- 2. Compute K_cache and V_cache elements ---
    for (int hk = blockIdx.y; hk < H_kv; hk += gridDim.y) {
        for (int j_half = threadIdx.x; j_half < half_D; j_half += blockDim.x) {
            int j1 = j_half;
            int j2 = j_half + half_D;

            // Global column index in W_qkv for Key and Value components
            int k_col1 = q_elements + hk * D + j1;
            int k_col2 = q_elements + hk * D + j2;
            int v_col1 = q_elements + k_elements + hk * D + j1;
            int v_col2 = q_elements + k_elements + hk * D + j2;

            // Compute Key and Value matrix projections
            float k1 = 0.0f;
            float k2 = 0.0f;
            float v1 = 0.0f;
            float v2 = 0.0f;
            
            for (int d = 0; d < d_model; ++d) {
                float x_val = x[b * d_model + d];
                k1 += x_val * W_qkv[d * total_qkv_cols + k_col1];
                k2 += x_val * W_qkv[d * total_qkv_cols + k_col2];
                v1 += x_val * W_qkv[d * total_qkv_cols + v_col1];
                v2 += x_val * W_qkv[d * total_qkv_cols + v_col2];
            }

            // Fetch sine and cosine values
            float c = cos_sin_cache[pos * D + j1];
            float s = cos_sin_cache[pos * D + j2];

            // Apply RoPE to Key elements
            float rope_k1 = k1 * c - k2 * s;
            float rope_k2 = k2 * c + k1 * s;

            // Cache linear indexing layout: [B, H_kv, S_max, D]
            int cache_idx1 = ((b * H_kv + hk) * S_max + pos) * D + j1;
            int cache_idx2 = ((b * H_kv + hk) * S_max + pos) * D + j2;

            K_cache[cache_idx1] = rope_k1;
            K_cache[cache_idx2] = rope_k2;
            V_cache[cache_idx1] = v1;
            V_cache[cache_idx2] = v2;
        }
    }
}

extern "C" void solve(const float* x, const float* W_qkv, const float* cos_sin_cache,
                      const int* positions, float* K_cache, float* V_cache, float* Q_out, int B,
                      int d_model, int H_q, int H_kv, int D, int S_max) 
{
    // Configure Execution Grid Setup
    // Use block size that pairs efficiently with the half-dimension workload
    int threads_per_block = 128; 
    
    // Grid configuration:
    // dim3.x treats each batch row context independently
    // dim3.y maps across the maximum heads sequence safely
    dim3 block(threads_per_block);
    dim3 grid(B, H_q); 

    fused_qkv_rope_cache_kernel<<<grid, block>>>(
        x, W_qkv, cos_sin_cache, positions, K_cache, V_cache, Q_out, B, d_model, H_q, H_kv, D, S_max
    );
    
    // Ensure kernel executes synchronously or safely matches stream requirements
    cudaDeviceSynchronize();
}
