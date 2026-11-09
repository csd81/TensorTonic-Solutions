#include <cuda_runtime.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Block dimensions tuning:
// Each block processes a tile of Keys (S_TILE) and a tile of Queries (B_TILE)
#define S_TILE 16
#define B_TILE 16

__global__ void turboquant_attention_kernel(
    const float* __restrict__ Q,           // [B, D]
    const unsigned char* __restrict__ K_idx, // [S, D]
    const signed char* __restrict__ qjl_signs, // [S, D]
    const float* __restrict__ gamma,       // [S]
    const float* __restrict__ Pi,          // [D, D]
    const float* __restrict__ S_mat,       // [D, D]
    const float* __restrict__ codebook,    // [C]
    float* __restrict__ scores,            // [B, S]
    int B, int S, int D, int C
) {
    // Shared memory allocations for staging matrix coordinates statically or dynamically
    // For D=128, a single tile slice fits beautifully within hardware limits
    extern __shared__ float s_mem[];
    
    // We dynamically portion shared memory for caching Q vectors to minimize global lookup
    float* s_Q = (float*)&s_mem[0]; // size: B_TILE * D
    
    int tx = threadIdx.x; // local block dimension across D or S
    int ty = threadIdx.y; // local block dimension across B
    
    int block_s_start = blockIdx.x * S_TILE;
    int block_b_start = blockIdx.y * B_TILE;

    // 1. Cooperative layout loading: Cache Q chunk into shared memory
    for (int b = ty; b < B_TILE; ++b + blockDim.y) {
        int global_b = block_b_start + b;
        if (global_b < B) {
            for (int d = tx; d < D; d += blockDim.x) {
                s_Q[b * D + d] = Q[global_b * D + d];
            }
        }
    }
    __syncthreads();

    // 2. Compute individual Key elements locally and accumulate into the final score
    // To remain highly parallelized without enormous shared memory, each thread handles a Key-Query pair
    int s_idx = block_s_start + tx;
    int b_idx = block_b_start + ty;

    if (s_idx < S && b_idx < B) {
        // Pre-fetch global properties for the current assigned Key
        float g_val = gamma[s_idx];
        float res_scale = (sqrtf(M_PI / 2.0f) / (float)D) * g_val;
        
        float accum_score = 0.0f;

        // Loop over the hidden dimension D to reconstruct K_tilde[d] on the fly
        for (int d = 0; d < D; ++d) {
            float k_mse_d = 0.0f;
            float k_res_d = 0.0f;

            // Compute the vector-matrix products for decompression
            for (int m = 0; m < D; ++m) {
                // MSE part: codebook lookup + Pi multiplication
                unsigned char idx = K_idx[s_idx * D + m];
                float cb_val = codebook[idx];
                k_mse_d += cb_val * Pi[m * D + d];

                // QJL Residual part: signs projection + S_mat multiplication
                float sign_val = (float)qjl_signs[s_idx * D + m];
                k_res_d += sign_val * S_mat[m * D + d];
            }

            float k_tilde_d = k_mse_d + (k_res_d * res_scale);

            // Fetch matched full-precision Query element from shared memory and dot-product
            accum_score += s_Q[(ty) * D + d] * k_tilde_d;
        }

        // Store computed raw float32 result directly into the correct output coordinates
        scores[b_idx * S + s_idx] = accum_score;
    }
}

extern "C" void solve(const float* Q, const unsigned char* K_idx, const signed char* qjl_signs,
                      const float* gamma, const float* Pi, const float* S_mat,
                      const float* codebook, float* scores, int B, int S, int D, int C) {
    
    // Grid sizing setup
    dim3 threadsPerBlock(S_TILE, B_TILE);
    dim3 numBlocks((S + S_TILE - 1) / S_TILE, (B + B_TILE - 1) / B_TILE);
    
    // Allocate dynamic shared memory space for the cached Query tile block
    size_t shared_mem_size = B_TILE * D * sizeof(float);

    turboquant_attention_kernel<<<numBlocks, threadsPerBlock, shared_mem_size>>>(
        Q, K_idx, qjl_signs, gamma, Pi, S_mat, codebook, scores, B, S, D, C
    );
    
    // Ensure accurate pipeline termination
    cudaDeviceSynchronize();
}
