#include <cuda_runtime.h>
#include <math.h>

// Macro for block sizes
#define BLOCK_SIZE 256

// Kernel to compute Group-Normalized Advantages over G responses per prompt
__global__ void compute_advantages_kernel(const float* rewards, float* advantages, int B, int G) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    // 1. Calculate Mean
    float sum = 0.0f;
    for (int g = 0; g < G; ++g) {
        sum += rewards[b * G + g];
    }
    float mean = sum / G;

    // 2. Calculate Population Standard Deviation
    float variance_sum = 0.0f;
    for (int g = 0; g < G; ++g) {
        float diff = rewards[b * G + g] - mean;
        variance_sum += diff * diff;
    }
    float std_dev = sqrtf(variance_sum / G);

    // 3. Compute and write normalized advantages
    for (int g = 0; g < G; ++g) {
        advantages[b * G + g] = (rewards[b * G + g] - mean) / (std_dev + 1e-8f);
    }
}

// Kernel to compute token-level objectives and accumulate them via parallel reduction
__global__ void compute_loss_kernel(const float* advantages, const float* log_pi, 
                                    const float* log_pi_old, const float* log_ref, 
                                    float* block_losses, float clip_eps, float beta, 
                                    int B, int G, int S) {
    // Each block handles one specific response group (b, g)
    int bg_idx = blockIdx.x; 
    if (bg_idx >= B * G) return;

    int tid = threadIdx.x;
    float local_sum = 0.0f;

    // Fetch the pre-computed advantage for this response group
    float adv = advantages[bg_idx];

    // Grid-stride loop over the sequence/tokens dimension S
    for (int s = tid; s < S; s += blockDim.x) {
        int idx = bg_idx * S + s;

        float l_pi = log_pi[idx];
        float l_pi_old = log_pi_old[idx];
        float l_ref = log_ref[idx];

        // 1. Clipped Surrogate Objective
        float ratio = expf(l_pi - l_pi_old);
        float clipped_ratio = fminf(fmaxf(ratio, 1.0f - clip_eps), 1.0f + clip_eps);
        float l_clip = fminf(ratio * adv, clipped_ratio * adv);

        // 2. KL Divergence Penalty (k3 estimator)
        float d = l_ref - l_pi;
        float k = expf(d) - d - 1.0f;

        // GRPO Objective = L_clip - beta * K
        local_sum += (l_clip - beta * k);
    }

    // Shared memory for block-level reduction
    __shared__ float sdata[BLOCK_SIZE];
    sdata[tid] = local_sum;
    __syncthreads();

    // Tree reduction inside the block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the accumulated objective for this block to global memory
    if (tid == 0) {
        block_losses[bg_idx] = sdata[0];
    }
}

// Final reduction kernel to sum block results and compute final negated average loss
__global__ void finalize_loss_kernel(const float* block_losses, float* output, int total_elements, float total_count_inv) {
    int tid = threadIdx.x;
    float global_sum = 0.0f;

    // Grid-stride loop to sum up all block-level elements
    for (int i = tid; i < total_elements; i += blockDim.x) {
        global_sum += block_losses[i];
    }

    __shared__ float sdata[BLOCK_SIZE];
    sdata[tid] = global_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Negate the final average since GRPO maximizes, but we are minimizing loss
    if (tid == 0) {
        output[0] = -sdata[0] * total_count_inv;
    }
}

extern "C" void solve(const float* rewards, const float* log_pi, const float* log_pi_old,
                      const float* log_ref, float* output, float clip_eps, float beta, int B, int G,
                      int S) {
    
    // Allocate temporary device buffers for transient stages
    float* d_advantages = nullptr;
    float* d_block_losses = nullptr;
    cudaMalloc(&d_advantages, B * G * sizeof(float));
    cudaMalloc(&d_block_losses, B * G * sizeof(float));

    // Step 1: Compute group-normalized rewards (Advantages)
    int threads_per_block_adv = 64;
    int blocks_per_grid_adv = (B + threads_per_block_adv - 1) / threads_per_block_adv;
    compute_advantages_kernel<<<blocks_per_grid_adv, threads_per_block_adv>>>(rewards, d_advantages, B, G);

    // Step 2: Compute token-level clipped surrogate + reference KL loss values
    // Map exactly 1 block to each response sequence (B * G total blocks)
    int blocks_per_grid_loss = B * G;
    compute_loss_kernel<<<blocks_per_grid_loss, BLOCK_SIZE>>>(d_advantages, log_pi, log_pi_old, log_ref, d_block_losses, clip_eps, beta, B, G, S);

    // Step 3: Perform final reduction over all blocks into output[0]
    float total_count_inv = 1.0f / (float)(B * G * S);
    finalize_loss_kernel<<<1, BLOCK_SIZE>>>(d_block_losses, output, B * G, total_count_inv);

    // Synchronize and free transient buffers
    cudaDeviceSynchronize();
    cudaFree(d_advantages);
    cudaFree(d_block_losses);
}
