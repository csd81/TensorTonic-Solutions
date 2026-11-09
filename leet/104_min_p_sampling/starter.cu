#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Block-level reduction helpers using warp shuffles and shared memory
__device__ float blockMax(float val, cg::thread_block ctb) {
    __shared__ float shared_max[32]; // Maximum possible warps in a block of 1024 threads is 32
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }

    if (lane == 0) {
        shared_max[warp_id] = val;
    }
    ctb.sync();

    // Final reduction over warp results
    val = (threadIdx.x < (ctb.num_threads() / 32)) ? shared_max[lane] : -CUDART_INF_F;
    if (warp_id == 0) {
        for (int offset = 16; offset > 0; offset /= 2) {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
    }
    return val;
}

__device__ float blockSum(float val, cg::thread_block ctb) {
    __shared__ float shared_sum[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if (lane == 0) {
        shared_sum[warp_id] = val;
    }
    ctb.sync();

    // Final reduction over warp results
    val = (threadIdx.x < (ctb.num_threads() / 32)) ? shared_sum[lane] : 0.0f;
    if (warp_id == 0) {
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }
    return val;
}

__global__ void min_p_sampling_kernel(const float* __restrict__ logits, float* __restrict__ probs, float min_p, int B, int V) {
    cg::thread_block ctb = cg::this_thread_block();
    int b = blockIdx.x; // One block handles one row
    if (b >= B) return;

    int tid = threadIdx.x;
    int b_offset = b * V;

    // --- Step 1: Find the maximum logit in the row for numerical stability ---
    float max_logit = -CUDART_INF_F;
    for (int v = tid; v < V; v += blockDim.x) {
        max_logit = fmaxf(max_logit, logits[b_offset + v]);
    }
    // Broadcast the true global max across the entire block
    __shared__ float s_max_logit;
    float res_max = blockMax(max_logit, ctb);
    if (tid == 0) {
        s_max_logit = res_max;
    }
    ctb.sync();
    max_logit = s_max_logit;

    // --- Step 2: Compute exponentiated values and find raw max probability element ---
    // Note: Since p_i = exp(z_i - max_z) / sum(exp(z_j - max_z)), the token with the maximum 
    // probability is identical to the token with the maximum logit value. 
    // Thus, p_max_unnormalized = exp(max_logit - max_logit) = 1.0f.
    // The threshold condition: p_i < min_p * p_max is equivalent to:
    // exp(z_i - max_z) / sum_exp < min_p * 1.0f / sum_exp  ==>  exp(z_i - max_z) < min_p
    float sum_exp = 0.0f;
    for (int v = tid; v < V; v += blockDim.x) {
        float unnormalized_p = expf(logits[b_offset + v] - max_logit);
        
        // --- Step 3: Apply the Min-P Cutoff Mask ---
        if (unnormalized_p < min_p) {
            unnormalized_p = 0.0f;
        }
        probs[b_offset + v] = unnormalized_p; // Temporarily store unnormalized/masked probabilities
        sum_exp += unnormalized_p;
    }

    // --- Step 4: Renormalize surviving elements ---
    __shared__ float s_sum_exp;
    float res_sum = blockSum(sum_exp, ctb);
    if (tid == 0) {
        s_sum_exp = res_sum;
    }
    ctb.sync();
    sum_exp = s_sum_exp;

    // Safe division rule in case all inputs get filtered or sum to 0
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;

    for (int v = tid; v < V; v += blockDim.x) {
        probs[b_offset + v] *= inv_sum;
    }
}

extern "C" void solve(const float* logits, float* probs, float min_p, int B, int V) {
    // 512 threads per block strikes a good balance between grid size and shared memory layout
    int threads_per_block = 512;
    int blocks_per_grid = B; // One block per row for batch independence

    min_p_sampling_kernel<<<blocks_per_grid, threads_per_block>>>(logits, probs, min_p, B, V);
}
