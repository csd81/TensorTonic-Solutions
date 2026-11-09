#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <math.h>

#define WARP_SIZE 32

// Utility function for block-wide maximum reduction
__device__ __forceinline__ float blockReduceMax(float val) {
    static __shared__ float shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    // Warp-level reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // Final warp reduction
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : -CUDART_INF_F;
    if (wid == 0) {
        for (int offset = (blockDim.x / WARP_SIZE) / 2; offset > 0; offset /= 2) {
            val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
        }
    }
    return val;
}

// Utility function for block-wide sum reduction
__device__ __forceinline__ float blockReduceSum(float val) {
    static __shared__ float shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
    if (wid == 0) {
        for (int offset = (blockDim.x / WARP_SIZE) / 2; offset > 0; offset /= 2) {
            val += __shfl_xor_sync(0xffffffff, val, offset);
        }
    }
    return val;
}

__global__ void verification_attention_kernel(
    const float* __restrict__ Q, 
    const float* __restrict__ K_cache, 
    const float* __restrict__ V_cache,
    const float* __restrict__ K_tree, 
    const float* __restrict__ V_tree, 
    const int* __restrict__ parents, 
    float* __restrict__ output,
    int P, int T, int H, int D, float scale) 
{
    // Map each block to one specific (node i, head h)
    int i = blockIdx.y; 
    int h = blockIdx.x; 
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    // Step 1: Trace root-to-node ancestry path for tree node i
    // Maximum theoretical path depth is T (up to 128 elements)
    __shared__ int path[128];
    __shared__ int path_len;
    if (tid == 0) {
        int curr = i;
        int len = 0;
        int temp_path[128];
        while (curr != -1) {
            temp_path[len++] = curr;
            curr = parents[curr];
        }
        path_len = len;
        for (int idx = 0; idx < len; ++idx) {
            path[idx] = temp_path[len - 1 - idx];
        }
    }
    __syncthreads();

    int len = path_len;

    // Step 2: Compute running maximum score for numerical stability
    float local_max = -CUDART_INF_F;

    // Context Cache Pass (Tokens 0 to P-1)
    for (int j = tid; j < P; j += bdim) {
        float score = 0.0f;
        for (int d = 0; d < D; ++d) {
            float q_val = Q[((i * H + h) * D) + d];
            float k_val = K_cache[((j * H + h) * D) + d];
            score += q_val * k_val;
        }
        score *= scale;
        if (score > local_max) local_max = score;
    }

    // Tree Ancestry Nodes Pass
    for (int p_idx = tid; p_idx < len; p_idx += bdim) {
        int target_node = path[p_idx];
        float score = 0.0f;
        for (int d = 0; d < D; ++d) {
            float q_val = Q[((i * H + h) * D) + d];
            float k_val = K_tree[((target_node * H + h) * D) + d];
            score += q_val * k_val;
        }
        score *= scale;
        if (score > local_max) local_max = score;
    }

    // Stabilize and sync maximum across the whole block
    float global_max = blockReduceMax(local_max);
    __shared__ float shared_max;
    if (tid == 0) shared_max = global_max;
    __syncthreads();
    global_max = shared_max;

    // Step 3: Compute denominator sum of exponentials
    float local_sum = 0.0f;

    for (int j = tid; j < P; j += bdim) {
        float score = 0.0f;
        for (int d = 0; d < D; ++d) {
            score += Q[((i * H + h) * D) + d] * K_cache[((j * H + h) * D) + d];
        }
        score *= scale;
        local_sum += expf(score - global_max);
    }

    for (int p_idx = tid; p_idx < len; p_idx += bdim) {
        int target_node = path[p_idx];
        float score = 0.0f;
        for (int d = 0; d < D; ++d) {
            score += Q[((i * H + h) * D) + d] * K_tree[((target_node * H + h) * D) + d];
        }
        score *= scale;
        local_sum += expf(score - global_max);
    }

    float global_sum = blockReduceSum(local_sum);
    __shared__ float shared_sum;
    if (tid == 0) shared_sum = global_sum;
    __syncthreads();
    global_sum = shared_sum;

    // Step 4: Compute weighted attention output values across feature dimensions
    for (int d = tid; d < D; d += bdim) {
        float out_val = 0.0f;

        // Context items accumulation
        for (int j = 0; j < P; ++j) {
            float score = 0.0f;
            for (int k = 0; k < D; ++k) {
                score += Q[((i * H + h) * D) + k] * K_cache[((j * H + h) * D) + k];
            }
            score *= scale;
            float weight = expf(score - global_max) / global_sum;
            out_val += weight * V_cache[((j * H + h) * D) + d];
        }

        // Ancestor tree items accumulation
        for (int p_idx = 0; p_idx < len; ++p_idx) {
            int target_node = path[p_idx];
            float score = 0.0f;
            for (int k = 0; k < D; ++k) {
                score += Q[((i * H + h) * D) + k] * K_tree[((target_node * H + h) * D) + k];
            }
            score *= scale;
            float weight = expf(score - global_max) / global_sum;
            out_val += weight * V_tree[((target_node * H + h) * D) + d];
        }

        // Write resulting projection to global memory
        output[((i * H + h) * D) + d] = out_val;
    }
}

extern "C" void solve(const float* Q, const float* K_cache, const float* V_cache,
                      const float* K_tree, const float* V_tree, const int* parents, float* output,
                      int P, int T, int H, int D) 
{
    float scale = 1.0f / sqrtf((float)D);
    
    // Grid sizing optimized for execution parallelism properties
    dim3 grid(H, T); 
    int threads = 128; // Standard flexible execution boundary block size

    verification_attention_kernel<<<grid, threads>>>(
        Q, K_cache, V_cache, K_tree, V_tree, parents, output, P, T, H, D, scale
    );
}
