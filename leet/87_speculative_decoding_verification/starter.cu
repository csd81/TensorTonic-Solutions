#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Block reduction helper for a sum of floats
__device__ float blockReduceSum(float val) {
    static __shared__ float shared[32]; // Shared memory for up to 1024 threads (32 warps)
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // Read from shared memory only if warp existed
    val = (threadIdx.x < (blockDim.x / 32)) ? shared[lane] : 0.0f;

    if (wid == 0) {
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }

    return val;
}

__global__ void speculative_decoding_kernel(
    const int* __restrict__ draft_tokens,
    const float* __restrict__ draft_probs,
    const float* __restrict__ target_probs,
    const float* __restrict__ uniform_samples,
    int* __restrict__ output_tokens,
    int B, int T, int V) 
{
    // Each block handles exactly one sequence batch element 'b'
    int b = blockIdx.x;
    if (b >= B) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Shared variables for block-wide communication during sequential loop
    __shared__ int stop_loop;
    __shared__ float sum_adj;
    __shared__ int accepted_count;
    __shared__ int selected_token;

    if (tid == 0) {
        stop_loop = 0;
        accepted_count = 0;
    }
    __syncthreads();

    // 1. Process positions 0 to T-1 sequentially
    for (int i = 0; i < T; ++i) {
        if (stop_loop) break;

        // Base offsets for memory access
        int prob_offset_base = b * T * V + i * V;
        int t_i = draft_tokens[b * T + i];

        float p_t = draft_probs[prob_offset_base + t_i];
        float q_t = target_probs[prob_offset_base + t_i];
        
        float alpha = (q_t / p_t < 1.0f) ? (q_t / p_t) : 1.0f;
        float u_i = uniform_samples[b * (T + 1) + i];

        if (u_i < alpha) {
            // Token accepted!
            if (tid == 0) {
                output_tokens[b * (T + 1) + i] = t_i;
                accepted_count++;
            }
            __syncthreads();
        } else {
            // Token rejected. Compute the adjusted distribution across vocabulary.
            float local_max_diff = 0.0f;
            for (int v = tid; v < V; v += block_size) {
                float p_v = draft_probs[prob_offset_base + v];
                float q_v = target_probs[prob_offset_base + v];
                float diff = q_v - p_v;
                if (diff > 0.0f) {
                    local_max_diff += diff;
                }
            }

            // Reduction to get the total normalization denominator
            float total_diff = blockReduceSum(local_max_diff);
            if (tid == 0) {
                sum_adj = total_diff;
                stop_loop = 1; // Mark to break out of the sequential loop
            }
            __syncthreads();

            // Perform Inverse CDF Sampling using the remaining global uniform sample
            float r = uniform_samples[b * (T + 1) + T];
            int sampled_v = -1;

            if (sum_adj > 0.0f) {
                // Compute prefix sum and locate the inverse CDF point cooperatively
                // To minimize shared memory overhead and multi-pass syncs, each thread computes 
                // its own running local accumulation chunk safely via a collective approach
                __shared__ float warp_aggregates[32];
                
                // We chunk the V elements across block runs
                float thread_running_sum = 0.0f;
                bool found = false;

                // Pass 1: Compute cumulative sums locally per thread
                for (int v_base = 0; v_base < V; v_base += block_size) {
                    int v = v_base + tid;
                    float val = 0.0f;
                    if (v < V) {
                        float diff = target_probs[prob_offset_base + v] - draft_probs[prob_offset_base + v];
                        val = (diff > 0.0f) ? (diff / sum_adj) : 0.0f;
                    }

                    // Warp intra-exclusive scan to find dynamic match
                    float warp_sum = val;
                    int lane = tid % 32;
                    int wid = tid / 32;
                    for (int offset = 1; offset < 32; offset *= 2) {
                        float remote = __shfl_up_sync(0xffffffff, warp_sum, offset);
                        if (lane >= offset) warp_sum += remote;
                    }
                    
                    if (lane == 32 - 1) warp_aggregates[wid] = warp_sum;
                    __syncthreads();

                    // Inter-warp prefix sum
                    if (wid == 0 && lane < (block_size / 32)) {
                        float w_val = warp_aggregates[lane];
                        for (int offset = 1; offset < 32; offset *= 2) {
                            float remote = __shfl_up_sync(0xffffffff, w_val, offset);
                            if (lane >= offset) w_val += remote;
                        }
                        warp_aggregates[lane] = w_val;
                    }
                    __syncthreads();

                    float prev_warps_sum = (wid > 0) ? warp_aggregates[wid - 1] : 0.0f;
                    float global_cdf_val = thread_running_sum + prev_warps_sum + warp_sum;

                    if (!found && v < V && (global_cdf_val >= r || v == V - 1)) {
                        // Check if the previous point was below r
                        float prev_val = global_cdf_val - val;
                        if (prev_val < r) {
                            sampled_v = v;
                            found = true;
                        }
                    }

                    // Total block sum of this iteration chunk to carry over
                    float total_block_chunk = blockReduceSum(val);
                    if (tid == 0) warp_aggregates[0] = total_block_chunk;
                    __syncthreads();
                    thread_running_sum += warp_aggregates[0];
                }
            } else {
                // Fallback to Uniform 1/V distribution
                // Index is determined by floor(r * V) safely clamped
                sampled_v = (int)(r * V);
                if (sampled_v >= V) sampled_v = V - 1;
                if (sampled_v < 0) sampled_v = 0;
            }

            // Write the found token out safely from the thread that discovered it
            if (sampled_v != -1) {
                selected_token = sampled_v;
            }
            __syncthreads();

            if (tid == 0) {
                output_tokens[b * (T + 1) + i] = selected_token;
            }
            __syncthreads();
            break; 
        }
    }

    // 2. Bonus Token Step: If all T tokens are successfully accepted, sample a bonus token from q_{T-1}
    if (accepted_count == T) {
        float r = uniform_samples[b * (T + 1) + T];
        int prob_offset_base = b * T * V + (T - 1) * V;
        int sampled_v = -1;
        
        __shared__ float warp_aggregates[32];
        float thread_running_sum = 0.0f;
        bool found = false;

        for (int v_base = 0; v_base < V; v_base += block_size) {
            int v = v_base + tid;
            float val = (v < V) ? target_probs[prob_offset_base + v] : 0.0f;

            float warp_sum = val;
            int lane = tid % 32;
            int wid = tid / 32;
            for (int offset = 1; offset < 32; offset *= 2) {
                float remote = __shfl_up_sync(0xffffffff, warp_sum, offset);
                if (lane >= offset) warp_sum += remote;
            }
            
            if (lane == 32 - 1) warp_aggregates[wid] = warp_sum;
            __syncthreads();

            if (wid == 0 && lane < (block_size / 32)) {
                float w_val = warp_aggregates[lane];
                for (int offset = 1; offset < 32; offset *= 2) {
                    float remote = __shfl_up_sync(0xffffffff, w_val, offset);
                    if (lane >= offset) w_val += remote;
                }
                warp_aggregates[lane] = w_val;
            }
            __syncthreads();

            float prev_warps_sum = (wid > 0) ? warp_aggregates[wid - 1] : 0.0f;
            float global_cdf_val = thread_running_sum + prev_warps_sum + warp_sum;

            if (!found && v < V && (global_cdf_val >= r || v == V - 1)) {
                float prev_val = global_cdf_val - val;
                if (prev_val < r) {
                    sampled_v = v;
                    found = true;
                }
            }

            float total_block_chunk = blockReduceSum(val);
            if (tid == 0) warp_aggregates[0] = total_block_chunk;
            __syncthreads();
            thread_running_sum += warp_aggregates[0];
        }

        if (sampled_v != -1) {
            selected_token = sampled_v;
        }
        __syncthreads();

        if (tid == 0) {
            output_tokens[b * (T + 1) + T] = selected_token;
            accepted_count++;
        }
        __syncthreads();
    }

    // 3. Zero-pad remaining sequence positions
    int start_pad = accepted_count;
    for (int i = start_pad + tid; i < T + 1; i += block_size) {
        output_tokens[b * (T + 1) + i] = 0;
    }
}

extern "C" void solve(const int* draft_tokens, const float* draft_probs, const float* target_probs,
                      const float* uniform_samples, int* output_tokens, int B, int T, int V) 
{
    // 512 threads per block strikes an optimal balance for parallel reductions over large V dimensions
    int threads_per_block = 512;
    int blocks_per_grid = B;

    speculative_decoding_kernel<<<blocks_per_grid, threads_per_block>>>(
