#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// Tuned for D <= 256 and high memory bandwidth on decode workloads
constexpr int BLOCK_S = 128;      // Number of K/V tokens processed per thread block
constexpr int NUM_THREADS = 128;  // 4 warps per block
constexpr int MAX_SPLITS = 256;   // Caps workspace size and reduction loop overhead

// Warp-level sum reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel 1: Compute partial attention outputs & online softmax stats per KV split
__global__ void flash_decode_split_kv(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ partial_out,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    int T, int S, int H, int D,
    int num_splits, int tokens_per_split) {
    
    int split_idx = blockIdx.x;
    int h = blockIdx.y;
    int t = blockIdx.z;

    int s_start = split_idx * tokens_per_split;
    int s_end = min(s_start + tokens_per_split, S);
    if (s_start >= s_end) return;

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;

    // Shared memory layout:
    // q_shared:  [D]
    // scores:    [BLOCK_S]
    // reduce_smem: [4] (for warp reduction)
    extern __shared__ float smem[];
    float* q_shared = smem;
    float* scores = smem + D;
    float* reduce_smem = scores + BLOCK_S;

    // 1. Load Q[t, h, :] into shared memory
    const float* q_ptr = Q + (t * H + h) * D;
    for (int d = tid; d < D; d += NUM_THREADS) {
        q_shared[d] = q_ptr[d];
    }
    __syncthreads();

    float scale = rsqrtf((float)D);

    // Online softmax state for this split
    float m_i = -FLT_MAX;
    float l_i = 0.0f;

    // Register accumulator for output vector (supports D up to 256)
    // Thread tid handles dimensions: tid, tid + 128
    float acc[2] = {0.0f, 0.0f};

    // 2. Iterate over the assigned KV chunk in tiles of BLOCK_S
    for (int tile_start = s_start; tile_start < s_end; tile_start += BLOCK_S) {
        int tile_len = min(BLOCK_S, s_end - tile_start);

        // Step A: Compute dot products S_j = (Q . K_j) / sqrt(D)
        // Each warp computes dot products for a subset of tokens in the tile
        for (int j = warp_id; j < tile_len; j += (NUM_THREADS >> 5)) {
            int s_idx = tile_start + j;
            const float* k_ptr = K + (s_idx * H + h) * D;

            float dot = 0.0f;
            for (int d = lane_id; d < D; d += 32) {
                dot += q_shared[d] * k_ptr[d];
            }
            dot = warp_reduce_sum(dot);

            if (lane_id == 0) {
                scores[j] = dot * scale;
            }
        }
        __syncthreads();

        // Step B: Find tile maximum score across all threads
        float local_max = -FLT_MAX;
        for (int j = tid; j < tile_len; j += NUM_THREADS) {
            local_max = fmaxf(local_max, scores[j]);
        }
        // Warp-reduce max
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
        }
        if (lane_id == 0) reduce_smem[warp_id] = local_max;
        __syncthreads();

        if (tid == 0) {
            float block_max = -FLT_MAX;
#pragma unroll
            for (int w = 0; w < (NUM_THREADS >> 5); ++w) {
                block_max = fmaxf(block_max, reduce_smem[w]);
            }
            reduce_smem[0] = block_max;
        }
        __syncthreads();
        float m_tile = reduce_smem[0];

        // Step C: Update online softmax statistics & rescale previous accumulator
        float m_new = fmaxf(m_i, m_tile);
        float alpha = (m_i == -FLT_MAX) ? 0.0f : expf(m_i - m_new);

        if (tid < D) acc[0] *= alpha;
        if (tid + 128 < D) acc[1] *= alpha;

        // Compute P_j = exp(scores[j] - m_new) and sum(P_j)
        float local_sum = 0.0f;
        for (int j = tid; j < tile_len; j += NUM_THREADS) {
            float p = expf(scores[j] - m_new);
            scores[j] = p;
            local_sum += p;
        }
        local_sum = warp_reduce_sum(local_sum);
        if (lane_id == 0) reduce_smem[warp_id] = local_sum;
        __syncthreads();

        if (tid == 0) {
            float block_sum = 0.0f;
#pragma unroll
            for (int w = 0; w < (NUM_THREADS >> 5); ++w) {
                block_sum += reduce_smem[w];
            }
            reduce_smem[0] = block_sum;
        }
        __syncthreads();

        float l_tile = reduce_smem[0];
        l_i = l_i * alpha + l_tile;
        m_i = m_new;

        // Step D: Accumulate weighted V values: acc += sum_j (P_j * V_j)
        // Coalesced reads across D dimensions
        if (tid < D) {
            float v_acc0 = acc[0];
            float v_acc1 = acc[1];
            bool has_second = (tid + 128 < D);

            for (int j = 0; j < tile_len; ++j) {
                float p = scores[j];
                int s_idx = tile_start + j;
                const float* v_ptr = V + (s_idx * H + h) * D;
                v_acc0 += p * v_ptr[tid];
                if (has_second) {
                    v_acc1 += p * v_ptr[tid + 128];
                }
            }
            acc[0] = v_acc0;
            acc[1] = v_acc1;
        }
        __syncthreads();
    }

    // 3. Write partial results to global workspace
    int split_linear_idx = (t * H + h) * num_splits + split_idx;
    if (tid == 0) {
        partial_m[split_linear_idx] = m_i;
        partial_l[split_linear_idx] = l_i;
    }

    float* out_ptr = partial_out + split_linear_idx * D;
    if (tid < D) {
        out_ptr[tid] = acc[0];
    }
    if (tid + 128 < D) {
        out_ptr[tid + 128] = acc[1];
    }
}

// Kernel 2: Combine partial splits using log-sum-exp rescaling
__global__ void flash_decode_reduce(
    const float* __restrict__ partial_out,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    float* __restrict__ output,
    int H, int D, int num_splits) {
    
    int h = blockIdx.x;
    int t = blockIdx.y;
    int tid = threadIdx.x;

    int base_idx = (t * H + h) * num_splits;

    // 1. Find global max across all valid splits
    float global_m = -FLT_MAX;
    for (int s = 0; s < num_splits; ++s) {
        float m_s = partial_m[base_idx + s];
        if (m_s > global_m) {
            global_m = m_s;
        }
    }

    // 2. Compute global denominator and accumulate rescaled partial outputs
    float global_l = 0.0f;
    float out0 = 0.0f;
    float out1 = 0.0f;
    bool has_first = (tid < D);
    bool has_second = (tid + 128 < D);

    for (int s = 0; s < num_splits; ++s) {
        float l_s = partial_l[base_idx + s];
        if (l_s > 0.0f) {
            float m_s = partial_m[base_idx + s];
            float weight = expf(m_s - global_m);
            global_l += l_s * weight;

            const float* p_out_ptr = partial_out + (base_idx + s) * D;
            if (has_first) out0 += p_out_ptr[tid] * weight;
            if (has_second) out1 += p_out_ptr[tid + 128] * weight;
        }
    }

    // 3. Normalize and write final output
    float inv_l = (global_l > 0.0f) ? (1.0f / global_l) : 0.0f;
    float* final_out_ptr = output + (t * H + h) * D;

    if (has_first) final_out_ptr[tid] = out0 * inv_l;
    if (has_second) final_out_ptr[tid + 128] = out1 * inv_l;
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int T, int S,
                      int H, int D) {
    // Determine optimal split count based on sequence length S and problem concurrency (T * H)
    int num_kv_blocks = (S + BLOCK_S - 1) / BLOCK_S;
    int target_total_blocks = 256; // Ensures high SM occupancy across modern GPUs
    int concurrency = T * H;
    int desired_splits = (target_total_blocks + concurrency - 1) / concurrency;

    int num_splits = min(num_kv_blocks, max(1, min(MAX_SPLITS, desired_splits)));
    // For very long contexts (like S = 65536), ensure enough splits to maximize DRAM bandwidth
    if (S >= 4096) {
        num_splits = min(num_kv_blocks, max(num_splits, 64));
    }

    int tokens_per_split = ((S + num_splits - 1) / num_splits + BLOCK_S - 1) / BLOCK_S * BLOCK_S;
    num_splits = (S + tokens_per_split - 1) / tokens_per_split;

    // Allocate temporary workspace for partial outputs and softmax statistics
    size_t total_splits = (size_t)T * H * num_splits;
    size_t partial_out_bytes = total_splits * D * sizeof(float);
    size_t partial_stat_bytes = total_splits * sizeof(float);

    float* workspace = nullptr;
    cudaMalloc(&workspace, partial_out_bytes + 2 * partial_stat_bytes);

    float* partial_out = workspace;
    float* partial_m = workspace + total_splits * D;
    float* partial_l = partial_m + total_splits;

    // Launch Kernel 1: Split-KV Online Softmax Attention
    dim3 grid1(num_splits, H, T);
    dim3 block1(NUM_THREADS);
    size_t smem_bytes = (D + BLOCK_S + (NUM_THREADS >> 5)) * sizeof(float);

    flash_decode_split_kv<<<grid1, block1, smem_bytes>>>(
        Q, K, V, partial_out, partial_m, partial_l,
        T, S, H, D, num_splits, tokens_per_split);

    // Launch Kernel 2: Cross-Split Reduction
    dim3 grid2(H, T);
    dim3 block2(NUM_THREADS);

    flash_decode_reduce<<<grid2, block2>>>(
        partial_out, partial_m, partial_l, output,
        H, D, num_splits);

    cudaFree(workspace);
}
