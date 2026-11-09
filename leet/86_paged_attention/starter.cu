#include <cuda_runtime.h>
#include <math.h>

// Coaxial warp reduction helpers using shuffle instructions
__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void paged_decode_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K_cache,
    const float* __restrict__ V_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ context_lens,
    float* __restrict__ output,
    int num_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq) 
{
    // Each thread block is dedicated to computing attention for one sequence and one head
    int s = blockIdx.x; // sequence index
    int h = blockIdx.y; // head index

    int len = context_lens[s];
    if (len <= 0) return;

    float scale = 1.0f / sqrtf((float)head_dim);

    // Dynamic shared memory allocations
    // 1. Storage for the local Query vector for this specific head (size: head_dim floats)
    // 2. Storage for reductions during softmax (size: blockDim.x floats)
    extern __shared__ char shared_mem[];
    float* q_local = (float*)shared_mem;
    float* smem_red = (float*)(shared_mem + head_dim * sizeof(float));

    int tid = threadIdx.x;

    // 1. Collaboratively load the Query vector Q[s, h] into shared memory
    for (int i = tid; i < head_dim; i += blockDim.x) {
        q_local[i] = Q[(s * num_heads + h) * head_dim + i];
    }
    __syncthreads();

    // 2. Phase 1: Compute raw attention scores & find local maximum
    float local_max = -1e20f;
    const int* seq_block_table = block_table + s * max_blocks_per_seq;

    // Grid-stride loop across all tokens in the current sequence context
    for (int t = tid; t < len; t += blockDim.x) {
        int logical_block_idx = t / block_size;
        int block_offset = t % block_size;
        int physical_block_idx = seq_block_table[logical_block_idx];

        // Locate start of the key vector in the K_cache pool
        // Shape: (num_blocks, block_size, num_heads, head_dim)
        int k_offset = ((physical_block_idx * block_size + block_offset) * num_heads + h) * head_dim;
        const float* k_ptr = K_cache + k_offset;

        // Compute dot product Q_local . K
        float score = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            score += q_local[i] * k_ptr[i];
        }
        score *= scale;

        // Keep track of the maximum score encountered by this thread
        local_max = fmaxf(local_max, score);
    }

    // Block-wide reduction to find the true global maximum score
    float warp_max = warpReduceMax(local_max);
    if ((tid % 32) == 0) {
        smem_red[tid / 32] = warp_max;
    }
    __syncthreads();

    float global_max = (tid < blockDim.x / 32) ? smem_red[tid] : -1e20f;
    if (tid < 32) {
        global_max = warpReduceMax(global_max);
    }
    global_max = __shfl_sync(0xffffffff, global_max, 0);
    __syncthreads(); // shared memory ready for reused/sum reduction

    // 3. Phase 2: Compute exp(score - max) and calculate denominator sum
    float exp_sum = 0.0f;
    for (int t = tid; t < len; t += blockDim.x) {
        int logical_block_idx = t / block_size;
        int block_offset = t % block_size;
        int physical_block_idx = seq_block_table[logical_block_idx];

        int k_offset = ((physical_block_idx * block_size + block_offset) * num_heads + h) * head_dim;
        const float* k_ptr = K_cache + k_offset;

        float score = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            score += q_local[i] * k_ptr[i];
        }
        score *= scale;

        exp_sum += expf(score - global_max);
    }

    // Block-wide reduction to find the sum of all exponentials (softmax denominator)
    float warp_sum = warpReduceSum(exp_sum);
    if ((tid % 32) == 0) {
        smem_red[tid / 32] = warp_sum;
    }
    __syncthreads();

    float global_sum = (tid < blockDim.x / 32) ? smem_red[tid] : 0.0f;
    if (tid < 32) {
        global_sum = warpReduceSum(global_sum);
    }
    global_sum = __shfl_sync(0xffffffff, global_sum, 0);

    // 4. Phase 3: Multiply softmax weights by Values and accumulate into local output vectors
    // Allocate space in shared memory dynamically for the thread block output accumulator
    float* out_local = q_local; // Safely reuse q_local storage since Q vector isn't needed anymore
    for (int i = tid; i < head_dim; i += blockDim.x) {
        out_local[i] = 0.0f;
    }
    __syncthreads();

    // Accumulate weighted values element-by-element across context elements
    for (int t = 0; t < len; ++t) {
        int logical_block_idx = t / block_size;
        int block_offset = t % block_size;
        int physical_block_idx = seq_block_table[logical_block_idx];

        // Recompute the score for weighting inside a predictable loop layout
        int k_offset = ((physical_block_idx * block_size + block_offset) * num_heads + h) * head_dim;
        const float* k_ptr = K_cache + k_offset;

        float score = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            score += q_local[i] * k_ptr[i]; // using original Q weights
        }
        score *= scale;

        float weight = expf(score - global_max) / (global_sum + 1e-6f);

        // Fetch physical values from V_cache pool
        int v_offset = ((physical_block_idx * block_size + block_offset) * num_heads + h) * head_dim;
        const float* v_ptr = V_cache + v_offset;

        // Distribute adding columns across block threads
        for (int i = tid; i < head_dim; i += blockDim.x) {
            atomicAdd(&out_local[i], weight * v_ptr[i]);
        }
    }
    __syncthreads();

    // 5. Write final computed row out to the global output tensor
    for (int i = tid; i < head_dim; i += blockDim.x) {
        output[(s * num_heads + h) * head_dim + i] = out_local[i];
    }
}

extern "C" void solve(const float* Q, const float* K_cache, const float* V_cache,
                      const int* block_table, const int* context_lens, float* output,
                      int batch_size, int num_heads, int head_dim, int block_size,
                      int max_blocks_per_seq) 
{
    // Launch configured with 128 threads per block (4 full warps)
    int threads = 128;
    dim3 grid(batch_size, num_heads);

    // Calculate dynamic shared memory footprint: 
    // Q vector buffer (head_dim) + Max reduction workspace blocks (threads/32)
    size_t shared_mem_size = head_dim * sizeof(float) + (threads / 32) * sizeof(float);

    paged_decode_attention_kernel<<<grid, threads, shared_mem_size>>>(
        Q, K_cache, V_cache, block_table, context_lens, output,
        num_heads, head_dim, block_size, max_blocks_per_seq
    );
}
