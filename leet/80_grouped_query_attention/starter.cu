#include <cuda_runtime.h>
#include <math.h>

__global__ void gqa_kernel(const float* __restrict__ Q, 
                           const float* __restrict__ K, 
                           const float* __restrict__ V, 
                           float* __restrict__ output,
                           int num_q_heads, int num_kv_heads, int seq_len, int head_dim) {
    
    int q_head = blockIdx.y;
    int m = blockIdx.x; // current query token row index

    if (m >= seq_len || q_head >= num_q_heads) return;

    int heads_per_group = num_q_heads / num_kv_heads;
    int kv_head = q_head / heads_per_group;
    float scale = 1.0f / sqrtf((float)head_dim);

    // Dynamic pointers for current query sequence row
    const float* q_ptr = Q + (q_head * seq_len + m) * head_dim;
    
    // Shared memory for block-wide reductions
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    // --- Pass 1: Compute attention scores, track Max for Softmax Stability ---
    float local_max = -1e20f;
    
    // Each thread evaluates a subset of the keys (n dimensions)
    for (int n = tid; n < seq_len; n += blockDim.x) {
        const float* k_ptr = K + (kv_head * seq_len + n) * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q_ptr[d] * k_ptr[d];
        }
        score *= scale;
        
        if (score > local_max) {
            local_max = score;
        }
    }

    // Block reduction to find global maximum score across all threads
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }
    float max_score = sdata[0];
    __syncthreads();

    // --- Pass 2: Compute Sum of Exponentials (Softmax Denominator) ---
    float local_sum = 0.0f;
    for (int n = tid; n < seq_len; n += blockDim.x) {
        const float* k_ptr = K + (kv_head * seq_len + n) * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q_ptr[d] * k_ptr[d];
        }
        score = expf(score * scale - max_score);
        local_sum += score;
    }

    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float sum_exp = sdata[0];
    __syncthreads();

    // --- Pass 3: Compute Weighted Sum of Values (Output Matrix) ---
    // Initialize output accumulation registers
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float out_val = 0.0f;
        
        for (int n = 0; n < seq_len; ++n) {
            // Recompute the specific row attention weight safely
            const float* k_ptr = K + (kv_head * seq_len + n) * head_dim;
            float score = 0.0f;
            for (int k_d = 0; k_d < head_dim; ++k_d) {
                score += q_ptr[k_d] * k_ptr[k_d];
            }
            float weight = expf(score * scale - max_score) / sum_exp;
            
            const float* v_ptr = V + (kv_head * seq_len + n) * head_dim;
            out_val += weight * v_ptr[d];
        }
        
        float* out_ptr = output + (q_head * seq_len + m) * head_dim;
        out_ptr[d] = out_val;
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output,
                      int num_q_heads, int num_kv_heads, int seq_len, int head_dim) {
    
    // 256 threads balance register allocation and reduction speed perfectly
    int threads = 256; 
    dim3 grid(seq_len, num_q_heads);
    int shared_mem_size = threads * sizeof(float);

    gqa_kernel<<<grid, threads, shared_mem_size>>>(Q, K, V, output, num_q_heads, num_kv_heads, seq_len, head_dim);
    cudaDeviceSynchronize();
}
