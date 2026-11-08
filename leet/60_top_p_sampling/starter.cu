#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <float.h>

// PCG hash function on unsigned 32-bit integers
__device__ __forceinline__ uint32_t pcg_hash(uint32_t x) {
    uint32_t state = x * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Convert IEEE-754 float to uint32 such that descending float order
// maps to ascending uint32 order (handles -0.0f and +0.0f identically).
__device__ __forceinline__ uint32_t float_to_descending_key(float f) {
    if (f == 0.0f) f = 0.0f; // Normalize -0.0f to +0.0f
    uint32_t bits = __float_as_uint(f);
    uint32_t mask = (bits & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
    return ~(bits ^ mask);
}

// Kernel 1: Build 64-bit sort keys: upper 32 bits = descending logit key, lower 32 bits = ascending index
__global__ void build_keys_kernel(const float* __restrict__ logits, uint64_t* __restrict__ keys, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vocab_size) {
        uint64_t high = static_cast<uint64_t>(float_to_descending_key(logits[idx]));
        uint64_t low = static_cast<uint64_t>(static_cast<uint32_t>(idx));
        keys[idx] = (high << 32) | low;
    }
}

// Kernel 2: Parallel Bottom-Up Merge Sort step on 64-bit keys
__global__ void merge_sort_step_kernel(const uint64_t* __restrict__ src, uint64_t* __restrict__ dst,
                                       int vocab_size, int width) {
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = pair_idx * (2 * width);
    if (start >= vocab_size) return;

    int mid = min(start + width, vocab_size);
    int end = min(start + 2 * width, vocab_size);

    int i = start;
    int j = mid;
    int k = start;

    while (i < mid && j < end) {
        if (src[i] <= src[j]) {
            dst[k++] = src[i++];
        } else {
            dst[k++] = src[j++];
        }
    }
    while (i < mid) {
        dst[k++] = src[i++];
    }
    while (j < end) {
        dst[k++] = src[j++];
    }
}

// Kernel 3: Block-level local sort for chunks of size 256 using Odd-Even Transposition / Bitonic Sort in Shared Memory
constexpr int CHUNK_SIZE = 256;

__global__ void block_sort_kernel(uint64_t* __restrict__ keys, int vocab_size) {
    __shared__ uint64_t s_data[CHUNK_SIZE];

    int base = blockIdx.x * CHUNK_SIZE;
    int tid = threadIdx.x;

    // Load element or sentinel max value
    s_data[tid] = (base + tid < vocab_size) ? keys[base + tid] : 0xFFFFFFFFFFFFFFFFULL;
    __syncthreads();

    // Bitonic sort within block of 256 elements
    for (int k = 2; k <= CHUNK_SIZE; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                bool ascending = ((tid & k) == 0);
                uint64_t a = s_data[tid];
                uint64_t b = s_data[ixj];
                if ((ascending && a > b) || (!ascending && a < b)) {
                    s_data[tid] = b;
                    s_data[ixj] = a;
                }
            }
            __syncthreads();
        }
    }

    if (base + tid < vocab_size) {
        keys[base + tid] = s_data[tid];
    }
}

// Kernel 4: Softmax, Nucleus Cutoff, and Exponential Race Sampling (Single Block of 512 threads)
constexpr int SAMPLE_THREADS = 512;

__global__ void nucleus_sample_kernel(const float* __restrict__ logits,
                                      const uint64_t* __restrict__ sorted_keys,
                                      const float* __restrict__ p_ptr,
                                      const int* __restrict__ seed_ptr,
                                      int* __restrict__ sampled_token,
                                      int vocab_size) {
    __shared__ double s_reduce[SAMPLE_THREADS];
    __shared__ double s_max_logit;
    __shared__ double s_inv_sum;
    __shared__ int s_nucleus_size;
    __shared__ int s_best_token[SAMPLE_THREADS];
    __shared__ double s_best_score[SAMPLE_THREADS];

    int tid = threadIdx.x;

    // Step 1: Max logit is directly at sorted_keys[0] since keys are sorted by descending logit
    if (tid == 0) {
        int top_idx = static_cast<int>(sorted_keys[0] & 0xFFFFFFFFULL);
        s_max_logit = static_cast<double>(logits[top_idx]);
    }
    __syncthreads();

    double max_logit = s_max_logit;

    // Step 2: Compute sum of exp(logit - max_logit) in double precision
    double local_sum = 0.0;
    for (int i = tid; i < vocab_size; i += SAMPLE_THREADS) {
        int tok_idx = static_cast<int>(sorted_keys[i] & 0xFFFFFFFFULL);
        local_sum += exp(static_cast<double>(logits[tok_idx]) - max_logit);
    }
    s_reduce[tid] = local_sum;
    __syncthreads();

    for (int stride = SAMPLE_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        s_inv_sum = 1.0 / s_reduce[0];
        double p = static_cast<double>(*p_ptr);
        double inv_sum = s_inv_sum;

        // Step 3: Find smallest set where cumulative probability >= p
        double cum_prob = 0.0;
        int cutoff = vocab_size;
        for (int i = 0; i < vocab_size; ++i) {
            int tok_idx = static_cast<int>(sorted_keys[i] & 0xFFFFFFFFULL);
            double prob = exp(static_cast<double>(logits[tok_idx]) - max_logit) * inv_sum;
            cum_prob += prob;
            if (cum_prob >= p) {
                cutoff = i + 1;
                break;
            }
        }
        s_nucleus_size = cutoff;
    }
    __syncthreads();

    // Step 4 & 5: Exponential race over the nucleus tokens
    // Note: Renormalization scales all probabilities by the same constant,
    // which does not change the argmax of q_i / (-ln(u_i)).
    int nucleus_size = s_nucleus_size;
    double inv_sum = s_inv_sum;
    uint32_t seed_u32 = static_cast<uint32_t>(*seed_ptr);
    uint32_t base_hash = pcg_hash(seed_u32);

    double best_score = -1.0;
    int best_tok = static_cast<int>(sorted_keys[0] & 0xFFFFFFFFULL);

    for (int i = tid; i < nucleus_size; i += SAMPLE_THREADS) {
        uint32_t tok_idx = static_cast<uint32_t>(sorted_keys[i] & 0xFFFFFFFFULL);
        double prob = exp(static_cast<double>(logits[tok_idx]) - max_logit) * inv_sum;

        uint32_t h_val = pcg_hash(base_hash + tok_idx);
        double u = (static_cast<double>(h_val) + 0.5) / 4294967296.0; // 2^32
        double neg_ln_u = -log(u);
        double score = prob / neg_ln_u;

        if (score > best_score) {
            best_score = score;
            best_tok = static_cast<int>(tok_idx);
        }
    }

    s_best_score[tid] = best_score;
    s_best_token[tid] = best_tok;
    __syncthreads();

    // Parallel reduction to find global winner
    for (int stride = SAMPLE_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_best_score[tid + stride] > s_best_score[tid]) {
                s_best_score[tid] = s_best_score[tid + stride];
                s_best_token[tid] = s_best_token[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *sampled_token = s_best_token[0];
    }
}

extern "C" void solve(const float* logits, const float* p, const int* seed, int* sampled_token,
                      int vocab_size) {
    uint64_t* d_buf_a = nullptr;
    uint64_t* d_buf_b = nullptr;
    size_t bytes = static_cast<size_t>(vocab_size) * sizeof(uint64_t);

    cudaMalloc(&d_buf_a, bytes);
    cudaMalloc(&d_buf_b, bytes);

    // 1. Build 64-bit composite keys (descending logit, ascending index)
    int num_blocks = (vocab_size + CHUNK_SIZE - 1) / CHUNK_SIZE;
    build_keys_kernel<<<num_blocks, CHUNK_SIZE>>>(logits, d_buf_a, vocab_size);

    // 2. Sort 256-element chunks in shared memory
    block_sort_kernel<<<num_blocks, CHUNK_SIZE>>>(d_buf_a, vocab_size);

    // 3. Merge sorted chunks bottom-up
    uint64_t* src = d_buf_a;
    uint64_t* dst = d_buf_b;
    for (int width = CHUNK_SIZE; width < vocab_size; width <<= 1) {
        int num_pairs = (vocab_size + (2 * width) - 1) / (2 * width);
        int merge_blocks = (num_pairs + 255) / 256;
        merge_sort_step_kernel<<<merge_blocks, 256>>>(src, dst, vocab_size, width);

        uint64_t* tmp = src;
        src = dst;
        dst = tmp;
    }

    // 4. Compute softmax, find nucleus cutoff, and perform exponential race sampling
    nucleus_sample_kernel<<<1, SAMPLE_THREADS>>>(logits, src, p, seed, sampled_token, vocab_size);

    cudaFree(d_buf_a);
    cudaFree(d_buf_b);
}
