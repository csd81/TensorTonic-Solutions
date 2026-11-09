#include <cuda_runtime.h>

#define MAX_K 32

// Structure to store a single candidate's score and tracking indices
struct Candidate {
    float score;
    int parent_k;
    int token_v;
};

// Insert a candidate into a thread's local sorted array of top-K elements
__device__ void insert_candidate(Candidate* local_top, const Candidate& cand, int K) {
    if (cand.score <= local_top[K - 1].score) return;
    
    int i = K - 1;
    while (i > 0 && cand.score > local_top[i - 1].score) {
        local_top[i] = local_top[i - 1];
        i--;
    }
    local_top[i] = cand;
}

// Merge two sorted arrays of size K into the first array
__device__ void merge_arrays(Candidate* a, const Candidate* b, int K) {
    Candidate temp[MAX_K];
    int i = 0, j = 0, k = 0;
    while (k < K) {
        if (i < K && (j >= K || a[i].score >= b[j].score)) {
            temp[k++] = a[i++];
        } else {
            temp[k++] = b[j++];
        }
    }
    for (int m = 0; m < K; m++) {
        a[m] = temp[m];
    }
}

__global__ void beam_search_kernel(const float* __restrict__ beam_scores,
                                   const float* __restrict__ token_logprobs,
                                   float* __restrict__ new_beam_scores,
                                   int* __restrict__ parent_beam_indices,
                                   int* __restrict__ next_tokens,
                                   int K, int V) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    // Track thread-local top K candidates
    Candidate local_top[MAX_K];
    for (int i = 0; i < K; i++) {
        local_top[i].score = -1e30f; // Initialize to negative infinity
        local_top[i].parent_k = -1;
        local_top[i].token_v = -1;
    }

    // Strided loop over all K * V candidates for this batch
    int total_candidates = K * V;
    for (int idx = tid; idx < total_candidates; idx += bdim) {
        int k = idx / V;
        int v = idx % V;

        float b_score = beam_scores[batch_idx * K + k];
        float t_logprob = token_logprobs[(batch_idx * K + k) * V + v];
        
        Candidate cand;
        cand.score = b_score + t_logprob;
        cand.parent_k = k;
        cand.token_v = v;

        insert_candidate(local_top, cand, K);
    }

    // Shared memory allocated dynamically for block reduction
    extern __shared__ char shared_mem[];
    Candidate* shared_cand = reinterpret_cast<Candidate*>(shared_mem);

    // Copy local top K to shared memory block
    for (int i = 0; i < K; i++) {
        shared_cand[tid * K + i] = local_top[i];
    }
    __syncthreads();

    // Parallel reduction tree within the thread block
    for (int stride = bdim / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            merge_arrays(&shared_cand[tid * K], &shared_cand[(tid + stride) * K], K);
        }
        __syncthreads();
    }

    // Thread 0 writes the globally accurate batch top-K elements to global memory
    if (tid == 0) {
        for (int i = 0; i < K; i++) {
            int out_idx = batch_idx * K + i;
            new_beam_scores[out_idx] = shared_cand[i].score;
            parent_beam_indices[out_idx] = shared_cand[i].parent_k;
            next_tokens[out_idx] = shared_cand[i].token_v;
        }
    }
}

extern "C" void solve(const float* beam_scores, const float* token_logprobs, float* new_beam_scores,
                      int* parent_beam_indices, int* next_tokens, int B, int K, int V) {
    // Determine an optimal number of threads per block (must be a power of 2 for reduction)
    int threads_per_block = 256;
    if (V < 256) threads_per_block = 128;
    if (V < 128) threads_per_block = 64;

    // Shared memory dimension dynamically configured: threads_per_block * K * sizeof(Candidate)
    size_t shared_mem_size = threads_per_block * K * sizeof(Candidate);

    beam_search_kernel<<<B, threads_per_block, shared_mem_size>>>(
        beam_scores, token_logprobs, new_beam_scores, parent_beam_indices, next_tokens, K, V
    );
}
