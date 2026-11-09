#include <cuda_runtime.h>

__global__ void logit_penalty_kernel(
    const float* __restrict__ logits, 
    const int* __restrict__ prompt_tokens, 
    const int* __restrict__ output_tokens,
    const float* __restrict__ presence_penalty, 
    const float* __restrict__ frequency_penalty,
    const float* __restrict__ repetition_penalty, 
    float* __restrict__ output, 
    int B, int V, int P, int G) 
{
    // Map each block to one specific batch request row
    int b = blockIdx.x;
    if (b >= B) return;

    // Dynamically sized tracking arrays in shared memory
    // Max constraints: P <= 1024, G <= 512
    extern __shared__ int shared_mem[];
    int* local_seen = shared_mem;               // Size: P + G ints
    int* local_generated = shared_mem + (P + G); // Size: G ints

    int tid = threadIdx.x;
    int bdim = blockDim.x;

    // 1. Cooperatively load all valid prompt & generated tokens into shared memory
    // This allows fast, low-latency checking during the large vocabulary streaming step
    int total_history = P + G;
    for (int i = tid; i < total_history; i += bdim) {
        if (i < P) {
            local_seen[i] = prompt_tokens[b * P + i];
        } else {
            int g_idx = i - P;
            int tok = output_tokens[b * G + g_idx];
            local_seen[i] = tok;
            local_generated[g_idx] = tok;
        }
    }
    __syncthreads();

    // Fetch the scalar penalty parameters for this specific request
    float r = repetition_penalty[b];
    float freq = frequency_penalty[b];
    float pres = presence_penalty[b];

    // 2. Coalesced streaming through the vocabulary entries for row 'b'
    for (int v = tid; v < V; v += bdim) {
        // Compute frequencies and seen status
        int count_c = 0;
        bool seen_s = false;

        // Scan the shared history to evaluate the token 'v'
        for (int i = 0; i < total_history; i++) {
            if (local_seen[i] == v) {
                seen_s = true;
                break;
            }
        }

        if (seen_s) {
            for (int i = 0; i < G; i++) {
                if (local_generated[i] == v) {
                    count_c++;
                }
            }
        }

        // Apply transformations sequentially on the logit value
        float z = logits[b * V + v];

        // Penalty 1: Repetition (multiplicative)
        if (seen_s) {
            if (z > 0.0f) {
                z /= r;
            } else {
                z *= r;
            }
        }

        // Penalty 2: Frequency
        z -= freq * (float)count_c;

        // Penalty 3: Presence
        if (count_c > 0) {
            z -= pres;
        }

        // Write the finalized penalized logit value to the output buffer
        output[b * V + v] = z;
    }
}

extern "C" void solve(const float* logits, const int* prompt_tokens, const int* output_tokens,
                      const float* presence_penalty, const float* frequency_penalty,
                      const float* repetition_penalty, float* output, int B, int V, int P, int G) 
{
    // Determine the necessary dynamic shared memory allocation footprint
    // Storage space required: (P + G) integers for seen items + G integers for counts
    size_t shared_mem_size = (P + 2 * G) * sizeof(int);

    // Using 256 threads per block gives a healthy balance of occupancy and performance
    int threads_per_block = 256;
    int blocks_in_grid = B; // One block dedicated per batch request

    logit_penalty_kernel<<<blocks_in_grid, threads_per_block, shared_mem_size>>>(
        logits, prompt_tokens, output_tokens, 
        presence_penalty, frequency_penalty, repetition_penalty, 
        output, B, V, P, G
    );
}
