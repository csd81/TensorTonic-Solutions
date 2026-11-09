#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Kernel to count tokens per expert and compute the local slot index for each token
__global__ void compute_slots_kernel(const int* expert_idx, int* token_slots, int* token_counts, int T, int E) {
    // Zero out token_counts using the first few threads
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < E) {
        token_counts[tid] = 0;
    }
    __syncthreads();

    // Use a single thread to sequentially compute assignments to guarantee stable ordering
    // For T up to 65,536, a single-pass or block-level reduction is fast, 
    // but a simple sequential pass on a single warp/block avoids multi-pass scan overhead.
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int t = 0; t < T; ++t) {
            int exp = expert_idx[t];
            if (exp >= 0 && exp < E) {
                token_slots[t] = token_counts[exp];
                token_counts[exp]++;
            }
        }
    }
}

// Kernel to dispatch token features into the expert buffers
__global__ void dispatch_kernel(const float* x, const int* expert_idx, const int* token_slots, 
                                float* dispatched_x, int T, int D, int capacity) {
    // Each block processes one token
    int t = blockIdx.x; 
    if (t >= T) return;

    int exp = expert_idx[t];
    int slot = token_slots[t];

    // Total elements per expert matrix row/slice is capacity * D
    // The target position for this token feature array is:
    // dispatched_x[exp * capacity * D + slot * D + d]
    int target_base = exp * capacity * D + slot * D;
    int src_base = t * D;

    // Cooperatively copy D features using all threads in the block
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        dispatched_x[target_base + d] = x[src_base + d];
    }
}

extern "C" void solve(const float* x, const int* expert_idx, float* dispatched_x, int* token_counts,
                      int T, int D, int E, int capacity) {
    // Allocate a temporary device buffer to store the assigned slot index for each token
    int* d_token_slots = nullptr;
    cudaMalloc(&d_token_slots, T * sizeof(int));

    // Step 1: Compute counts and target slots per token
    // Using a single block for the sequential prefix tracking ensures perfect original relative order
    compute_slots_kernel<<<1, 256>>>(expert_idx, d_token_slots, token_counts, T, E);
    cudaDeviceSynchronize();

    // Step 2: Dispatch token features in parallel
    // One block per token, 256 threads per block to cooperatively move D features
    int threadsPerBlock = 256;
    int blocksPerGrid = T;
    dispatch_kernel<<<blocksPerGrid, threadsPerBlock>>>(x, expert_idx, d_token_slots, dispatched_x, T, D, capacity);
    cudaDeviceSynchronize();

    // Clean up temporary workspace
    cudaFree(d_token_slots);
}
