#include <cuda_runtime.h>
#include <float.h>

// Helper to keep code clean and strictly enforce the tie-break rule. 
// Passed by reference to update the running minimum inline.
__device__ __forceinline__ void update_min(float &val, int &idx, float other_val, int other_idx) {
    if (other_val < val || (other_val == val && other_idx < idx)) {
        val = other_val;
        idx = other_idx;
    }
}

__global__ void argmin_kernel(const float* input, float* block_vals, int* block_idxs, int N) {
    // 1. Shared memory arrays for the (value, index) pairs
    __shared__ float s_val[256];
    __shared__ int s_idx[256];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // 2. Load from global memory or populate with losing sentinel values (+FLT_MAX)
    if (gid < N) {
        s_val[tid] = input[gid];
        s_idx[tid] = gid;
    } else {
        s_val[tid] = FLT_MAX;
        s_idx[tid] = 2147483647; // INT_MAX guarantees it loses tie-breaks
    }
    
    // Ensure all threads have their values written to shared memory
    __syncthreads();

    // 3. Tree reduction inside shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            update_min(s_val[tid], s_idx[tid], s_val[tid + stride], s_idx[tid + stride]);
        }
        // Sync required after every step so reads don't outpace writes
        __syncthreads();
    }

    // 4. The winner for this block is sitting at slot 0. Dump to scratch arrays.
    if (tid == 0) {
        block_vals[blockIdx.x] = s_val[0];
        block_idxs[blockIdx.x] = s_idx[0];
    }
}

__global__ void argmin_finalize_kernel(const float* block_vals, const int* block_idxs, int* result, int num_blocks) {
    __shared__ float s_val[256];
    __shared__ int s_idx[256];

    int tid = threadIdx.x;

    // 1. Thread-local accumulator initialization
    float my_val = FLT_MAX;
    int my_idx = 2147483647;

    // 2. Grid-stride loop: Handle cases where the intermediate block count > 256
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        update_min(my_val, my_idx, block_vals[i], block_idxs[i]);
    }

    // 3. Dump thread-local minimums to shared memory
    s_val[tid] = my_val;
    s_idx[tid] = my_idx;
    __syncthreads();

    // 4. Final tree reduction inside shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            update_min(s_val[tid], s_idx[tid], s_val[tid + stride], s_idx[tid + stride]);
        }
        __syncthreads();
    }

    // 5. Final index winner collapses to thread 0 -> output directly to result
    if (tid == 0) {
        result[0] = s_idx[0];
    }
}

extern "C" void solve(const float* input, int* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float* block_vals = nullptr;
    int* block_idxs = nullptr;
    
    // Allocate global scratch buffers for intermediate block winners
    cudaMalloc(&block_vals, blocks * sizeof(float));
    cudaMalloc(&block_idxs, blocks * sizeof(int));

    // Launch configuration mapping precisely up to N threads
    argmin_kernel<<<blocks, threads>>>(input, block_vals, block_idxs, N);
    
    // Single block aggregates the 'blocks' amount of local scratch values
    argmin_finalize_kernel<<<1, threads>>>(block_vals, block_idxs, result, blocks);
    cudaDeviceSynchronize();

    // Cleanup scratch arrays
    cudaFree(block_vals);
    cudaFree(block_idxs);
}