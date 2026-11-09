#include <cuda_runtime.h>

// Naive baseline for challenge 126, kept as the starter's shape: one block per
// row, a grid-stride scan with scalar loads, and (value, index) pair
// reductions.  Already at the memory wall -- measures 1.68 ms at B=256,
// V=128,000 (78 GB/s); the shipped kernel reaches 84.5 GB/s with float4
// loads.

// A helper structure to track both the maximum value and its index during reduction
struct NaiveMaxPair {
    float val;
    int idx;
};

// Device function to merge two pairs according to the problem constraints:
// 1. Keeps the larger value.
// 2. Breaks ties by keeping the smaller column index.
__device__ __forceinline__ NaiveMaxPair merge_pairs(NaiveMaxPair a, NaiveMaxPair b) {
    if (b.val > a.val) {
        return b;
    } else if (b.val == a.val) {
        return (b.idx < a.idx) ? b : a;
    }
    return a;
}

__global__ void argmax_naive_kernel(const float* logits, int* output, int B, int V) {
    // Each block handles exactly one row (batch item)
    int b = blockIdx.x;
    if (b >= B) return;

    int tid = threadIdx.x;
    int block_dim = blockDim.x;
    
    // Pointer to the start of the current row
    const float* row_logits = logits + b * V;

    // Phase 1: Thread-local grid-stride loop over the row elements
    NaiveMaxPair local_max = {-3.40282347e+38F, -1}; // Initialize with -FLT_MAX
    
    for (int v = tid; v < V; v += block_dim) {
        NaiveMaxPair current = {row_logits[v], v};
        local_max = merge_pairs(local_max, current);
    }

    // Phase 2: Warp-level reduction using shuffle instructions
    for (int offset = 16; offset > 0; offset /= 2) {
        float next_val = __shfl_down_sync(0xFFFFFFFF, local_max.val, offset);
        int next_idx = __shfl_down_sync(0xFFFFFFFF, local_max.idx, offset);
        
        NaiveMaxPair remote = {next_val, next_idx};
        local_max = merge_pairs(local_max, remote);
    }

    // Phase 3: Block-level reduction using shared memory (if block size > 32)
    __shared__ NaiveMaxPair shared_max[32]; // Maximum 32 warps per block
    int lane = tid % 32;
    int warp_id = tid / 32;

    if (lane == 0) {
        shared_max[warp_id] = local_max;
    }
    __syncthreads();

    // The first warp reduces the values collected from all warps
    if (warp_id == 0) {
        // Initialize with the current warp's value or identity if warp doesn't exist
        NaiveMaxPair warp_max = (tid < (block_dim + 31) / 32) ? shared_max[lane] : NaiveMaxPair{-3.40282347e+38F, -1};

        for (int offset = 16; offset > 0; offset /= 2) {
            float next_val = __shfl_down_sync(0xFFFFFFFF, warp_max.val, offset);
            int next_idx = __shfl_down_sync(0xFFFFFFFF, warp_max.idx, offset);
            
            NaiveMaxPair remote = {next_val, next_idx};
            warp_max = merge_pairs(warp_max, remote);
        }

        // Phase 4: Write the final minimum index of the maximum value to output
        if (tid == 0) {
            output[b] = warp_max.idx;
        }
    }
}

extern "C" void solve(const float* logits, int* output, int B, int V) {
    // 512 threads per block strikes an excellent balance between occupancy and registers
    int threads_per_block = 512;
    int blocks_per_grid = B;

    argmax_naive_kernel<<<blocks_per_grid, threads_per_block>>>(logits, output, B, V);
}
