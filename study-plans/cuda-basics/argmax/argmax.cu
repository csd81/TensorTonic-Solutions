#include <cuda_runtime.h>
#include <float.h>

// Helper inline function to perform the (value, index) reduction 
// directly inside the registers of a single warp.
__device__ void warp_reduce_argmax(float& val, int& idx) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float val2 = __shfl_down_sync(0xffffffff, val, offset);
        int idx2 = __shfl_down_sync(0xffffffff, idx, offset);
        if (val2 > val || (val2 == val && idx2 < idx)) {
            val = val2;
            idx = idx2;
        }
    }
}

__global__ void argmax_kernel(const float* input, float* block_vals, int* block_idxs, int N) {
    // Shared memory allocated for up to 32 warps (1024 threads maximum)
    __shared__ float s_vals[32];
    __shared__ int s_idxs[32];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Out of bounds threads default to -inf and INT_MAX (guaranteed to lose)
    float val = (idx < N) ? input[idx] : __int_as_float(0xff800000);
    int cur_idx = (idx < N) ? idx : 2147483647;

    // Step 1: Reduce within the warp
    warp_reduce_argmax(val, cur_idx);

    int lane = tid % 32;
    int warp_id = tid / 32;

    // Step 2: Write warp winners to shared memory
    if (lane == 0) {
        s_vals[warp_id] = val;
        s_idxs[warp_id] = cur_idx;
    }
    __syncthreads();

    // Step 3: Warp 0 sweeps across all preceding warp winners
    if (warp_id == 0) {
        int num_warps = blockDim.x / 32;
        val = (lane < num_warps) ? s_vals[lane] : __int_as_float(0xff800000);
        cur_idx = (lane < num_warps) ? s_idxs[lane] : 2147483647;
        
        warp_reduce_argmax(val, cur_idx);
        
        // Output the final local (value, index) winner for this block
        if (lane == 0) {
            block_vals[blockIdx.x] = val;
            block_idxs[blockIdx.x] = cur_idx;
        }
    }
}

__global__ void argmax_finalize_kernel(const float* block_vals, const int* block_idxs, int* result, int num_blocks) {
    __shared__ float s_vals[32];
    __shared__ int s_idxs[32];
    
    int tid = threadIdx.x;
    
    float my_val = __int_as_float(0xff800000); 
    int my_idx = 2147483647;

    // Grid-stride loop allowing a single block to reduce thousands of partials safely 
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        float val = block_vals[i];
        int idx = block_idxs[i];
        if (val > my_val || (val == my_val && idx < my_idx)) {
            my_val = val;
            my_idx = idx;
        }
    }

    // Step 1: Reduce within the warp
    warp_reduce_argmax(my_val, my_idx);

    int lane = tid % 32;
    int warp_id = tid / 32;

    // Step 2: Write warp winners to shared memory
    if (lane == 0) {
        s_vals[warp_id] = my_val;
        s_idxs[warp_id] = my_idx;
    }
    __syncthreads();

    // Step 3: Final sweep by Warp 0
    if (warp_id == 0) {
        int num_warps = blockDim.x / 32;
        my_val = (lane < num_warps) ? s_vals[lane] : __int_as_float(0xff800000);
        my_idx = (lane < num_warps) ? s_idxs[lane] : 2147483647;
        
        warp_reduce_argmax(my_val, my_idx);
        
        // Deposit the absolute winner into result buffer
        if (lane == 0) {
            result[0] = my_idx;
        }
    }
}

extern "C" void solve(const float* input, int* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float* block_vals = nullptr;
    int* block_idxs = nullptr;
    
    // Allocate scratch buffers mapped to grid blocks size footprint
    cudaMalloc(&block_vals, blocks * sizeof(float));
    cudaMalloc(&block_idxs, blocks * sizeof(int));

    // Launch configuration sequence
    argmax_kernel<<<blocks, threads>>>(input, block_vals, block_idxs, N);
    argmax_finalize_kernel<<<1, threads>>>(block_vals, block_idxs, result, blocks);

    // Free resources allocated mid-pipeline 
    cudaFree(block_vals);
    cudaFree(block_idxs);
}