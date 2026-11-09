#include <cuda_runtime.h>
#include <math.h>

// Device function to compute softplus(-z) = log(1 + exp(-z))
// For numerical stability, log(1 + exp(-z)) can be rewritten when -z is large
__device__ inline float stable_softplus_neg_z(float z) {
    if (z > 0.0f) {
        return logf(1.0f + expf(-z));
    } else {
        return -z + logf(1.0f + expf(z));
    }
}

// Block-level reduction helper using warp shuffles
__device__ inline float blockReduceSum(float val) {
    static __shared__ float shared[32]; // Shared memory for warp results
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    // Fully reduce within each warp
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // Write the local warp sum to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Read from shared memory only if the thread is in the first warp
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;

    if (wid == 0) {
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }

    return val;
}

// Global DPO reduction kernel
__global__ void dpo_loss_kernel(const float* chosen_logps, const float* rejected_logps,
                                const float* chosen_ref_logps, const float* rejected_ref_logps,
                                float* global_scratch, float beta, int B) {
    float local_sum = 0.0f;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop to process elements and calculate local softplus losses
    for (int i = tid; i < B; i += stride) {
        float pi_margin = chosen_logps[i] - rejected_logps[i];
        float ref_margin = chosen_ref_logps[i] - rejected_ref_logps[i];
        float z = beta * (pi_margin - ref_margin);
        
        local_sum += stable_softplus_neg_z(z);
    }

    // Reduce within the block
    float block_sum = blockReduceSum(local_sum);

    // Thread 0 of each block writes its partial sum to the global scratchpad
    if (threadIdx.x == 0) {
        global_scratch[blockIdx.x] = block_sum;
    }
}

// Final aggregation kernel to sum block-level calculations and average by B
__global__ void dpo_average_kernel(const float* global_scratch, float* output, int num_blocks, int B) {
    float total_sum = 0.0f;
    
    // Since num_blocks is small (e.g., 128-512), a single block can easily reduce it
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        total_sum += global_scratch[i];
    }

    float final_sum = blockReduceSum(total_sum);

    if (threadIdx.x == 0) {
        output[0] = final_sum / (float)B;
    }
}

extern "C" void solve(const float* chosen_logps, const float* rejected_logps,
                      const float* chosen_ref_logps, const float* rejected_ref_logps, float* output,
                      float beta, int B) {
    
    // Execution configuration
    int threads_per_block = 256;
    // Set grid size dynamically up to a reasonable cap to maximize wave utilization
    int blocks_per_grid = (B + threads_per_block - 1) / threads_per_block;
    if (blocks_per_grid > 512) blocks_per_grid = 512; 

    // Allocate temporary device memory for block-level reduction results
    float* d_scratch;
    cudaMalloc(&d_scratch, blocks_per_grid * sizeof(float));

    // Step 1: Compute per-element DPO loss and reduce block-wise
    dpo_loss_kernel<<<blocks_per_grid, threads_per_block>>>(
        chosen_logps, rejected_logps, chosen_ref_logps, rejected_ref_logps, d_scratch, beta, B
    );

    // Step 2: Accumulate block results and write the final mean into output[0]
    dpo_average_kernel<<<1, threads_per_block>>>(d_scratch, output, blocks_per_grid, B);

    // Clean up temporary workspace memory
    cudaFree(d_scratch);
}
