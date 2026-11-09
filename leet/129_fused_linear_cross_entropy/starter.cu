#include <cuda_runtime.h>
#include <cmath>

// Helper structure for online log-sum-exp reduction
struct LogSumExpPair {
    float max_val;
    float sum_exp;
};

// Device function to merge two running log-sum-exp statistics
__device__ __forceinline__ LogSumExpPair merge_lse(LogSumExpPair a, LogSumExpPair b) {
    if (a.max_val >= b.max_val) {
        if (b.sum_exp == 0.0f) return a;
        return {a.max_val, a.sum_exp + b.sum_exp * __expf(b.max_val - a.max_val)};
    } else {
        if (a.sum_exp == 0.0f) return b;
        return {b.max_val, b.sum_exp + a.sum_exp * __expf(a.max_val - b.max_val)};
    }
}

__global__ void memory_efficient_cross_entropy_kernel(
    const float* __restrict__ X, 
    const float* __restrict__ W, 
    const int* __restrict__ targets, 
    float* __restrict__ global_loss, 
    int N, int D, int V) 
{
    // Each block handles one row of X (one sample out of N)
    int row = blockIdx.x;
    if (row >= N) return;

    int tid = threadIdx.x;
    int bdim = blockDim.x;

    // Load target token index for this row
    int target_idx = targets[row];

    // Local accumulation pairs
    LogSumExpPair local_lse = {-INFINITY, 0.0f};
    float target_logit = 0.0f;

    // Stream over vocabulary elements
    for (int v = tid; v < V; v += bdim) {
        // Compute dot product of X[row, :] and W[v, :]
        float logit = 0.0f;
        for (int d = 0; d < D; ++d) {
            logit += X[row * D + d] * W[v * D + d];
        }

        // Keep track of the target logit when we hit its index
        if (v == target_idx) {
            target_logit = logit;
        }

        // Online Log-Sum-Exp update
        LogSumExpPair next_pair = {logit, 1.0f};
        local_lse = merge_lse(local_lse, next_pair);
    }

    // Shared memory for block-wide reduction (up to 1024 threads)
    extern __shared__ char shared_mem[];
    LogSumExpPair* s_lse = reinterpret_cast<LogSumExpPair*>(shared_mem);
    float* s_target = reinterpret_cast<float*>(&s_lse[bdim]);

    s_lse[tid] = local_lse;
    s_target[tid] = (tid == (target_idx % bdim) || (target_idx >= V && tid == 0)) ? target_logit : 0.0f; 
    // A more bulletproof way to reduce target logit since only one thread actually saves it:
    // Let's broadcast/collect target_logit carefully via shared memory reduction
    s_target[tid] = 0.0f;
    if (target_idx >= 0 && target_idx < V) {
        // Check if this thread was the one responsible for the target logit
        int owner_tid = target_idx % bdim;
        // Map vocabulary loops: if target_idx was processed by this thread
        if ((target_idx - tid) % bdim == 0) {
            s_target[0] = target_logit; // Since only one thread matches, write directly to a shared slot
        }
    }
    __syncthreads();
    
    // Explicitly grab the target logit on thread 0 after initialization
    if (target_idx >= 0 && target_idx < V) {
        if ((target_idx - tid) % bdim == 0) {
            s_target[0] = target_logit;
        }
    }
    __syncthreads();
    
    // Broadcast the target logit to a single register across block threads if needed, 
    // or let thread 0 handle the final loss write-back.
    float final_target_logit = s_target[0];

    // Tree reduction inside the block for LogSumExp
    for (int stride = bdim / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_lse[tid] = merge_lse(s_lse[tid], s_lse[tid + stride]);
        }
        __syncthreads();
    }

    // Thread 0 calculates final row loss and performs atomic add
    if (tid == 0) {
        LogSumExpPair total_lse = s_lse[0];
        float row_loss = (total_lse.max_val + __logf(total_lse.sum_exp)) - final_target_logit;
        
        // Accumulate mean chunk step safely
        atomicAdd(global_loss, row_loss / (float)N);
    }
}

extern "C" void solve(const float* X, const float* W, const int* targets, float* loss, int N, int D, int V) {
    // Initialize output loss value to 0 on the device
    cudaMemset(loss, 0, sizeof(float));

    // Define execution block configurations
    int threads_per_block = 256;
    if (V < 256) threads_per_block = 32; // Scale down for tiny vocabularies
    
    int blocks_per_grid = N;

    // Calculate required shared memory size: (sizeof(LogSumExpPair) + sizeof(float)) * threads
    size_t shared_mem_size = (sizeof(LogSumExpPair) + sizeof(float)) * threads_per_block;

    // Launch memory-efficient fused kernel
    memory_efficient_cross_entropy_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
        X, W, targets, loss, N, D, V
    );

    // Ensure synchronous completeness if required by caller harness environment
    cudaDeviceSynchronize();
}
