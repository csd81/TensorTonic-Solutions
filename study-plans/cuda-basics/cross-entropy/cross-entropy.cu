#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

__global__ void cross_entropy_row_kernel(const float* logits, const int* target, float* partial, int B, int C) {
    // Dynamic shared memory to cache the row and minimize global memory reads
    extern __shared__ float s_logits[];
    
    int b = blockIdx.x;
    if (b >= B) return;
    
    // Check if C fits safely into standard 48KB shared memory constraints (~12k floats max)
    bool use_smem = (C <= 10000);
    const float* row_logits = logits + b * C;
    
    // --- Pass 1: Global Read & Max Reduction ---
    float thread_max = -FLT_MAX;
    
    if (use_smem) {
        for (int c = threadIdx.x; c < C; c += blockDim.x) {
            float val = row_logits[c];
            s_logits[c] = val; // Stage row into shared memory
            thread_max = fmaxf(thread_max, val);
        }
    } else {
        for (int c = threadIdx.x; c < C; c += blockDim.x) {
            thread_max = fmaxf(thread_max, row_logits[c]);
        }
    }
    
    // Block-wide max reduction using register-to-register warp shuffles
    __shared__ float s_warp_max[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    float val_max = thread_max;
    for (int offset = 16; offset > 0; offset /= 2) {
        val_max = fmaxf(val_max, __shfl_down_sync(0xffffffff, val_max, offset));
    }
    
    if (lane == 0) s_warp_max[wid] = val_max;
    __syncthreads();
    
    float row_max = (threadIdx.x < (blockDim.x + 31) / 32) ? s_warp_max[lane] : -FLT_MAX;
    if (wid == 0) {
        for (int offset = 16; offset > 0; offset /= 2) {
            row_max = fmaxf(row_max, __shfl_down_sync(0xffffffff, row_max, offset));
        }
    }
    
    // Broadcast the final row max to all threads
    if (threadIdx.x == 0) s_warp_max[0] = row_max;
    __syncthreads();
    row_max = s_warp_max[0];
    
    // --- Pass 2: Exp Sum Reduction ---
    float thread_sum = 0.0f;
    if (use_smem) {
        for (int c = threadIdx.x; c < C; c += blockDim.x) {
            thread_sum += expf(s_logits[c] - row_max);
        }
    } else {
        for (int c = threadIdx.x; c < C; c += blockDim.x) {
            thread_sum += expf(row_logits[c] - row_max);
        }
    }
    
    // Block-wide sum reduction using warp shuffles
    __shared__ float s_warp_sum[32];
    float val_sum = thread_sum;
    for (int offset = 16; offset > 0; offset /= 2) {
        val_sum += __shfl_down_sync(0xffffffff, val_sum, offset);
    }
    
    if (lane == 0) s_warp_sum[wid] = val_sum;
    __syncthreads();
    
    float row_sum = (threadIdx.x < (blockDim.x + 31) / 32) ? s_warp_sum[lane] : 0.0f;
    if (wid == 0) {
        for (int offset = 16; offset > 0; offset /= 2) {
            row_sum += __shfl_down_sync(0xffffffff, row_sum, offset);
        }
    }
    
    // Broadcast the final row sum to all threads
    if (threadIdx.x == 0) s_warp_sum[0] = row_sum;
    __syncthreads();
    row_sum = s_warp_sum[0];
    
    // --- Target Calculation & Atomic Accumulation ---
    // Single atomicAdd per block keeps contention extremely low
    if (threadIdx.x == 0) {
        int t = target[b];
        float target_val = use_smem ? s_logits[t] : row_logits[t];
        float row_loss = logf(row_sum) - (target_val - row_max);
        atomicAdd(partial, row_loss);
    }
}

__global__ void cross_entropy_finalize_kernel(float* loss, int B) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && B > 0) {
        *loss = *loss / (float)B;
    }
}

extern "C" void solve(const float* logits, const int* target, float* loss, int B, int C) {
    cudaMemset(loss, 0, sizeof(float));

    int threads = 256;
    dim3 blocks(B);
    
    // Dynamically allocate shared memory up to ~40KB boundary. 
    // Secures against cudaErrorInvalidValue launch failures if C is vastly large (e.g. GPT-2 50k vocab).
    size_t smem = (C <= 10000) ? (C * sizeof(float)) : 0;
    
    cross_entropy_row_kernel<<<blocks, threads, smem>>>(logits, target, loss, B, C);
    cross_entropy_finalize_kernel<<<1, 1>>>(loss, B);
    cudaDeviceSynchronize();
}