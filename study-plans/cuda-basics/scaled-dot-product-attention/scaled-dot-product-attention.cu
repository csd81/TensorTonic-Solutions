#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

__global__ void scores_kernel(const float* Q, const float* K, float* scores, int N, int D) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // [16][17] padding avoids shared memory bank conflicts
    __shared__ float s_Q[16][17];
    __shared__ float s_K[16][17];
    
    float sum = 0.0f;
    for (int d_start = 0; d_start < D; d_start += 16) {
        // Load Q tile
        if (row < N && d_start + threadIdx.x < D) {
            s_Q[threadIdx.y][threadIdx.x] = Q[row * D + d_start + threadIdx.x];
        } else {
            s_Q[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load K tile (transposed conceptually during the dot product loop)
        int k_row = blockIdx.x * 16 + threadIdx.y;
        int k_col = d_start + threadIdx.x;
        if (k_row < N && k_col < D) {
            s_K[threadIdx.y][threadIdx.x] = K[k_row * D + k_col];
        } else {
            s_K[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Accumulate partial dot products
        for (int step = 0; step < 16; ++step) {
            sum += s_Q[threadIdx.y][step] * s_K[threadIdx.x][step];
        }
        
        __syncthreads();
    }
    
    // Scale by 1/sqrt(D) and write to global memory
    if (row < N && col < N) {
        scores[row * N + col] = sum / sqrtf((float)D);
    }
}

__global__ void softmax_rows_kernel(float* scores, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= N) return;
    
    float* row_ptr = &scores[row * N];
    
    // 1. Thread-local max
    float local_max = -FLT_MAX;
    for (int i = tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, row_ptr[i]);
    }
    
    // Block-wide max reduction
    __shared__ float s_data[256];
    s_data[tid] = local_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] = fmaxf(s_data[tid], s_data[tid + s]);
        }
        __syncthreads();
    }
    float row_max = s_data[0];
    
    // 2. Thread-local sum of exponentials
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float val = expf(row_ptr[i] - row_max);
        local_sum += val;
        row_ptr[i] = val; // Overwrite in-place to avoid recomputing expf
    }
    
    // Block-wide sum reduction
    s_data[tid] = local_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }
    float row_sum = s_data[0];
    
    // 3. Normalize
    for (int i = tid; i < N; i += blockDim.x) {
        row_ptr[i] /= row_sum;
    }
}

__global__ void av_kernel(const float* attn, const float* V, float* output, int N, int D) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float s_P[16][17];
    __shared__ float s_V[16][17];
    
    float sum = 0.0f;
    for (int j_start = 0; j_start < N; j_start += 16) {
        // Load Attention weights tile
        if (row < N && j_start + threadIdx.x < N) {
            s_P[threadIdx.y][threadIdx.x] = attn[row * N + j_start + threadIdx.x];
        } else {
            s_P[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load Values tile
        int v_row = j_start + threadIdx.y;
        int v_col = blockIdx.x * 16 + threadIdx.x;
        if (v_row < N && v_col < D) {
            s_V[threadIdx.y][threadIdx.x] = V[v_row * D + v_col];
        } else {
            s_V[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Accumulate partial sums
        for (int step = 0; step < 16; ++step) {
            sum += s_P[threadIdx.y][step] * s_V[step][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < D) {
        output[row * D + col] = sum;
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int N, int D) {
    float* scores;
    // N x N scratchpad for the intermediate scores
    cudaMalloc(&scores, (size_t)N * N * sizeof(float));

    // Stage 1: Scores (Matmul with scaling)
    dim3 sThreads(16, 16);
    dim3 sBlocks((N + 15) / 16, (N + 15) / 16);
    scores_kernel<<<sBlocks, sThreads>>>(Q, K, scores, N, D);

    // Stage 2: Numerically stable row-parallel Softmax
    // Launch one block per row with 256 threads to perform block-wise parallel reductions
    softmax_rows_kernel<<<N, 256>>>(scores, N);

    // Stage 3: Values aggregation (Matmul)
    dim3 avThreads(16, 16);
    dim3 avBlocks((D + 15) / 16, (N + 15) / 16);
    av_kernel<<<avBlocks, avThreads>>>(scores, V, output, N, D);
    
    cudaFree(scores);
}