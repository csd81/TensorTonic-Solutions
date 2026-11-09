#include <cuda_runtime.h>
#include <math.h>

// Element-wise feature map phi(x) = ELU(x) + 1
__device__ __forceinline__ float phi(float x) {
    return (x > 0.0f) ? (x + 1.0f) : expf(x);
}

// Phase 1: Compute the d x d numerator matrix and the d-dimensional sum vector for the denominator
__global__ void compute_intermediates_kernel(const float* __restrict__ K, const float* __restrict__ V, 
                                             float* __restrict__ KV_sum, float* __restrict__ K_sum, 
                                             int M, int d) {
    // Each thread handles one element of the d x d intermediate matrix or the d-dimensional column vector
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 1. Compute the entry (row, col) for the d x d matrix KV_sum = phi(K)^T * V
    if (row < d && col < d) {
        float sum_val = 0.0f;
        for (int j = 0; j < M; ++j) {
            sum_val += phi(K[j * d + row]) * V[j * d + col];
        }
        KV_sum[row * d + col] = sum_val;
    }

    // 2. Compute the col-th element for the d-dimensional vector K_sum = sum_j phi(K_j)
    // Run this exclusively using the first row of threads to prevent duplicated computation
    if (row == 0 && col < d) {
        float sum_k = 0.0f;
        for (int j = 0; j < M; ++j) {
            sum_k += phi(K[j * d + col]);
        }
        K_sum[col] = sum_k;
    }
}

// Phase 2: Compute final output using Q, KV_sum, and K_sum
__global__ void compute_output_kernel(const float* __restrict__ Q, const float* __restrict__ KV_sum, 
                                      const float* __restrict__ K_sum, float* __restrict__ output, 
                                      int M, int d) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index (up to M)
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index (up to d)

    if (i < M && j < d) {
        // Read the current row of Q into registers to optimize the inner loop calculations
        // Compute Denominator = phi(Q_i) * K_sum
        float denominator = 0.0f;
        for (int k = 0; k < d; ++k) {
            denominator += phi(Q[i * d + k]) * K_sum[k];
        }

        // Compute Numerator = phi(Q_i) * KV_sum[:, j]
        float numerator = 0.0f;
        for (int k = 0; k < d; ++k) {
            numerator += phi(Q[i * d + k]) * KV_sum[k * d + j];
        }

        // Avoid division by zero and write the result
        output[i * d + j] = (denominator != 0.0f) ? (numerator / denominator) : 0.0f;
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d) {
    // Allocate device workspace pointers for intermediate representations
    float* d_KV_sum;
    float* d_K_sum;
    cudaMalloc((void**)&d_KV_sum, d * d * sizeof(float));
    cudaMalloc((void**)&d_K_sum, d * sizeof(float));

    // Launch configuration for Phase 1 (Intermediates Kernel)
    dim3 threadsPerBlock1(16, 16);
    dim3 numBlocks1((d + threadsPerBlock1.x - 1) / threadsPerBlock1.x, 
                    (d + threadsPerBlock1.y - 1) / threadsPerBlock1.y);
    
    compute_intermediates_kernel<<<numBlocks1, threadsPerBlock1>>>(K, V, d_KV_sum, d_K_sum, M, d);

    // Launch configuration for Phase 2 (Output Generation Kernel)
    dim3 threadsPerBlock2(16, 16);
    dim3 numBlocks2((d + threadsPerBlock2.x - 1) / threadsPerBlock2.x, 
                    (M + threadsPerBlock2.y - 1) / threadsPerBlock2.y);

    compute_output_kernel<<<numBlocks2, threadsPerBlock2>>>(Q, d_KV_sum, d_K_sum, output, M, d);

    // Synchronize and clean up allocated temporary allocations
    cudaDeviceSynchronize();
    cudaFree(d_KV_sum);
    cudaFree(d_K_sum);
}
