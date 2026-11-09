#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>

// Helper for safe CUDA allocation checking if needed externally
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      if (abort) asm("trap;");
   }
}

// Global kernel to reset output gradients to zero
__global__ void zero_init_kernel(float* arr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        arr[idx] = 0.0f;
    }
}

// Combined forward/backward pass for dV and partial dS
// Each thread block processes one row i of Q (from 0 to M-1)
// We compute P_ij, dP_ij, and the row sum (sum_k P_ik dP_ik), then output dV elements atomically
__global__ void attention_bwd_dv_ds_kernel(
    const float* __restrict__ Q, 
    const float* __restrict__ K, 
    const float* __restrict__ V, 
    const float* __restrict__ dO, 
    float* __restrict__ dV,
    float* __restrict__ dS, // Temp buffer of size M x N
    int M, int N, int d) 
{
    int i = blockIdx.x; // Row index in Q (0 to M-1)
    if (i >= M) return;

    // Dynamically or statically allocated shared memory for stable row softmax max calculation
    // Since N can be up to 100,000, we will do a two-pass approach or use a localized buffer.
    // Given d <= 128, a thread block size of 256 or 512 is ideal.
    int tid = threadIdx.x;

    // 1. Compute dot products Q_i * K_j^T for all j, find max for stable softmax
    extern __shared__ float shared_mem[];
    float* s_max = &shared_mem[0]; // size 1
    float* s_sum = &shared_mem[1]; // size 1
    float* s_dp_sum = &shared_mem[2]; // size 1

    // A local block-level reduction scratchpad
    volatile __shared__ float s_reduce[512];

    float r_scale = 1.0f / sqrtf((float)d);

    // First, find max(Q_i * K_j^T) across all j
    float local_max = -1e20f;
    for (int j = tid; j < N; j += blockDim.x) {
        float score = 0.0f;
        for (int k = 0; k < d; ++k) {
            score += Q[i * d + k] * K[j * d + k];
        }
        score *= r_scale;
        s_reduce[j] = score; // Reused temporarily if N is small, but let's compute directly
        if (score > local_max) local_max = score;
    }

    // Reduce max
    s_reduce[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_reduce[tid + stride] > s_reduce[tid]) {
                s_reduce[tid] = s_reduce[tid + stride];
            }
        }
        __syncthreads();
    }
    if (tid == 0) *s_max = s_reduce[0];
    __syncthreads();
    float row_max = *s_max;

    // Compute exp and sum of exp
    float local_sum = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) {
        float score = 0.0f;
        for (int k = 0; k < d; ++k) {
            score += Q[i * d + k] * K[j * d + k];
        }
        score = score * r_scale - row_max;
        local_sum += expf(score);
    }

    s_reduce[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_reduce[tid] += s_reduce[tid + stride];
        __syncthreads();
    }
    if (tid == 0) *s_sum = s_reduce[0];
    __syncthreads();
    float row_sum = *s_sum;

    // Now compute P_ij, dP_ij, update dV, and calculate row sum of (P_ij * dP_ij)
    float local_dp_sum = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) {
        float score = 0.0f;
        for (int k = 0; k < d; ++k) {
            score += Q[i * d + k] * K[j * d + k];
        }
        float P_ij = expf(score * r_scale - row_max) / row_sum;

        // Compute dP_ij = sum_k (dO_ik * V_jk)
        float dP_ij = 0.0f;
        for (int k = 0; k < d; ++k) {
            dP_ij += dO[i * d + k] * V[j * d + k];
        }

        local_dp_sum += P_ij * dP_ij;

        // dV = P^T * dO -> dV_jk += P_ij * dO_ik
        for (int k = 0; k < d; ++k) {
            atomicAdd(&dV[j * d + k], P_ij * dO[i * d + k]);
        }
        
        // Temporarily store P_ij and dP_ij components implicitly or write intermediate to dS
        // We will reconstruct/use dS_ij later. Let's store P_ij and dP_ij to speed up dQ/dK passes
        // dynamic memory constraint: write dS_ij directly into global memory temp buffer
        dS[i * N + j] = P_ij * dP_ij; 
    }

    s_reduce[tid] = local_dp_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_reduce[tid] += s_reduce[tid + stride];
        __syncthreads();
    }
    if (tid == 0) *s_dp_sum = s_reduce[0];
    __syncthreads();
    float row_dp_sum = *s_dp_sum;

    // Second pass over the row to calculate finalized dS_tilde = (P_ij * dP_ij - P_ij * row_dp_sum) / sqrt(d)
    for (int j = tid; j < N; j += blockDim.x) {
        // Recompute P_ij to save shared memory tracking overhead
        float score = 0.0f;
        for (int k = 0; k < d; ++k) {
            score += Q[i * d + k] * K[j * d + k];
        }
        float P_ij = expf(score * r_scale - row_max) / row_sum;
        float p_dp = dS[i * N + j]; // contains P_ij * dP_ij
        
        // dS_ij = P_ij * dP_ij - P_ij * sum_k(P_ik * dP_ik)
        float dS_ij = p_dp - P_ij * row_dp_sum;
        dS[i * N + j] = dS_ij * r_scale; // Store \widetilde{dS}_{ij}
    }
}

// Compute dQ = \widetilde{dS} * K
__global__ void compute_dq_kernel(
    const float* __restrict__ dS_tilde, 
    const float* __restrict__ K, 
    float* __restrict__ dQ, 
    int M, int N, int d) 
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row in Q (0 to M-1)
    int k = blockIdx.x * blockDim.x + threadIdx.x; // Col in Q (0 to d-1)

    if (i < M && k < d) {
        float val = 0.0f;
        for (int j = 0; j < N; ++j) {
            val += dS_tilde[i * N + j] * K[j * d + k];
        }
        dQ[i * d + k] = val;
    }
}

// Compute dK = \widetilde{dS}^T * Q
__global__ void compute_dk_kernel(
    const float* __restrict__ dS_tilde, 
    const float* __restrict__ Q, 
    float* __restrict__ dK, 
    int M, int N, int d) 
{
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Row in K (0 to N-1)
    int k = blockIdx.x * blockDim.x + threadIdx.x; // Col in K (0 to d-1)

    if (j < N && k < d) {
        float val = 0.0f;
        for (int i = 0; i < M; ++i) {
            val += dS_tilde[i * N + j] * Q[i * d + k];
        }
        dK[j * d + k] = val;
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, const float* dO, float* dQ,
                      float* dK, float* dV, int M, int N, int d) 
{
    // 1. Initialize dV to 0 since we accumulate into it using atomic additions
    int zero_threads = 256;
    int zero_blocks_dv = (N * d + zero_threads - 1) / zero_threads;
    zero_init_kernel<<<zero_blocks_dv, zero_threads>>>(dV, N * d);

    // 2. Allocate intermediate storage for \widetilde{dS} (Size M x N)
    float* dS_tilde = nullptr;
    cudaMalloc((void**)&dS_tilde, (size_t)M * N * sizeof(float));

    // 3. Launch the combined backward pass for dV and dS_tilde computation
    int block_size = 256; 
    size_t shared_mem_size = 3 * sizeof(float); // s_max, s_sum, s_dp_sum
    attention_bwd_dv_ds_kernel<<<M, block_size, shared_mem_size>>>(Q, K, V, dO, dV, dS_tilde, M, N, d);
    cudaDeviceSynchronize();

    // 4. Compute dQ = \widetilde{dS} * K
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks_dQ((d + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                      (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    compute_dq_kernel<<<numBlocks_dQ, threadsPerBlock>>>(dS_tilde, K, dQ, M, N, d);

    // 5. Compute dK = \widetilde{dS}^T * Q
    dim3 numBlocks_dK((d + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                      (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    compute_dk_kernel<<<numBlocks_dK, threadsPerBlock>>>(dS_tilde, Q, dK, M, N, d);

    cudaDeviceSynchronize();

    // Free intermediate global memory allocation
    cudaFree(dS_tilde);
}
