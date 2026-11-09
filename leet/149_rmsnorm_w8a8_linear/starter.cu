#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdint.h>

// Vectorized load/store helpers for performance
#define UB_BLOCK_SIZE 256

// -------------------------------------------------------------------------
// Kernel 1: Fused RMSNorm + Dynamic Per-Token Quantization (FP32 -> INT8)
// -------------------------------------------------------------------------
__global__ void rmsnorm_quant_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    int8_t* __restrict__ q_x,
    float* __restrict__ s_x,
    int M, int K) 
{
    int row = blockIdx.x;
    if (row >= M) return;

    int tid = threadIdx.x;
    
    // Step 1: Compute sum of squares (Warp-shuffle reduction)
    float local_sq_sum = 0.0f;
    for (int k = tid; k < K; k += blockDim.x) {
        float val = x[row * K + k];
        local_sq_sum += val * val;
    }

    // Block reduction for sum of squares
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sq_sum += __shfl_xor_sync(0xffffffff, local_sq_sum, offset);
    }
    
    __shared__ float shared_sq_sum;
    if (tid % 32 == 0) {
        if (tid == 0) shared_sq_sum = 0.0f;
    }
    __syncthreads();

    if (tid % 32 == 0) {
        atomicAdd(&shared_sq_sum, local_sq_sum);
    }
    __syncthreads();

    float rms = sqrtf(shared_sq_sum / K + 1e-6f);
    float inv_rms = 1.0f / rms;

    // Step 2: Find max absolute value of y_ik for scaling factor s_i
    float local_max_abs = 0.0f;
    for (int k = tid; k < K; k += blockDim.x) {
        float y_ik = (x[row * K + k] * inv_rms) * gamma[k];
        float abs_val = fabsf(y_ik);
        if (abs_val > local_max_abs) {
            local_max_abs = abs_val;
        }
    }

    // Block reduction for max absolute value
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_xor_sync(0xffffffff, local_max_abs, offset);
        if (other > local_max_abs) local_max_abs = other;
    }

    __shared__ float shared_max_abs;
    if (tid == 0) shared_max_abs = 0.0f;
    __syncthreads();

    if (tid % 32 == 0) {
        // Atomic max for floats using integer casting
        int* shared_max_abs_int = (int*)&shared_max_abs;
        int old = *shared_max_abs_int, assumed;
        do {
            assumed = old;
            float max_val = fmaxf(*(float*)&assumed, local_max_abs);
            old = atomicCAS(shared_max_abs_int, assumed, *(int*)&max_val);
        } while (assumed != old);
    }
    __syncthreads();

    float s_i = shared_max_abs / 127.0f;
    if (tid == 0) {
        s_x[row] = s_i;
    }

    // Step 3: Quantize and clip to [-127, 127]
    float inv_s_i = (s_i > 0.0f) ? (1.0f / s_i) : 0.0f;

    for (int k = tid; k < K; k += blockDim.x) {
        if (s_i == 0.0f) {
            q_x[row * K + k] = 0;
        } else {
            float y_ik = (x[row * K + k] * inv_rms) * gamma[k];
            // rintf implements round-to-nearest (ties to even)
            float q_val = rintf(y_ik * inv_s_i);
            // Clamp strictly within [-127, 127]
            if (q_val > 127.0f) q_val = 127.0f;
            if (q_val < -127.0f) q_val = -127.0f;
            q_x[row * K + k] = (int8_t)q_val;
        }
    }
}

// -------------------------------------------------------------------------
// Kernel 2: INT8 GEMM using dp4a hardware intrinsics + Dequantization
// -------------------------------------------------------------------------
__global__ void w8a8_gemm_kernel(
    const int8_t* __restrict__ q_x,
    const int8_t* __restrict__ W,
    const float* __restrict__ s_x,
    const float* __restrict__ w_scale,
    float* __restrict__ output,
    int M, int N, int K) 
{
    // Block tracking for M and N dimensions
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    int32_t accum = 0;

    // Vectorized loops processing 4 elements of int8 concurrently (packed into int32)
    int k_vec = K / 4;
    const int32_t* q_x_vec = (const int32_t*)(q_x + row * K);
    
    // We iterate over the columns of W, but since W is shape (N, K),
    // each row j holds K weights of output channel j.
    const int32_t* W_vec = (const int32_t*)(W + col * K);

    for (int k = 0; k < k_vec; ++k) {
        int32_t x_4 = q_x_vec[k];
        int32_t w_4 = W_vec[k];
        // Exact hardware integer accumulation
        accum = __dp4a(x_4, w_4, accum);
    }

    // Cleanup residual elements if K is not divisible by 4
    for (int k = k_vec * 4; k < K; ++k) {
        accum += (int32_t)q_x[row * K + k] * (int32_t)W[col * K + k];
    }

    // Step 4: Dequantization back to float32
    output[row * N + col] = (float)accum * s_x[row] * w_scale[col];
}

// -------------------------------------------------------------------------
// Main API Entrypoint
// -------------------------------------------------------------------------
extern "C" void solve(
    const float* x, 
    const float* gamma, 
    const int8_t* W, 
    const float* w_scale,
    float* output, 
    int M, int N, int K) 
{
    // Allocate temporary global memory buffers for the intermediate activations
    int8_t* d_q_x;
    float* d_s_x;
    cudaMalloc((void**)&d_q_x, M * K * sizeof(int8_t));
    cudaMalloc((void**)&d_s_x, M * sizeof(float));

    // Phase 1: Launch RMSNorm and Quantization Layer
    int rmsnorm_threads = UB_BLOCK_SIZE;
    rmsnorm_quant_kernel<<<M, rmsnorm_threads>>>(x, gamma, d_q_x, d_s_x, M, K);

    // Phase 2: Launch high-performance exact integer execution GEMM kernel
    dim3 gemm_threads(16, 16);
    dim3 gemm_blocks((N + gemm_threads.x - 1) / gemm_threads.x, 
                     (M + gemm_threads.y - 1) / gemm_threads.y);
    w8a8_gemm_kernel<<<gemm_blocks, gemm_threads>>>(d_q_x, W, d_s_x, w_scale, output, M, N, K);

    // Synchronize to prevent memory leaks or early deallocation race conditions
    cudaDeviceSynchronize();

    cudaFree(d_q_x);
    cudaFree(d_s_x);
}
