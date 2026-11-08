#include <cuda_runtime.h>
#include <math.h>

#define MAX_ITEMS_PER_THREAD 16
#define MAX_FLOAT4_PER_THREAD 4

// ==========================================================
// 1. SCALAR KERNEL (Fallback for unaligned N, e.g., N=23)
// ==========================================================
__global__ void rms_norm_scalar_kernel(const float* input, const float* gamma, float* output, int M, int N, float eps) {
    int row = blockIdx.x;
    if (row >= M) return;

    int tid = threadIdx.x;
    int row_offset = row * N;

    float local_x[MAX_ITEMS_PER_THREAD];
    float local_sq_sum = 0.0f;
    int iters = (N + blockDim.x - 1) / blockDim.x;

    for (int i = 0; i < iters; i++) {
        int j = tid + i * blockDim.x;
        if (j < N) {
            float val = input[row_offset + j];
            local_x[i] = val;
            local_sq_sum += val * val;
        }
    }

    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sq_sum += __shfl_down_sync(mask, local_sq_sum, offset);
    }

    __shared__ float shared_sq_sum[32];
    int lane = tid % 32;
    int warp_id = tid / 32;

    if (lane == 0) shared_sq_sum[warp_id] = local_sq_sum;
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        local_sq_sum = (lane < num_warps) ? shared_sq_sum[lane] : 0.0f;

        for (int offset = 16; offset > 0; offset /= 2) {
            local_sq_sum += __shfl_down_sync(mask, local_sq_sum, offset);
        }

        if (lane == 0) {
            shared_sq_sum[0] = rsqrtf((local_sq_sum / (float)N) + eps);
        }
    }
    __syncthreads();

    float inv_rms = shared_sq_sum[0];
    for (int i = 0; i < iters; i++) {
        int j = tid + i * blockDim.x;
        if (j < N) {
            output[row_offset + j] = local_x[i] * inv_rms * gamma[j];
        }
    }
}

// ==========================================================
// 2. FLOAT4 KERNEL (Optimized for N % 4 == 0)
// ==========================================================
__global__ void rms_norm_float4_kernel(const float* input, const float* gamma, float* output, int M, int N, float eps) {
    int row = blockIdx.x;
    if (row >= M) return;

    int tid = threadIdx.x;
    int row_offset = row * N;
    
    const float4* input_f4 = reinterpret_cast<const float4*>(input + row_offset);
    const float4* gamma_f4 = reinterpret_cast<const float4*>(gamma);
    float4* output_f4      = reinterpret_cast<float4*>(output + row_offset);
    
    int N4 = N / 4; 
    float4 local_x[MAX_FLOAT4_PER_THREAD];
    float local_sq_sum = 0.0f;
    int iters = (N4 + blockDim.x - 1) / blockDim.x;
    
    for (int i = 0; i < iters; i++) {
        int j = tid + i * blockDim.x;
        if (j < N4) {
            float4 vec = input_f4[j];
            local_x[i] = vec;
            local_sq_sum += vec.x * vec.x + vec.y * vec.y + 
                            vec.z * vec.z + vec.w * vec.w;
        }
    }
    
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sq_sum += __shfl_down_sync(mask, local_sq_sum, offset);
    }

    __shared__ float shared_sq_sum[32];
    int lane = tid % 32;
    int warp_id = tid / 32;

    if (lane == 0) shared_sq_sum[warp_id] = local_sq_sum;
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        local_sq_sum = (lane < num_warps) ? shared_sq_sum[lane] : 0.0f;

        for (int offset = 16; offset > 0; offset /= 2) {
            local_sq_sum += __shfl_down_sync(mask, local_sq_sum, offset);
        }

        if (lane == 0) {
            shared_sq_sum[0] = rsqrtf((local_sq_sum / (float)N) + eps);
        }
    }
    __syncthreads();

    float inv_rms = shared_sq_sum[0];
    for (int i = 0; i < iters; i++) {
        int j = tid + i * blockDim.x;
        if (j < N4) {
            float4 vec = local_x[i];
            float4 g = gamma_f4[j];
            
            float4 out;
            out.x = vec.x * inv_rms * g.x;
            out.y = vec.y * inv_rms * g.y;
            out.z = vec.z * inv_rms * g.z;
            out.w = vec.w * inv_rms * g.w;
            
            output_f4[j] = out;
        }
    }
}

// ==========================================================
// 3. HOST DISPATCHER
// ==========================================================
extern "C" void solve(const float* input, const float* gamma, float* output, int M, int N, float eps) {
    int threads = 256;
    dim3 blocks(M);
    
    // Dynamically route to the float4 kernel ONLY if memory alignment is guaranteed
    if (N % 4 == 0) {
        rms_norm_float4_kernel<<<blocks, threads>>>(input, gamma, output, M, N, eps);
    } else {
        rms_norm_scalar_kernel<<<blocks, threads>>>(input, gamma, output, M, N, eps);
    }
    
    cudaDeviceSynchronize();
}