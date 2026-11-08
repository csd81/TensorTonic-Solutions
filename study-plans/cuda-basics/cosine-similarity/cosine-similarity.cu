#include <cuda_runtime.h>
#include <math.h>

__global__ void cosine_partials_kernel(const float* A, const float* B, float* scratch, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float dot = 0.0f;
    float normA = 0.0f;
    float normB = 0.0f;

    // Grid-stride loop: Load pairs once from global memory and derive all three components 
    // immediately using registers.
    for (int i = idx; i < N; i += stride) {
        float a = A[i];
        float b = B[i];
        dot += a * b;
        normA += a * a;
        normB += b * b;
    }

    // Provision large enough arrays for block sizes up to 1024 threads
    __shared__ float s_dot[1024];
    __shared__ float s_normA[1024];
    __shared__ float s_normB[1024];

    s_dot[tid] = dot;
    s_normA[tid] = normA;
    s_normB[tid] = normB;
    __syncthreads();

    // In-block tree reduction (leaves a short tail below one warp)
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            s_dot[tid] += s_dot[tid + s];
            s_normA[tid] += s_normA[tid + s];
            s_normB[tid] += s_normB[tid + s];
        }
        __syncthreads();
    }

    // The short tail is handled rapidly by a warp-shuffle
    if (tid < 32) {
        dot = s_dot[tid];
        normA = s_normA[tid];
        normB = s_normB[tid];
        
        if (blockDim.x > 32) {
            dot += s_dot[tid + 32];
            normA += s_normA[tid + 32];
            normB += s_normB[tid + 32];
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            dot += __shfl_down_sync(0xffffffff, dot, offset);
            normA += __shfl_down_sync(0xffffffff, normA, offset);
            normB += __shfl_down_sync(0xffffffff, normB, offset);
        }

        // Leader thread of the block atomically adds to the 3-element global scratchpad
        if (tid == 0) {
            atomicAdd(&scratch[0], dot);
            atomicAdd(&scratch[1], normA);
            atomicAdd(&scratch[2], normB);
        }
    }
}

__global__ void cosine_finalize_kernel(const float* scratch, float* result) {
    if (threadIdx.x == 0) {
        float dot = scratch[0];
        float normA = scratch[1];
        float normB = scratch[2];
        
        // rsqrtf performs a fast reciprocal square root on the device
        // Mathematically equivalent to dot / sqrtf(normA * normB) but prevents 
        // overflow that could occur if multiplying two extremely large norms
        result[0] = dot * rsqrtf(normA) * rsqrtf(normB);
    }
}

extern "C" void solve(const float* A, const float* B, float* result, int N) {
    float* scratch;
    cudaMalloc(&scratch, 3 * sizeof(float));
    cudaMemset(scratch, 0, 3 * sizeof(float));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;

    cosine_partials_kernel<<<blocks, threads>>>(A, B, scratch, N);
    cosine_finalize_kernel<<<1, 1>>>(scratch, result);

    cudaDeviceSynchronize();
    cudaFree(scratch);
}