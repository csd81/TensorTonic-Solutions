#include <cuda_runtime.h>
#include <math.h>

// Vectorized kernel processing 4 elements per thread
__global__ void sigmoid_kernel_float4(const float4* input, float4* output, int N_vec) {
    int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N_vec; idx += stride) {
        float4 in_val = input[idx];
        float4 out_val;
        
        // Compute 4 elements using the fast-math intrinsic
        out_val.x = 1.0f / (1.0f + __expf(-in_val.x));
        out_val.y = 1.0f / (1.0f + __expf(-in_val.y));
        out_val.z = 1.0f / (1.0f + __expf(-in_val.z));
        out_val.w = 1.0f / (1.0f + __expf(-in_val.w));
        
        output[idx] = out_val;
    }
}

// Fallback kernel for the remaining 1-3 elements if N is not a multiple of 4
__global__ void sigmoid_kernel_tail(const float* input, float* output, int N, int offset) {
    int idx = offset + threadIdx.x;
    if (idx < N) {
        output[idx] = 1.0f / (1.0f + __expf(-input[idx]));
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int N_vec = N / 4; // Number of float4 vectors
    
    if (N_vec > 0) {
        int blocks = (N_vec + threads - 1) / threads;
        // Cap blocks to avoid excessive launch overhead; grid-stride handles the rest
        if (blocks > 512) blocks = 512; 
        
        // Pointers from standard CUDA allocations are natively 256-byte aligned, 
        // making them safe to cast to float4* (which requires 16-byte alignment).
        sigmoid_kernel_float4<<<blocks, threads>>>((const float4*)input, (float4*)output, N_vec);
    }
    
    // Handle unaligned tail elements
    int tail = N % 4;
    if (tail > 0) {
        int offset = N - tail;
        sigmoid_kernel_tail<<<1, tail>>>(input, output, N, offset);
    }
    cudaDeviceSynchronize();
}