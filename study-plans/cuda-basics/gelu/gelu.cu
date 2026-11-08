#include <cuda_runtime.h>
#include <math.h>

#define RSQRT2 0.70710678118654752440f

__global__ void gelu_kernel_vec4(const float* input, float* output, int N) {
    // Process 4 elements per thread
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int stride = blockDim.x * gridDim.x * 4;
    
    // Find largest multiple of 4
    int n_vec = N & ~3; 

    // Vectorized grid-stride loop
    for (int i = idx; i < n_vec; i += stride) {
        float4 in4 = reinterpret_cast<const float4*>(&input[i])[0];
        float4 out4;
        
        out4.x = 0.5f * in4.x * (1.0f + erff(in4.x * RSQRT2));
        out4.y = 0.5f * in4.y * (1.0f + erff(in4.y * RSQRT2));
        out4.z = 0.5f * in4.z * (1.0f + erff(in4.z * RSQRT2));
        out4.w = 0.5f * in4.w * (1.0f + erff(in4.w * RSQRT2));
        
        reinterpret_cast<float4*>(&output[i])[0] = out4;
    }

    // Handle tail elements natively without branching inside the main loop
    for (int i = n_vec + blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + erff(x * RSQRT2));
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    // Target ~32-64 blocks per SM to saturate the GPU and let the stride loop handle N
    int num_blocks = 80 * 32; 
    
    gelu_kernel_vec4<<<num_blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}