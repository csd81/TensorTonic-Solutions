#include <cuda_runtime.h>

__global__ void dropout_kernel_optimized(const float* input, const float* mask, float* output, float p, int N) {
    // 1. Compute scale once per thread
    float scale = 1.0f / (1.0f - p);
    
    // 2. Set up the grid-stride variables
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // 3. Vectorized processing (float4)
    int n_vec = N / 4; 
    
    // Cast pointers to float4 to load/store 16 bytes at a time
    const float4* in_vec = reinterpret_cast<const float4*>(input);
    const float4* mask_vec = reinterpret_cast<const float4*>(mask);
    float4* out_vec = reinterpret_cast<float4*>(output);
    
    // Grid-stride loop processing 4 elements per iteration
    for (int i = idx; i < n_vec; i += stride) {
        float4 in_val = in_vec[i];
        float4 mask_val = mask_vec[i];
        float4 out_val;
        
        out_val.x = in_val.x * mask_val.x * scale;
        out_val.y = in_val.y * mask_val.y * scale;
        out_val.z = in_val.z * mask_val.z * scale;
        out_val.w = in_val.w * mask_val.w * scale;
        
        out_vec[i] = out_val;
    }
    
    // 4. Scalar tail processing for remaining elements
    // (If N is not a multiple of 4, up to 3 elements remain)
    int tail_start = n_vec * 4;
    int remainder = N % 4;
    
    // Let the first few threads handle the remaining 1-3 elements
    if (remainder > 0 && idx < remainder) {
        int tail_idx = tail_start + idx;
        output[tail_idx] = input[tail_idx] * mask[tail_idx] * scale;
    }
}

extern "C" void solve(const float* input, const float* mask, float* output, float p, int N) {
    int threads = 256;
    
    // Calculate blocks based on the vectorized size (N / 4)
    // We add a fallback to 1 block just in case N is very small (e.g., N < 4)
    // so the tail processing still has threads to run on.
    int n_vec = N / 4;
    int blocks = (n_vec + threads - 1) / threads;
    if (blocks == 0) blocks = 1; 

    // Call the newly named optimized kernel
    dropout_kernel_optimized<<<blocks, threads>>>(input, mask, output, p, N);
    cudaDeviceSynchronize();
}