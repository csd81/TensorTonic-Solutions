#include <cuda_runtime.h>

__global__ void leaky_relu_float4_stride_kernel(const float4* input, float4* output, float alpha, int n_vec) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop processing 4 elements per iteration
    for (int i = idx; i < n_vec; i += stride) {
        float4 in_val = input[i];
        float4 out_val;
        
        out_val.x = in_val.x > 0.0f ? in_val.x : alpha * in_val.x;
        out_val.y = in_val.y > 0.0f ? in_val.y : alpha * in_val.y;
        out_val.z = in_val.z > 0.0f ? in_val.z : alpha * in_val.z;
        out_val.w = in_val.w > 0.0f ? in_val.w : alpha * in_val.w;
        
        output[i] = out_val;
    }
}

__global__ void leaky_relu_tail_kernel(const float* input, float* output, float alpha, int tail_start, int N) {
    // Only launched for max 3 elements, so threadIdx.x is sufficient
    int idx = tail_start + threadIdx.x; 
    
    if (idx < N) {
        float x = input[idx];
        output[idx] = x > 0.0f ? x : alpha * x;
    }
}

extern "C" void solve(const float* input, float* output, float alpha, int N) {
    int threads = 256;
    int n_vec = N / 4;
    
    if (n_vec > 0) {
        // Calculate blocks, but cap at a fixed number to utilize the grid-stride.
        // A common heuristic is 32 * number_of_SMs on the GPU.
        int blocks = (n_vec + threads - 1) / threads;
        if (blocks > 1024) {
            blocks = 1024; 
        }

        leaky_relu_float4_stride_kernel<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(input), 
            reinterpret_cast<float4*>(output), 
            alpha, 
            n_vec
        );
    }
    
    int tail_start = n_vec * 4;
    int tail_elements = N - tail_start;
    
    if (tail_elements > 0) {
        leaky_relu_tail_kernel<<<1, tail_elements>>>(input, output, alpha, tail_start, N);
    }
    
    cudaDeviceSynchronize();
}