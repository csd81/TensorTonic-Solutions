#include <cuda_runtime.h>

__global__ void relu_kernel_vec4(const float* __restrict__ input, 
                                float* __restrict__ output, 
                                int N) {
    int num_vec = N / 4;
    const float4* in_vec = reinterpret_cast<const float4*>(input);
    float4* out_vec = reinterpret_cast<float4*>(output);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Vectorized 128-bit path
    if (idx < num_vec) {
        float4 v = in_vec[idx];
        v.x = fmaxf(v.x, 0.0f);
        v.y = fmaxf(v.y, 0.0f);
        v.z = fmaxf(v.z, 0.0f);
        v.w = fmaxf(v.w, 0.0f);
        out_vec[idx] = v;
    }

    // Scalar cleanup for remaining elements
    int tail_idx = num_vec * 4 + idx;
    if (tail_idx < N) {
        output[tail_idx] = fmaxf(input[tail_idx], 0.0f);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    // Base block calculation on the number of float4 elements (ceil(N / 4))
    int num_vec = (N + 3) / 4;
    int blocks = (num_vec + threads - 1) / threads;
    if (blocks == 0) blocks = 1;

    relu_kernel_vec4<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}