#include <cuda_runtime.h>

__global__ void leaky_relu_vec4_kernel(const float* __restrict__ input, float* __restrict__ output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 4;

    if (vec_idx + 3 < N) {
        float4 in_val = reinterpret_cast<const float4*>(input)[idx];
        float4 out_val;
        out_val.x = (in_val.x > 0.0f) ? in_val.x : 0.01f * in_val.x;
        out_val.y = (in_val.y > 0.0f) ? in_val.y : 0.01f * in_val.y;
        out_val.z = (in_val.z > 0.0f) ? in_val.z : 0.01f * in_val.z;
        out_val.w = (in_val.w > 0.0f) ? in_val.w : 0.01f * in_val.w;
        reinterpret_cast<float4*>(output)[idx] = out_val;
    } else {
        for (int i = vec_idx; i < N; ++i) {
            float x = input[i];
            output[i] = (x > 0.0f) ? x : 0.01f * x;
        }
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int numVec = (N + 3) / 4;
    int blocksPerGrid = (numVec + threadsPerBlock - 1) / threadsPerBlock;

    leaky_relu_vec4_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
