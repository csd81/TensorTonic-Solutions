#include <cuda_runtime.h>
#include <math.h>

__global__ void geglu_kernel(const float* input, float* output, int halfN) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < halfN) {
        float x1 = input[idx];
        float x2 = input[idx + halfN];
        
        // 1 / sqrt(2) = 0.7071067811865475f
        const float INV_SQRT_2 = 0.7071067811865475f;
        float gelu_x2 = 0.5f * x2 * (1.0f + erff(x2 * INV_SQRT_2));
        
        output[idx] = x1 * gelu_x2;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    geglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
