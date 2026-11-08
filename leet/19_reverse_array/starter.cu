#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Only threads in the first half need to perform swaps
    if (idx < N / 2) {
        int mirror_idx = N - 1 - idx;
        
        float temp = input[idx];
        input[idx] = input[mirror_idx];
        input[mirror_idx] = temp;
    }
}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int halfN = N / 2;
    if (halfN == 0) return; // Nothing to reverse if N <= 1

    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}
