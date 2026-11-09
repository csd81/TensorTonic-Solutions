#include <cuda_runtime.h>

// Custom matrix multiplication kernel for Row-Major outputs: C = A @ B^T
__global__ void matmul_transpose_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[col * K + i]; // B is transposed, so B^T[i, col] is B[col, i]
        }
        C[row * N + col] = sum;
    }
}

// Accumulating matrix multiplication kernel: C = C + scale * (A @ B^T)
__global__ void matmul_transpose_accumulate_kernel(const float* A, const float* B, float* C, int M, int N, int K, float scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[col * K + i];
        }
        C[row * N + col] += scale * sum;
    }
}

extern "C" void solve(const float* x, const float* W, const float* A, const float* B, float* output,
                      int batch, int d_in, int d_out, int rank, float lora_scale) {
    
    // Allocate intermediate buffer for (X @ A^T) -> shape: (batch, rank)
    float* temp;
    cudaMalloc((void**)&temp, batch * rank * sizeof(float));

    dim3 blockSize(16, 16);

    // 1. Compute output = X @ W^T
    dim3 gridW((d_out + blockSize.x - 1) / blockSize.x, (batch + blockSize.y - 1) / blockSize.y);
    matmul_transpose_kernel<<<gridW, blockSize>>>(x, W, output, batch, d_out, d_in);

    // 2. Compute temp = X @ A^T
    dim3 gridA((rank + blockSize.x - 1) / blockSize.x, (batch + blockSize.y - 1) / blockSize.y);
    matmul_transpose_kernel<<<gridA, blockSize>>>(x, A, temp, batch, rank, d_in);

    // 3. Compute output += lora_scale * (temp @ B^T)
    matmul_transpose_accumulate_kernel<<<gridW, blockSize>>>(temp, B, output, batch, d_out, rank, lora_scale);

    // Synchronize and free memory
    cudaDeviceSynchronize();
    cudaFree(temp);
}
