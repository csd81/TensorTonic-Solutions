#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

extern "C" void solve(const float* x, const float* W, const float* A, const float* B, float* output,
                      int batch, int d_in, int d_out, int rank, float lora_scale) {
    
    // Create a cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate temporary workspace for intermediate tensor: temp = X @ A^T
    // Shape of X: (batch, d_in)
    // Shape of A: (rank, d_in) -> A^T: (d_in, rank)
    // Shape of temp: (batch, rank)
    float* d_temp = nullptr;
    cudaMalloc((void**)&d_temp, batch * rank * sizeof(float));

    float alpha1 = 1.0f;
    float beta0 = 0.0f;

    // 1. Compute temp = X @ A^T
    // cuBLAS uses column-major order. For row-major C = M1 @ M2:
    // We compute C^T = M2^T @ M1^T
    // temp^T (rank x batch) = A (rank x d_in) @ X^T (d_in x batch)
    cublasSgemm(handle, 
                CUBLAS_OP_N, CUBLAS_OP_T, 
                rank, batch, d_in, 
                &alpha1, 
                A, rank, 
                x, d_in, 
                &beta0, 
                d_temp, rank);

    // 2. Compute output = X @ W^T
    // output^T (d_out x batch) = W (d_out x d_in) @ X^T (d_in x batch)
    cublasSgemm(handle, 
                CUBLAS_OP_N, CUBLAS_OP_T, 
                d_out, batch, d_in, 
                &alpha1, 
                W, d_out, 
                x, d_in, 
                &beta0, 
                output, d_out);

    // 3. Compute output = output + lora_scale * (temp @ B^T)
    // output^T (d_out x batch) = B (d_out x rank) @ temp^T (rank x batch) + lora_scale * output^T
    // Note: cuBLAS performs C = alpha * A @ B + beta * C.
    // Here we want: output = 1.0 * output + lora_scale * (temp @ B^T)
    // So alpha = lora_scale, beta = 1.0f
    cublasSgemm(handle, 
                CUBLAS_OP_N, CUBLAS_OP_N, 
                d_out, batch, rank, 
                &lora_scale, 
                B, d_out, 
                d_temp, rank, 
                &alpha1, 
                output, d_out);

    // Clean up resources
    cudaFree(d_temp);
    cublasDestroy(handle);
}
