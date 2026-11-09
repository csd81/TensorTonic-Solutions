#include <cuda_runtime.h>
#include <cublas_v2.h>

// LoRA linear with cuBLAS (reference implementation only -- the challenge
// forbids external libraries, and this environment has no cuBLAS installed,
// so this file cannot be built here).
//
// Layout notes: cuBLAS is column-major, while x/W/A/B/output are row-major.
// A row-major [rows, cols] buffer read as column-major with lda = rows is its
// transpose, so each product is formed transposed:
//
//   output   [batch, d_out]  ==  outputᵀ [d_out, batch]  (column-major, ld d_out)
//   outputᵀ = (x Wᵀ)ᵀ = W xᵀ
//
// With op(A) = W (d_out x d_in) obtained by CUBLAS_OP_T on the stored
// column-major Wᵀ (d_in x d_out, ld d_in), and op(B) = xᵀ (d_in x batch)
// obtained with CUBLAS_OP_N on x (ld d_in):
//
//   cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, d_out, batch, d_in,
//               &alpha, W, d_in, x, d_in, &beta, output, d_out);
//
// The previous version of this file used (CUBLAS_OP_N, CUBLAS_OP_T) with
// lda = d_out / ldb = d_in, which reads W as Wᵀ and computes x·W instead of
// x·Wᵀ (and mis-maps the other two products the same way).

extern "C" void solve(const float* x, const float* W, const float* A,
                      const float* B, float* output, int batch, int d_in,
                      int d_out, int rank, float lora_scale) {
    if (batch <= 0 || d_in <= 0 || d_out <= 0 || rank <= 0) return;

    cublasHandle_t handle;
    cublasCreate(&handle);

    // temp = x @ Aᵀ  [batch, rank], stored row-major.
    float* d_temp = nullptr;
    cudaMalloc(&d_temp, (size_t)batch * rank * sizeof(float));

    const float one = 1.0f;
    const float zero = 0.0f;

    // 1. tempᵀ (rank x batch) = A (rank x d_in) @ xᵀ (d_in x batch)
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, rank, batch, d_in,
                &one, A, d_in, x, d_in, &zero, d_temp, rank);

    // 2. outputᵀ (d_out x batch) = W (d_out x d_in) @ xᵀ (d_in x batch)
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, d_out, batch, d_in,
                &one, W, d_in, x, d_in, &zero, output, d_out);

    // 3. outputᵀ += lora_scale * B (d_out x rank) @ tempᵀ (rank x batch)
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, d_out, batch, rank,
                &lora_scale, B, rank, d_temp, rank, &one, output, d_out);

    cudaFree(d_temp);
    cublasDestroy(handle);
}
