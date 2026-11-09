#include <cuda_runtime.h>

// Naive LoRA linear: three untiled matrix products, each with one thread per
// output element and a direct K-loop -- no shared memory, no tiling, no
// reuse (B rows are re-read for every output row).  The plainest correct
// form, and the starter's shape.

__global__ void matmul_transpose_kernel(const float* A, const float* B,
                                        float* C, int M, int N, int K) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float sum = 0.0f;
    for (int i = 0; i < K; ++i) sum += A[row * K + i] * B[col * K + i];
    C[row * N + col] = sum;
  }
}

__global__ void matmul_transpose_accumulate_kernel(const float* A, const float* B,
                                                   float* C, int M, int N, int K,
                                                   float scale) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float sum = 0.0f;
    for (int i = 0; i < K; ++i) sum += A[row * K + i] * B[col * K + i];
    C[row * N + col] += scale * sum;
  }
}

// x, W, A, B, output are device pointers
extern "C" void solve(const float* x, const float* W, const float* A,
                      const float* B, float* output, int batch, int d_in,
                      int d_out, int rank, float lora_scale) {
  if (batch <= 0 || d_in <= 0 || d_out <= 0 || rank <= 0) return;
  float* temp;
  cudaMalloc((void**)&temp, (size_t)batch * rank * sizeof(float));
  dim3 blockSize(16, 16);
  dim3 gridW((d_out + 15) / 16, (batch + 15) / 16);
  matmul_transpose_kernel<<<gridW, blockSize>>>(x, W, output, batch, d_out, d_in);
  dim3 gridA((rank + 15) / 16, (batch + 15) / 16);
  matmul_transpose_kernel<<<gridA, blockSize>>>(x, A, temp, batch, rank, d_in);
  matmul_transpose_accumulate_kernel<<<gridW, blockSize>>>(temp, B, output, batch,
                                                           d_out, rank, lora_scale);
  cudaDeviceSynchronize();
  cudaFree(temp);
}
