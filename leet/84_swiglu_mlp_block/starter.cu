#include <cuda_runtime.h>
#include <cmath>

// Fused kernel to compute: gate = x * W_gate, up = x * W_up, and hidden = SiLU(gate) * up
__global__ void swiglu_forward_fused_kernel(
    const float* __restrict__ x,
    const float* __restrict__ W_gate,
    const float* __restrict__ W_up,
    float* __restrict__ inter_hidden,
    int M, int d_model, int d_ffn) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < d_ffn) {
        float gate_val = 0.0f;
        float up_val = 0.0f;

        // Compute dot product for both projections simultaneously to reuse x[row, k] in cache
        for (int k = 0; k < d_model; ++k) {
            float x_elem = x[row * d_model + k];
            gate_val += x_elem * W_gate[k * d_ffn + col];
            up_val += x_elem * W_up[k * d_ffn + col];
        }

        // Apply SiLU: gate * sigmoid(gate)
        float sigmoid_gate = 1.0f / (1.0f + __expf(-gate_val));
        float silu_gate = gate_val * sigmoid_gate;

        // Element-wise multiplication (Hadamard product)
        inter_hidden[row * d_ffn + col] = silu_gate * up_val;
    }
}

// Kernel to compute the down projection: output = inter_hidden * W_down
__global__ void down_projection_kernel(
    const float* __restrict__ inter_hidden,
    const float* __restrict__ W_down,
    float* __restrict__ output,
    int M, int d_model, int d_ffn) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < d_model) {
        float sum = 0.0f;
        for (int k = 0; k < d_ffn; ++k) {
            sum += inter_hidden[row * d_ffn + k] * W_down[k * d_model + col];
        }
        output[row * d_model + col] = sum;
    }
}

extern "C" void solve(const float* x, const float* W_gate, const float* W_up, const float* W_down,
                      float* output, int M, int d_model, int d_ffn) 
{
    // Allocate temporary workspace on the GPU for the intermediate hidden state matrix [M, d_ffn]
    float* d_inter_hidden = nullptr;
    cudaMalloc(&d_inter_hidden, M * d_ffn * sizeof(float));

    // Configure execution configuration for the first fused kernel
    dim3 threadsPerBlock1(16, 16);
    dim3 numBlocks1((d_ffn + threadsPerBlock1.x - 1) / threadsPerBlock1.x, 
                    (M + threadsPerBlock1.y - 1) / threadsPerBlock1.y);

    swiglu_forward_fused_kernel<<<numBlocks1, threadsPerBlock1>>>(
        x, W_gate, W_up, d_inter_hidden, M, d_model, d_ffn
    );

    // Configure execution configuration for the down-projection kernel
    dim3 threadsPerBlock2(16, 16);
    dim3 numBlocks2((d_model + threadsPerBlock2.x - 1) / threadsPerBlock2.x, 
                    (M + threadsPerBlock2.y - 1) / threadsPerBlock2.y);

    down_projection_kernel<<<numBlocks2, threadsPerBlock2>>>(
        d_inter_hidden, W_down, output, M, d_model, d_ffn
    );

    // Clean up temporary workspace allocation
    cudaFree(d_inter_hidden);
}
