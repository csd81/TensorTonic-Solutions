#include <cuda_runtime.h>
#include <math.h>

// Coaxial / Coarse-grained thread block mapping for fused operations
// Assumes 1D thread blocks where threads within a block calculate the row statistics 
// and then cooperate to perform the linear projection.
__global__ void fused_transformer_mlp_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ gamma, 
    const float* __restrict__ beta, 
    const float* __restrict__ W,
    const float* __restrict__ bias, 
    float* __restrict__ y, 
    int M, int K, int N, float eps) 
{
    // A block coordinates a single row of the computation or a set of rows
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M) return;

    // 1. Compute mean and variance for the current row
    // In production settings with large K (up to 8,192), we loop over K elements.
    float sum_val = 0.0f;
    float sum_sq_val = 0.0f;

    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float val = x[row * K + k];
        sum_val += val;
        sum_sq_val += val * val;
    }

    // Block-wide parallel reduction for sum and sum_sq
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        sum_val += __shfl_down_sync(0xffffffff, sum_val, offset);
        sum_sq_val += __shfl_down_sync(0xffffffff, sum_sq_val, offset);
    }

    // Broadcast the reduced row statistics to all threads in the warp/block
    __shared__ float shared_mean;
    __shared__ float shared_inv_std;
    
    if (threadIdx.x == 0) {
        float mean = sum_val / K;
        float variance = (sum_sq_val / K) - (mean * mean);
        if (variance < 0.0f) variance = 0.0f; // numerical safety
        shared_mean = mean;
        shared_inv_std = rsqrtf(variance + eps);
    }
    __syncthreads();

    float mean = shared_mean;
    float inv_std = shared_inv_std;

    // 2. Linear projection (GEMM) paired with epilogue processing
    // Iterate over output columns N
    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        float h_acc = 0.0f;

        // Perform the inner product with matrix W
        for (int k = 0; k < K; ++k) {
            float x_raw = x[row * K + k];
            // Compute LayerNorm value on-the-fly without global memory traffic
            float x_norm = ((x_raw - mean) * inv_std) * gamma[k] + beta[k];
            // Accumulate projection (Assuming W is stored in Column-Major order or Row-Major order)
            // Defaulting standard Row-Major matching layout: W is (K x N) -> index: k * N + col
            h_acc += x_norm * W[k * N + col];
        }

        // Add bias element
        float h = h_acc + bias[col];

        // 3. Exact erf-based GELU Activation Function
        // y = 0.5 * h * (1 + erf(h / sqrt(2)))
        float erf_val = erf_f(h * 0.7071067811865475f); // 1.0f / sqrtf(2.0f)
        y[row * N + col] = 0.5f * h * (1.0f + erf_val);
    }
}

extern "C" void solve(const float* x, const float* gamma, const float* beta, const float* W,
                      const float* bias, float* y, int M, int K, int N, float eps) 
{
    // Configure executing thread configuration
    // Utilizing 256 threads along the X dimension to maximize throughput 
    // and parallel processing capability across vectors.
    dim3 threadsPerBlock(256, 1);
    dim3 numBlocks(1, M); 

    fused_transformer_mlp_kernel<<<numBlocks, threadsPerBlock>>>(x, gamma, beta, W, bias, y, M, K, N, eps);
}
