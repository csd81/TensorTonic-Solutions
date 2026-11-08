
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

// 1. Compute predictions p = sigmoid(X * beta) and weights w = p * (1 - p)
__global__ void compute_probs_and_weights(
    const float* __restrict__ X,
    const float* __restrict__ y,
    const float* __restrict__ beta,
    float* __restrict__ diff,    // y - p
    float* __restrict__ weights, // p * (1 - p)
    int n_samples,
    int n_features) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_samples) {
        float z = 0.0f;
        const float* row = X + i * n_features;
        for (int j = 0; j < n_features; ++j) {
            z += row[j] * beta[j];
        }
        // Numerically stable sigmoid
        float p = (z >= 0.0f) ? (1.0f / (1.0f + expf(-z))) : (expf(z) / (1.0f + expf(z)));
        diff[i] = y[i] - p;
        weights[i] = fmaxf(p * (1.0f - p), 1e-6f); // Clamp to avoid zero weights
    }
}

// 2. Compute Gradient g = X^T * (y - p)
__global__ void compute_gradient(
    const float* __restrict__ X,
    const float* __restrict__ diff,
    float* __restrict__ grad,
    int n_samples,
    int n_features) 
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n_features) {
        float sum = 0.0f;
        for (int i = 0; i < n_samples; ++i) {
            sum += X[i * n_features + j] * diff[i];
        }
        grad[j] = sum;
    }
}

// 3. Compute Hessian H = X^T * W * X + lambda * I
__global__ void compute_hessian(
    const float* __restrict__ X,
    const float* __restrict__ weights,
    float* __restrict__ H,
    int n_samples,
    int n_features,
    float reg) 
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n_features && col <= row) {
        float sum = 0.0f;
        for (int i = 0; i < n_samples; ++i) {
            sum += X[i * n_features + row] * weights[i] * X[i * n_features + col];
        }
        if (row == col) {
            sum += reg;
        }
        H[row * n_features + col] = sum;
        H[col * n_features + row] = sum; // Symmetric
    }
}

// 4. Solve H * delta = grad using Cholesky Decomposition (Single Block for F <= 1000)
__global__ void solve_cholesky(
    float* H,
    const float* grad,
    float* beta,
    int n_features) 
{
    int tid = threadIdx.x;
    
    // Cholesky factorization: H = L * L^T (in-place lower triangle)
    for (int k = 0; k < n_features; ++k) {
        if (tid == 0) {
            float diag = H[k * n_features + k];
            for (int p = 0; p < k; ++p) {
                float val = H[k * n_features + p];
                diag -= val * val;
            }
            H[k * n_features + k] = sqrtf(fmaxf(diag, 1e-8f));
        }
        __syncthreads();

        float l_kk = H[k * n_features + k];
        for (int i = k + 1 + tid; i < n_features; i += blockDim.x) {
            float sum = H[i * n_features + k];
            for (int p = 0; p < k; ++p) {
                sum -= H[i * n_features + p] * H[k * n_features + p];
            }
            H[i * n_features + k] = sum / l_kk;
        }
        __syncthreads();
    }

    // Forward & Backward substitution on thread 0
    if (tid == 0) {
        // Use upper triangle of H as temporary workspace for y_vec
        // Forward substitution: L * y_vec = grad
        for (int i = 0; i < n_features; ++i) {
            float sum = grad[i];
            for (int j = 0; j < i; ++j) {
                sum -= H[i * n_features + j] * H[j * n_features + i]; // H[j*F + i] stores y_vec[j]
            }
            // Store y_vec[i] in grad-like fashion (using a local array or upper triangle)
            // Note: Since j < i, upper triangle [j * F + i] is free!
            if (i > 0) {
                H[0 * n_features + i] = sum / H[i * n_features + i]; 
            } else {
                // Handle i = 0 safely using a separate pass or local buffer
            }
        }
    }
}


__global__ void solve_linear_system(
    float* __restrict__ H,
    const float* __restrict__ grad,
    float* __restrict__ delta,
    float* __restrict__ beta,
    int n_features)
{
    int tid = threadIdx.x;

    // Parallel Cholesky: H = L * L^T
    for (int k = 0; k < n_features; ++k) {
        if (tid == 0) {
            float diag = H[k * n_features + k];
            for (int p = 0; p < k; ++p) {
                float v = H[k * n_features + p];
                diag -= v * v;
            }
            H[k * n_features + k] = sqrtf(fmaxf(diag, 1e-8f));
        }
        __syncthreads();

        float l_kk = H[k * n_features + k];
        for (int i = k + 1 + tid; i < n_features; i += blockDim.x) {
            float sum = H[i * n_features + k];
            for (int p = 0; p < k; ++p) {
                sum -= H[i * n_features + p] * H[k * n_features + p];
            }
            H[i * n_features + k] = sum / l_kk;
        }
        __syncthreads();
    }

    // Forward & Backward substitution + Beta update
    if (tid == 0) {
        // Forward: L * delta = grad
        for (int i = 0; i < n_features; ++i) {
            float sum = grad[i];
            for (int j = 0; j < i; ++j) {
                sum -= H[i * n_features + j] * delta[j];
            }
            delta[i] = sum / H[i * n_features + i];
        }
        // Backward: L^T * delta = delta
        for (int i = n_features - 1; i >= 0; --i) {
            float sum = delta[i];
            for (int j = i + 1; j < n_features; ++j) {
                sum -= H[j * n_features + i] * delta[j];
            }
            delta[i] = sum / H[i * n_features + i];
            beta[i] += delta[i];
        }
    }
}

// X, y, beta are device pointers
extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) {
    cudaMemset(beta, 0, n_features * sizeof(float));

    float *d_diff, *d_weights, *d_grad, *d_H, *d_delta;
    cudaMalloc(&d_diff, n_samples * sizeof(float));
    cudaMalloc(&d_weights, n_samples * sizeof(float));
    cudaMalloc(&d_grad, n_features * sizeof(float));
    cudaMalloc(&d_H, n_features * n_features * sizeof(float));
    cudaMalloc(&d_delta, n_features * sizeof(float));

    int threads = 256;
    int sample_blocks = (n_samples + threads - 1) / threads;
    int feature_blocks = (n_features + threads - 1) / threads;

    dim3 hess_block(16, 16);
    dim3 hess_grid((n_features + 15) / 16, (n_features + 15) / 16);

    const int max_iters = 15;
    const float reg = 1e-4f; // Small L2 regularization for numerical stability

    for (int iter = 0; iter < max_iters; ++iter) {
        compute_probs_and_weights<<<sample_blocks, threads>>>(
            X, y, beta, d_diff, d_weights, n_samples, n_features);

        compute_gradient<<<feature_blocks, threads>>>(
            X, d_diff, d_grad, n_samples, n_features);

        compute_hessian<<<hess_grid, hess_block>>>(
            X, d_weights, d_H, n_samples, n_features, reg);

        solve_linear_system<<<1, 256>>>(
            d_H, d_grad, d_delta, beta, n_features);
    }

    cudaFree(d_diff);
    cudaFree(d_weights);
    cudaFree(d_grad);
    cudaFree(d_H);
    cudaFree(d_delta);
}