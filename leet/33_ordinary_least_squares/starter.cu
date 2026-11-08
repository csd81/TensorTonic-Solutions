#include <cuda_runtime.h>
#include <cmath>

// Kernel 1: Compute A = X^T * X (F x F) and b = X^T * y (F) in double precision
// Augmented matrix Aug has dimensions F x (F + 1), storing [A | b]
__global__ void compute_normal_equations(
    const float* __restrict__ X,
    const float* __restrict__ y,
    double* __restrict__ Aug,
    int n_samples,
    int n_features
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n_features && col <= n_features) {
        double sum = 0.0;
        if (col < n_features) {
            // Compute (X^T * X)[row, col]
            for (int k = 0; k < n_samples; ++k) {
                sum += static_cast<double>(X[k * n_features + row]) *
                       static_cast<double>(X[k * n_features + col]);
            }
        } else {
            // Last column: Compute (X^T * y)[row]
            for (int k = 0; k < n_samples; ++k) {
                sum += static_cast<double>(X[k * n_features + row]) *
                       static_cast<double>(y[k]);
            }
        }
        Aug[row * (n_features + 1) + col] = sum;
    }
}

// Kernel 2a: Find pivot row for column `k` and swap rows `k` and `pivot`
__global__ void pivot_and_swap(double* Aug, int n_features, int k) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int stride = n_features + 1;
        int pivot_row = k;
        double max_val = fabs(Aug[k * stride + k]);

        for (int i = k + 1; i < n_features; ++i) {
            double val = fabs(Aug[i * stride + k]);
            if (val > max_val) {
                max_val = val;
                pivot_row = i;
            }
        }

        if (pivot_row != k) {
            for (int j = k; j <= n_features; ++j) {
                double tmp = Aug[k * stride + j];
                Aug[k * stride + j] = Aug[pivot_row * stride + j];
                Aug[pivot_row * stride + j] = tmp;
            }
        }

        // Normalize pivot row so Aug[k, k] == 1.0
        double pivot_val = Aug[k * stride + k];
        for (int j = k; j <= n_features; ++j) {
            Aug[k * stride + j] /= pivot_val;
        }
    }
}

// Kernel 2b: Eliminate column `k` from all other rows `i != k`
__global__ void eliminate_column(double* Aug, int n_features, int k) {
    int col = blockIdx.x * blockDim.x + threadIdx.x + k;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int stride = n_features + 1;

    if (row < n_features && row != k && col <= n_features) {
        double factor = Aug[row * stride + k];
        // Note: To avoid race conditions on column `k` itself while reading `factor`,
        // we only update columns `col > k` here and zero out column `k` afterward,
        // or read `factor` before any thread writes to column `k`.
        if (col > k) {
            Aug[row * stride + col] -= factor * Aug[k * stride + col];
        }
    }
}

// Kernel 3: Extract the final column of the reduced augmented matrix into beta
__global__ void extract_solution(const double* __restrict__ Aug, float* __restrict__ beta, int n_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_features) {
        beta[idx] = static_cast<float>(Aug[idx * (n_features + 1) + n_features]);
    }
}

// X, y, beta are device pointers
extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) {
    double* d_Aug = nullptr;
    size_t aug_bytes = static_cast<size_t>(n_features) * (n_features + 1) * sizeof(double);
    cudaMalloc(&d_Aug, aug_bytes);

    // 1. Build augmented matrix [X^T X | X^T y]
    dim3 block2D(16, 16);
    dim3 grid_build(
        (n_features + 1 + block2D.x - 1) / block2D.x,
        (n_features + block2D.y - 1) / block2D.y
    );
    compute_normal_equations<<<grid_build, block2D>>>(X, y, d_Aug, n_samples, n_features);

    // 2. Gauss-Jordan Elimination
    for (int k = 0; k < n_features; ++k) {
        pivot_and_swap<<<1, 1>>>(d_Aug, n_features, k);

        int rem_cols = (n_features + 1) - k;
        dim3 grid_elim(
            (rem_cols + block2D.x - 1) / block2D.x,
            (n_features + block2D.y - 1) / block2D.y
        );
        eliminate_column<<<grid_elim, block2D>>>(d_Aug, n_features, k);
    }

    // 3. Copy solution from augmented column to float* beta
    int threads1D = 256;
    int blocks1D = (n_features + threads1D - 1) / threads1D;
    extract_solution<<<blocks1D, threads1D>>>(d_Aug, beta, n_features);

    cudaFree(d_Aug);
}
