#include <cuda_runtime.h>
#include <math.h>

// CUDA Kernel performing the fused elementwise AdamW update
__global__ void adamw_kernel(float* params, const float* grad, float* m, float* v, int N, 
                             float lr, float beta1, float beta2, float eps, 
                             float weight_decay, float bias_corr1, float bias_corr2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // 1. Read current parameter value and apply decoupled weight decay
    float p_val = params[i];
    p_val *= (1.0f - lr * weight_decay);

    // 2. Read gradient and moments
    float g_val = grad[i];
    float m_val = m[i];
    float v_val = v[i];

    // 3. Update running first and second moments in-place
    m_val = beta1 * m_val + (1.0f - beta1) * g_val;
    v_val = beta2 * v_val + (1.0f - beta2) * g_val * g_val;

    // 4. Compute bias-corrected moments
    float m_hat = m_val / bias_corr1;
    float v_hat = v_val / bias_corr2;

    // 5. Update parameter value using the adaptive step
    p_val -= lr * m_hat / (sqrtf(v_hat) + eps);

    // 6. Write the updated values back to global memory
    params[i] = p_val;
    m[i] = m_val;
    v[i] = v_val;
}

// Host launcher function matching the required external signature
extern "C" void solve(float* params, const float* grad, float* m, float* v, int N, float lr,
                      float beta1, float beta2, float eps, float weight_decay, int t) {
    if (N <= 0) return;

    // Precompute the scalar bias correction denominators on the host to save device operations
    float bias_corr1 = 1.0f - powf(beta1, (float)t);
    float bias_corr2 = 1.0f - powf(beta2, (float)t);

    // Configure a 1D grid layout with standard block sizes (e.g., 256 threads per block)
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the fused AdamW kernel asynchronously on the default stream
    adamw_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        params, grad, m, v, N, lr, beta1, beta2, eps, weight_decay, bias_corr1, bias_corr2
    );
}
