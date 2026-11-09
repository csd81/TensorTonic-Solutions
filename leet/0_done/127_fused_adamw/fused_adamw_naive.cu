#include <cuda_runtime.h>
#include <math.h>

// Naive fused AdamW: one thread per element with scalar loads and stores.
// No vector types, no grid-stride loop -- the plainest correct form (and the
// starter's shape).

__global__ void adamw_naive_kernel(float* params, const float* grad, float* m,
                                   float* v, int N, float lr, float beta1,
                                   float beta2, float eps, float weight_decay,
                                   float bias_corr1, float bias_corr2) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  float p_val = params[i];
  p_val *= (1.0f - lr * weight_decay);

  const float g_val = grad[i];
  float m_val = m[i];
  float v_val = v[i];

  m_val = beta1 * m_val + (1.0f - beta1) * g_val;
  v_val = beta2 * v_val + (1.0f - beta2) * g_val * g_val;

  const float m_hat = m_val / bias_corr1;
  const float v_hat = v_val / bias_corr2;
  p_val -= lr * m_hat / (sqrtf(v_hat) + eps);

  params[i] = p_val;
  m[i] = m_val;
  v[i] = v_val;
}

// params, grad, m, v are device pointers
extern "C" void solve(float* params, const float* grad, float* m, float* v,
                      int N, float lr, float beta1, float beta2, float eps,
                      float weight_decay, int t) {
  if (N <= 0) return;
  const float bias_corr1 = 1.0f - powf(beta1, (float)t);
  const float bias_corr2 = 1.0f - powf(beta2, (float)t);
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  adamw_naive_kernel<<<blocks, threads>>>(params, grad, m, v, N, lr, beta1,
                                          beta2, eps, weight_decay, bias_corr1,
                                          bias_corr2);
  cudaDeviceSynchronize();
}
