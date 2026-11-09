#include <cuda_runtime.h>

#include <cstdint>

// One fused AdamW step, in place:
//   p *= (1 - lr*wd);
//   m = b1*m + (1-b1)*g;  v = b2*v + (1-b2)*g*g;
//   p -= lr * (m/(1-b1^t)) / (sqrt(v/(1-b2^t)) + eps);
//
// At the benchmark (N = 16,777,216) the kernel reads params, grad, m and v
// (4 x 64 MB) and writes params, m and v (3 x 64 MB) — 448 MB — so it is
// memory-bound: the starter's scalar kernel measures 5.66-5.76 ms (82-83
// GB/s) and this float4 version measures the same 5.71-5.76 ms.  The sqrt
// and the division are completely hidden behind the traffic.
//
// The shipped version processes four elements per thread with float4 loads
// and stores (fewer instructions for the same traffic), with the N % 4 tail
// handled in-kernel and a scalar kernel for misaligned pointers.  The bias
// corrections are computed once on the host.

namespace {

__device__ __forceinline__ float adamw_step(float p, float g, float& m, float& v,
                                            float lr, float b1, float b2,
                                            float eps, float wd, float ibc1,
                                            float ibc2) {
  p *= (1.0f - lr * wd);  // decoupled weight decay, pre-update value
  m = b1 * m + (1.0f - b1) * g;
  v = b2 * v + (1.0f - b2) * g * g;
  const float mh = m * ibc1;
  const float vh = v * ibc2;
  p -= lr * mh / (sqrtf(vh) + eps);
  return p;
}

__global__ void adamw_vec_kernel(float4* __restrict__ P,
                                 const float4* __restrict__ G,
                                 float4* __restrict__ M, float4* __restrict__ V,
                                 int n4, int N, float lr, float b1, float b2,
                                 float eps, float wd, float ibc1, float ibc2) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) {
    float4 p = __ldg(&P[i]);
    const float4 g = __ldg(&G[i]);
    float4 m = __ldg(&M[i]);
    float4 v = __ldg(&V[i]);
    p.x = adamw_step(p.x, g.x, m.x, v.x, lr, b1, b2, eps, wd, ibc1, ibc2);
    p.y = adamw_step(p.y, g.y, m.y, v.y, lr, b1, b2, eps, wd, ibc1, ibc2);
    p.z = adamw_step(p.z, g.z, m.z, v.z, lr, b1, b2, eps, wd, ibc1, ibc2);
    p.w = adamw_step(p.w, g.w, m.w, v.w, lr, b1, b2, eps, wd, ibc1, ibc2);
    P[i] = p;
    M[i] = m;
    V[i] = v;
  } else if (i == n4) {
    // Scalar tail: the last N % 4 elements, handled by one thread.
    float* P1 = reinterpret_cast<float*>(P);
    const float* G1 = reinterpret_cast<const float*>(G);
    float* M1 = reinterpret_cast<float*>(M);
    float* V1 = reinterpret_cast<float*>(V);
    for (int k = 4 * n4; k < N; ++k) {
      float m = M1[k], v = V1[k];
      const float p = adamw_step(P1[k], G1[k], m, v, lr, b1, b2, eps, wd, ibc1,
                                 ibc2);
      P1[k] = p;
      M1[k] = m;
      V1[k] = v;
    }
  }
}

__global__ void adamw_scalar_kernel(float* __restrict__ P,
                                    const float* __restrict__ G,
                                    float* __restrict__ M, float* __restrict__ V,
                                    int N, float lr, float b1, float b2,
                                    float eps, float wd, float ibc1,
                                    float ibc2) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += gridDim.x * blockDim.x) {
    float m = M[i], v = V[i];
    const float p = adamw_step(P[i], G[i], m, v, lr, b1, b2, eps, wd, ibc1, ibc2);
    P[i] = p;
    M[i] = m;
    V[i] = v;
  }
}

}  // namespace

// params, grad, m, v are device pointers; params, m, v are updated in place
extern "C" void solve(float* params, const float* grad, float* m, float* v,
                      int N, float lr, float beta1, float beta2, float eps,
                      float weight_decay, int t) {
  if (N <= 0) return;
  const float bc1 = 1.0f - powf(beta1, (float)t);
  const float bc2 = 1.0f - powf(beta2, (float)t);
  const float ibc1 = 1.0f / bc1;
  const float ibc2 = 1.0f / bc2;

  const bool aligned =
      ((reinterpret_cast<uintptr_t>(params) | reinterpret_cast<uintptr_t>(grad) |
        reinterpret_cast<uintptr_t>(m) | reinterpret_cast<uintptr_t>(v)) &
       15u) == 0;
  if (aligned) {
    const int n4 = N >> 2;
    const int threads = 256;
    const int blocks = (n4 + 1 + threads - 1) / threads;  // covers the tail thread
    adamw_vec_kernel<<<blocks, threads>>>(
        reinterpret_cast<float4*>(params), reinterpret_cast<const float4*>(grad),
        reinterpret_cast<float4*>(m), reinterpret_cast<float4*>(v), n4, N, lr,
        beta1, beta2, eps, weight_decay, ibc1, ibc2);
  } else {
    const int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    adamw_scalar_kernel<<<blocks, threads>>>(params, grad, m, v, N, lr, beta1,
                                             beta2, eps, weight_decay, ibc1,
                                             ibc2);
  }
  cudaDeviceSynchronize();
}
