#include <cuda_runtime.h>

// Naive RoPE: one thread per element, scalar loads of Q, cos, sin and one
// store, with the column computed by an integer modulus.  No float4, no
// half-split reuse of cos/sin -- the plainest correct form (and the
// starter's shape).

__global__ void rope_naive_kernel(const float* __restrict__ Q,
                                  const float* __restrict__ cos_vals,
                                  const float* __restrict__ sin_vals,
                                  float* __restrict__ output, long total,
                                  int D) {
  const int half = D / 2;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += (long)gridDim.x * blockDim.x) {
    const int col = (int)(i % D);
    const float q = Q[i];
    const float rot = (col < half) ? -Q[i + half] : Q[i - half];
    output[i] = q * cos_vals[i] + rot * sin_vals[i];
  }
}

// Q, cos, sin, output are device pointers
extern "C" void solve(float* Q, float* cos_vals, float* sin_vals, float* output,
                      int M, int D) {
  if (M <= 0 || D <= 0) return;
  const long total = (long)M * D;
  const int threads = 256;
  long want = (total + threads - 1) / threads;
  int blocks = (int)(want > 4096 ? 4096 : want);
  rope_naive_kernel<<<blocks, threads>>>(Q, cos_vals, sin_vals, output, total, D);
  cudaDeviceSynchronize();
}
