#include <cuda_runtime.h>
#include <math.h>

// Naive GEGLU: one thread per output element, plain scalar loads of the two
// halves, erff and IEEE arithmetic.  No vector types, no grid-stride loop --
// the plainest correct form (and the starter's shape).

__global__ void geglu_naive_kernel(const float* __restrict__ input,
                                   float* __restrict__ output, int halfN) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= halfN) return;
  const float x1 = input[i];
  const float x2 = input[i + halfN];
  const float gelu = 0.5f * x2 * (1.0f + erff(x2 * 0.7071067811865475f));
  output[i] = x1 * gelu;
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
  if (N <= 0) return;
  const int halfN = N / 2;
  if (halfN <= 0) return;
  const int threads = 256;
  const int blocks = (halfN + threads - 1) / threads;
  geglu_naive_kernel<<<blocks, threads>>>(input, output, halfN);
  cudaDeviceSynchronize();
}
