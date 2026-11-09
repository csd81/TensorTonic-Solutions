#include <cuda_runtime.h>
#include <math.h>

// Naive SWiGLU: one thread per output element, plain scalar loads of the two
// halves, expf and IEEE division.  No vector types, no fast intrinsics --
// the plainest correct form (and the starter's shape).

__global__ void swiglu_naive_kernel(const float* __restrict__ input,
                                    float* __restrict__ output, int halfN) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= halfN) return;
  const float x1 = input[i];
  const float x2 = input[i + halfN];
  output[i] = (x1 / (1.0f + expf(-x1))) * x2;
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
  if (N <= 0) return;
  const int halfN = N / 2;
  if (halfN <= 0) return;
  const int threads = 256;
  const int blocks = (halfN + threads - 1) / threads;
  swiglu_naive_kernel<<<blocks, threads>>>(input, output, halfN);
  cudaDeviceSynchronize();
}
