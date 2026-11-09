#include <cuda_runtime.h>
#include <math.h>

// Naive sigmoid: one thread per element, plain scalar load/store, expf and
// IEEE division.  No vector types, no fast intrinsics -- the plainest correct
// form (and the starter's shape).

__global__ void sigmoid_naive_kernel(const float* __restrict__ X,
                                     float* __restrict__ Y, int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  Y[i] = 1.0f / (1.0f + expf(-X[i]));
}

// X, Y are device pointers
extern "C" void solve(const float* X, float* Y, int N) {
  if (N <= 0) return;
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  sigmoid_naive_kernel<<<blocks, threads>>>(X, Y, N);
  cudaDeviceSynchronize();
}
