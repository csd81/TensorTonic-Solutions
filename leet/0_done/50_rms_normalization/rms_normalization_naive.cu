#include <cuda_runtime.h>

#include <cmath>

// Naive RMS normalization: y_i = gamma * x_i / sqrt(mean(x^2) + eps) + beta.
//
// The output element depends on a whole-array reduction, so a scratch buffer
// is genuinely required between the two passes.  The first kernel gives one
// thread to each contiguous chunk of NAIVE_CHUNK elements and stores the
// chunk's sum of squares as a double.  The second kernel gives one thread per
// output element; every thread sums the small per-chunk array itself and then
// writes its element.  No shared memory, no vector loads, no warp intrinsics.

#define NAIVE_CHUNK 1024

__global__ void rms_sumsq_naive_kernel(const float* __restrict__ x,
                                       double* __restrict__ partial, int N) {
  const int b = blockIdx.x;
  const int start = b * NAIVE_CHUNK;
  const int end = (start + NAIVE_CHUNK < N) ? start + NAIVE_CHUNK : N;
  double s = 0.0;
  for (int i = start; i < end; ++i) {
    const double v = static_cast<double>(x[i]);
    s += v * v;
  }
  partial[b] = s;
}

__global__ void rms_apply_naive_kernel(const float* __restrict__ x,
                                       float* __restrict__ y,
                                       const double* __restrict__ partial,
                                       int num_chunks, int N, float gamma,
                                       float beta, float eps) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  double total = 0.0;
  for (int b = 0; b < num_chunks; ++b) total += partial[b];

  const float mean_sq = static_cast<float>(total / static_cast<double>(N));
  const float rms = sqrtf(mean_sq + eps);
  const float scale = gamma / rms;
  y[i] = scale * x[i] + beta;
}

static double* g_partial = nullptr;
static int g_partial_capacity = 0;

// input, output are device pointers; gamma and beta are scalars.
extern "C" void solve(const float* input, float gamma, float beta, float* output,
                      int N, float eps) {
  if (input == nullptr || output == nullptr || N <= 0) return;

  const int num_chunks = (N + NAIVE_CHUNK - 1) / NAIVE_CHUNK;
  if (num_chunks > g_partial_capacity) {
    cudaFree(g_partial);
    cudaMalloc(&g_partial, static_cast<size_t>(num_chunks) * sizeof(double));
    g_partial_capacity = num_chunks;
  }

  rms_sumsq_naive_kernel<<<num_chunks, 1>>>(input, g_partial, N);

  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  rms_apply_naive_kernel<<<blocks, threads>>>(input, output, g_partial,
                                              num_chunks, N, gamma, beta, eps);
  cudaDeviceSynchronize();
}
