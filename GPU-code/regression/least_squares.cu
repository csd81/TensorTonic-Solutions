// least_squares.cu -- CUDA C++ port of regression/least_squares.py
//
// Univariate ordinary least squares:  y = beta1 * x + beta0
//
// One kernel performs four parallel reductions over the data:
//   sum(x), sum(y), sum(x^2), sum(x*y)
// with shared-memory block trees and one atomicAdd per block, exactly like
// the Numba version.  The host then computes
//   beta1 = (N*sum_xy - sum_x*sum_y) / (N*sum_xx - sum_x^2)
//   beta0 = (sum_y - beta1*sum_x) / N
//
// float64 is used throughout for numerical stability.  atomicAdd(double)
// requires compute capability >= 6.0.
//
// Build:
//   nvcc -O3 -arch=native -o least_squares least_squares.cu
// (T4: -arch=sm_75, Quadro P2000: -arch=sm_61)

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>

#define THREADS_PER_BLOCK 256

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err__ = (call);                                               \
    if (err__ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                    \
                   cudaGetErrorName(err__), __FILE__, __LINE__,               \
                   cudaGetErrorString(err__));                                \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

using Clock = std::chrono::steady_clock;
static double elapsed_ms(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

__global__ void linear_regression_kernel(const double* __restrict__ x,
                                         const double* __restrict__ y, int N,
                                         double* __restrict__ out) {
  __shared__ double s_x[THREADS_PER_BLOCK];
  __shared__ double s_y[THREADS_PER_BLOCK];
  __shared__ double s_xx[THREADS_PER_BLOCK];
  __shared__ double s_xy[THREADS_PER_BLOCK];

  const int tid = threadIdx.x;
  const int gid = blockIdx.x * blockDim.x + tid;
  const int stride = blockDim.x * gridDim.x;

  double local_sum_x = 0.0;
  double local_sum_y = 0.0;
  double local_sum_xx = 0.0;
  double local_sum_xy = 0.0;

  for (int i = gid; i < N; i += stride) {
    const double xi = x[i];
    const double yi = y[i];
    local_sum_x += xi;
    local_sum_y += yi;
    local_sum_xx += xi * xi;
    local_sum_xy += xi * yi;
  }

  s_x[tid] = local_sum_x;
  s_y[tid] = local_sum_y;
  s_xx[tid] = local_sum_xx;
  s_xy[tid] = local_sum_xy;
  __syncthreads();

  for (int step = blockDim.x / 2; step > 0; step >>= 1) {
    if (tid < step) {
      s_x[tid] += s_x[tid + step];
      s_y[tid] += s_y[tid + step];
      s_xx[tid] += s_xx[tid + step];
      s_xy[tid] += s_xy[tid + step];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(&out[0], s_x[0]);
    atomicAdd(&out[1], s_y[0]);
    atomicAdd(&out[2], s_xx[0]);
    atomicAdd(&out[3], s_xy[0]);
  }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

struct LinearFit {
  double beta1 = 0.0;
  double beta0 = 0.0;
  double sum_x = 0.0;
  double sum_y = 0.0;
  double sum_xx = 0.0;
  double sum_xy = 0.0;
  double h2d_ms = 0.0;
  double kernel_ms = 0.0;
  double d2h_ms = 0.0;
  double total_ms = 0.0;
};

static LinearFit fit_linear_regression_gpu(const double* x, const double* y,
                                           int N) {
  if (N < 2) {
    throw std::invalid_argument("at least two points are required");
  }

  LinearFit fit;

  double* d_x = nullptr;
  double* d_y = nullptr;
  double* d_out = nullptr;
  const size_t bytes = static_cast<size_t>(N) * sizeof(double);

  const auto total_t0 = Clock::now();

  const auto h2d_t0 = Clock::now();
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_x), bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_y), bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out), 4 * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, y, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out, 0, 4 * sizeof(double)));
  CUDA_CHECK(cudaDeviceSynchronize());
  fit.h2d_ms = elapsed_ms(h2d_t0);

  const int threads = THREADS_PER_BLOCK;
  const int blocks = static_cast<int>(std::min<long long>(
      1024LL, (static_cast<long long>(N) + threads - 1) / threads));

  cudaEvent_t k0, k1;
  CUDA_CHECK(cudaEventCreate(&k0));
  CUDA_CHECK(cudaEventCreate(&k1));

  CUDA_CHECK(cudaEventRecord(k0));
  linear_regression_kernel<<<blocks, threads>>>(d_x, d_y, N, d_out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(k1));
  CUDA_CHECK(cudaDeviceSynchronize());

  float kernel_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, k0, k1));
  fit.kernel_ms = kernel_ms;

  double h_out[4] = {0.0, 0.0, 0.0, 0.0};
  const auto d2h_t0 = Clock::now();
  CUDA_CHECK(cudaMemcpy(h_out, d_out, 4 * sizeof(double),
                        cudaMemcpyDeviceToHost));
  fit.d2h_ms = elapsed_ms(d2h_t0);

  fit.sum_x = h_out[0];
  fit.sum_y = h_out[1];
  fit.sum_xx = h_out[2];
  fit.sum_xy = h_out[3];

  const double denominator =
      static_cast<double>(N) * fit.sum_xx - fit.sum_x * fit.sum_x;

  // Scale-aware singularity check: an absolute isclose(x, 0) test would only
  // catch exactly zero (relative tolerance against 0 is useless).
  const double scale = std::fabs(static_cast<double>(N) * fit.sum_xx) +
                       std::fabs(fit.sum_x * fit.sum_x);
  if (std::fabs(denominator) <= 1e-12 * scale) {
    CUDA_CHECK(cudaEventDestroy(k0));
    CUDA_CHECK(cudaEventDestroy(k1));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_out));
    throw std::invalid_argument(
        "degenerate input: x values are (nearly) constant");
  }

  fit.beta1 = (static_cast<double>(N) * fit.sum_xy - fit.sum_x * fit.sum_y) /
              denominator;
  fit.beta0 = (fit.sum_y - fit.beta1 * fit.sum_x) / static_cast<double>(N);

  fit.total_ms = elapsed_ms(total_t0);

  CUDA_CHECK(cudaEventDestroy(k0));
  CUDA_CHECK(cudaEventDestroy(k1));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_out));
  return fit;
}

static LinearFit fit_linear_regression_gpu(const std::vector<double>& x,
                                           const std::vector<double>& y) {
  if (x.size() != y.size()) {
    throw std::invalid_argument("x and y must have the same length");
  }
  return fit_linear_regression_gpu(x.data(), y.data(),
                                   static_cast<int>(x.size()));
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

int main() {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s (FP32/FP64 ratio %d)\n", prop.name,
              prop.singleToDoublePrecisionPerfRatio);

  constexpr int N = 10000000;
  constexpr double TRUE_SLOPE = 2.5;
  constexpr double TRUE_INTERCEPT = 1.0;

  std::mt19937 gen(42);
  std::uniform_real_distribution<double> uniform(0.0, 100.0);
  std::normal_distribution<double> noise(0.0, 1.0);

  std::vector<double> x(N);
  std::vector<double> y(N);
  for (int i = 0; i < N; ++i) {
    x[i] = uniform(gen);
    y[i] = TRUE_SLOPE * x[i] + TRUE_INTERCEPT + noise(gen);
  }

  const LinearFit fit = fit_linear_regression_gpu(x, y);

  // Sequential CPU reference over the same data.
  double sum_x = 0.0;
  double sum_y = 0.0;
  double sum_xx = 0.0;
  double sum_xy = 0.0;
  for (int i = 0; i < N; ++i) {
    sum_x += x[i];
    sum_y += y[i];
    sum_xx += x[i] * x[i];
    sum_xy += x[i] * y[i];
  }
  const double denom = static_cast<double>(N) * sum_xx - sum_x * sum_x;
  const double cpu_beta1 =
      (static_cast<double>(N) * sum_xy - sum_x * sum_y) / denom;
  const double cpu_beta0 =
      (sum_y - cpu_beta1 * sum_x) / static_cast<double>(N);

  std::printf("N=%d, true slope/intercept: %.6f / %.6f\n", N, TRUE_SLOPE,
              TRUE_INTERCEPT);
  std::printf("GPU: beta1=%.12f beta0=%.12f\n", fit.beta1, fit.beta0);
  std::printf("CPU: beta1=%.12f beta0=%.12f\n", cpu_beta1, cpu_beta0);
  std::printf("|diff|: beta1=%.3e beta0=%.3e\n",
              std::fabs(fit.beta1 - cpu_beta1),
              std::fabs(fit.beta0 - cpu_beta0));
  std::printf("time: total=%.3f ms (h2d=%.3f kernel=%.3f d2h=%.3f)\n",
              fit.total_ms, fit.h2d_ms, fit.kernel_ms, fit.d2h_ms);

  return EXIT_SUCCESS;
}
