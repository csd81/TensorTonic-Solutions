// conv1d.cu -- CUDA C++ port of conv1d.py (fixed version).
//
// Two 1D FIR convolution kernels:
//   1. conv1d_naive_kernel  - every thread reads its K input samples directly
//                             from global memory (zero padding at the edges).
//   2. conv1d_shared_kernel - each block stages its tile plus left/right halos
//                             in shared memory, then computes from there.
//
// The output convention matches numpy.convolve(..., mode="same"):
//   y[n] = sum_k x[n + (K-1)/2 - k] * h[k]
// including even K.  K must satisfy 1 <= K <= 2*MAX_KERNEL_RADIUS + 1.
//
// Build:
//   nvcc -O3 -arch=native -o conv1d conv1d.cu
// (T4: -arch=sm_75, Quadro P2000: -arch=sm_61)
//
// Run:
//   ./conv1d                 # N = 5,000,000, K = 31
//   ./conv1d 10000000 127    # custom N and K

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

#define TPB_1D 256
#define MAX_KERNEL_RADIUS 64
#define SHARED_SIZE (TPB_1D + 2 * MAX_KERNEL_RADIUS)

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
// Kernels
// ---------------------------------------------------------------------------

__global__ void conv1d_naive_kernel(const float* __restrict__ x,
                                    const float* __restrict__ h,
                                    float* __restrict__ y, int n_samples,
                                    int n_kernel) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int k_half = (n_kernel - 1) / 2;

  if (idx < n_samples) {
    float acc = 0.0f;
    for (int k = 0; k < n_kernel; ++k) {
      const int sample_idx = idx - (k - k_half);
      if (sample_idx >= 0 && sample_idx < n_samples) {
        acc += x[sample_idx] * h[k];
      }
    }
    y[idx] = acc;
  }
}

__global__ void conv1d_shared_kernel(const float* __restrict__ x,
                                     const float* __restrict__ h,
                                     float* __restrict__ y, int n_samples,
                                     int n_kernel) {
  __shared__ float shared_mem[SHARED_SIZE];

  const int tid = threadIdx.x;
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int k_half = (n_kernel - 1) / 2;
  const int halo_size = n_kernel / 2;
  const int shared_idx = tid + halo_size;

  // Main tile
  if (gid < n_samples) {
    shared_mem[shared_idx] = x[gid];
  } else {
    shared_mem[shared_idx] = 0.0f;
  }

  // Left halo
  if (halo_size > 0 && tid < halo_size) {
    const int left_gid = gid - halo_size;
    shared_mem[tid] =
        (left_gid >= 0 && left_gid < n_samples) ? x[left_gid] : 0.0f;
  }

  // Right halo
  if (halo_size > 0 && tid >= blockDim.x - halo_size) {
    const int right_gid = gid + halo_size;
    const int target_idx = shared_idx + halo_size;
    shared_mem[target_idx] = (right_gid < n_samples) ? x[right_gid] : 0.0f;
  }

  __syncthreads();

  if (gid < n_samples) {
    float acc = 0.0f;
    for (int k = 0; k < n_kernel; ++k) {
      const int s_idx = shared_idx - (k - k_half);
      acc += shared_mem[s_idx] * h[k];
    }
    y[gid] = acc;
  }
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

static std::vector<float> cpu_conv1d_same(const std::vector<float>& x,
                                          const std::vector<float>& h) {
  const int n = static_cast<int>(x.size());
  const int k_size = static_cast<int>(h.size());
  const int k_half = (k_size - 1) / 2;

  std::vector<float> y(n);
  for (int i = 0; i < n; ++i) {
    double acc = 0.0;
    for (int k = 0; k < k_size; ++k) {
      const int j = i + k_half - k;
      if (j >= 0 && j < n) {
        acc += static_cast<double>(x[j]) * static_cast<double>(h[k]);
      }
    }
    y[i] = static_cast<float>(acc);
  }
  return y;
}

template <typename Launch>
static double median_kernel_ms(Launch launch, int warmup = 5,
                               int repeats = 20) {
  for (int i = 0; i < warmup; ++i) {
    launch();
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<double> times;
  times.reserve(repeats);
  for (int i = 0; i < repeats; ++i) {
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));
    launch();
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
    times.push_back(ms);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
  }
  std::sort(times.begin(), times.end());
  return times[times.size() / 2];
}

static float max_abs_diff(const std::vector<float>& a,
                          const std::vector<float>& b) {
  float result = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    result = std::max(result, std::fabs(a[i] - b[i]));
  }
  return result;
}

// ---------------------------------------------------------------------------
// Demo / benchmark
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  const int n_samples = (argc > 1) ? std::atoi(argv[1]) : 5000000;
  const int kernel_size = (argc > 2) ? std::atoi(argv[2]) : 31;

  if (n_samples < 1) {
    std::fprintf(stderr, "n_samples must be >= 1\n");
    return EXIT_FAILURE;
  }
  if (kernel_size < 1 || kernel_size > 2 * MAX_KERNEL_RADIUS + 1) {
    std::fprintf(stderr, "kernel_size must be in [1, %d]\n",
                 2 * MAX_KERNEL_RADIUS + 1);
    return EXIT_FAILURE;
  }

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  const int blocks = (n_samples + TPB_1D - 1) / TPB_1D;
  std::printf("GPU: %s\n", prop.name);
  std::printf("--- 1D Konvolucios Benchmark (CUDA C++) ---\n");
  std::printf("Jel hossza: %d minta\n", n_samples);
  std::printf("Szuromag merete: %d tap\n", kernel_size);
  std::printf("Grid elrendezes: %d blokk x %d szal\n\n", blocks, TPB_1D);

  std::mt19937 gen(42);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::vector<float> x(n_samples);
  for (float& value : x) {
    value = normal(gen);
  }
  std::vector<float> h(kernel_size, 1.0f / static_cast<float>(kernel_size));

  // CPU reference (median of 3)
  std::vector<double> cpu_times;
  std::vector<float> y_cpu;
  for (int r = 0; r < 3; ++r) {
    const auto t0 = Clock::now();
    y_cpu = cpu_conv1d_same(x, h);
    cpu_times.push_back(elapsed_ms(t0));
  }
  std::sort(cpu_times.begin(), cpu_times.end());
  const double cpu_ms = cpu_times[1];
  std::printf("CPU referencia: %.2f ms (3 futas medianja)\n", cpu_ms);

  float* d_x = nullptr;
  float* d_h = nullptr;
  float* d_y_naive = nullptr;
  float* d_y_shared = nullptr;
  const size_t x_bytes = static_cast<size_t>(n_samples) * sizeof(float);
  const size_t h_bytes = static_cast<size_t>(kernel_size) * sizeof(float);

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_x), x_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_h), h_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_y_naive), x_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_y_shared), x_bytes));

  const auto h2d_t0 = Clock::now();
  CUDA_CHECK(cudaMemcpy(d_x, x.data(), x_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_h, h.data(), h_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
  const double h2d_ms = elapsed_ms(h2d_t0);
  std::printf("Host -> Device masolas: %.2f ms\n", h2d_ms);

  const auto launch_naive = [&]() {
    conv1d_naive_kernel<<<blocks, TPB_1D>>>(d_x, d_h, d_y_naive, n_samples,
                                            kernel_size);
    CUDA_CHECK(cudaGetLastError());
  };
  const auto launch_shared = [&]() {
    conv1d_shared_kernel<<<blocks, TPB_1D>>>(d_x, d_h, d_y_shared, n_samples,
                                             kernel_size);
    CUDA_CHECK(cudaGetLastError());
  };

  const double naive_ms = median_kernel_ms(launch_naive);
  std::vector<float> y_naive(n_samples);
  const auto d2h_naive_t0 = Clock::now();
  CUDA_CHECK(cudaMemcpy(y_naive.data(), d_y_naive, x_bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());
  const double d2h_naive_ms = elapsed_ms(d2h_naive_t0);
  const float err_naive = max_abs_diff(y_cpu, y_naive);

  const double shared_ms = median_kernel_ms(launch_shared);
  std::vector<float> y_shared(n_samples);
  const auto d2h_shared_t0 = Clock::now();
  CUDA_CHECK(cudaMemcpy(y_shared.data(), d_y_shared, x_bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());
  const double d2h_shared_ms = elapsed_ms(d2h_shared_t0);
  const float err_shared = max_abs_diff(y_cpu, y_shared);

  std::printf(
      "Naiv GPU:   %.2f ms (D2H: %.2f ms) | max hiba: %.2e | gyorsulas: "
      "%.1fx\n",
      naive_ms, d2h_naive_ms, err_naive, cpu_ms / naive_ms);
  std::printf(
      "Shared GPU: %.2f ms (D2H: %.2f ms) | max hiba: %.2e | gyorsulas: "
      "%.1fx\n",
      shared_ms, d2h_shared_ms, err_shared, cpu_ms / shared_ms);

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_h));
  CUDA_CHECK(cudaFree(d_y_naive));
  CUDA_CHECK(cudaFree(d_y_shared));

  return EXIT_SUCCESS;
}
