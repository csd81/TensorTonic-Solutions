// 2D-Poisson-Jacobi.cu -- CUDA C++ port of 2D-Poisson-Jacobi.py.
//
// Naive 2-D Poisson Jacobi iteration with global-memory reads:
//   u_new[i,j] = 0.25 * (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1]
//                        - h^2 f[i,j])
// Zero Dirichlet boundaries; ping-pong buffers; float32.
//
// Build:
//   nvcc -O3 -arch=native -o poisson-jacobi 2D-Poisson-Jacobi.cu
// Run:
//   ./poisson-jacobi

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#define BLOCK_X 16
#define BLOCK_Y 16

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

__global__ void poisson_jacobi_kernel(const float* __restrict__ u_old,
                                      float* __restrict__ u_new,
                                      const float* __restrict__ f, float h2,
                                      int n) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row > 0 && row < n - 1 && col > 0 && col < n - 1) {
    const size_t idx = static_cast<size_t>(row) * n + col;
    u_new[idx] = 0.25f * (
        u_old[static_cast<size_t>(row + 1) * n + col] +
        u_old[static_cast<size_t>(row - 1) * n + col] +
        u_old[static_cast<size_t>(row) * n + col + 1] +
        u_old[static_cast<size_t>(row) * n + col - 1] -
        h2 * f[idx]);
  }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

struct JacobiStats {
  double h2d_ms = 0.0;
  double kernel_ms = 0.0;
  double d2h_ms = 0.0;
  double total_ms = 0.0;
  int iterations = 0;
};

static std::vector<float> run_poisson_solver(int n, int max_iters,
                                             JacobiStats* stats) {
  if (n < 3) {
    throw std::invalid_argument("n must be at least 3");
  }

  const float h = 1.0f / static_cast<float>(n - 1);
  const float h2 = h * h;

  std::vector<float> u0(static_cast<size_t>(n) * n, 0.0f);
  std::vector<float> f(static_cast<size_t>(n) * n, 1.0f);

  float* d_old = nullptr;
  float* d_new = nullptr;
  float* d_f = nullptr;
  const size_t bytes = static_cast<size_t>(n) * n * sizeof(float);

  JacobiStats local;
  const auto h2d_t0 = Clock::now();
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_old), bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_new), bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_f), bytes));
  CUDA_CHECK(cudaMemcpy(d_old, u0.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_new, u0.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_f, f.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
  local.h2d_ms = elapsed_ms(h2d_t0);

  const dim3 block(BLOCK_X, BLOCK_Y);
  const dim3 grid((n + BLOCK_X - 1) / BLOCK_X, (n + BLOCK_Y - 1) / BLOCK_Y);

  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));
  CUDA_CHECK(cudaEventRecord(start));

  for (int it = 0; it < max_iters; ++it) {
    poisson_jacobi_kernel<<<grid, block>>>(d_old, d_new, d_f, h2, n);
    CUDA_CHECK(cudaGetLastError());
    std::swap(d_old, d_new);
  }
  local.iterations = max_iters;

  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));
  float kernel_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, end));
  local.kernel_ms = kernel_ms;

  std::vector<float> result(static_cast<size_t>(n) * n);
  const auto d2h_t0 = Clock::now();
  CUDA_CHECK(cudaMemcpy(result.data(), d_old, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());
  local.d2h_ms = elapsed_ms(d2h_t0);
  local.total_ms = local.h2d_ms + local.kernel_ms + local.d2h_ms;

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));
  CUDA_CHECK(cudaFree(d_old));
  CUDA_CHECK(cudaFree(d_new));
  CUDA_CHECK(cudaFree(d_f));

  if (stats != nullptr) {
    *stats = local;
  }
  return result;
}

static std::vector<double> cpu_jacobi_reference(int n, int iters) {
  const double h = 1.0 / (n - 1);
  const double h2 = h * h;
  std::vector<double> u(static_cast<size_t>(n) * n, 0.0);
  std::vector<double> un(static_cast<size_t>(n) * n, 0.0);

  for (int it = 0; it < iters; ++it) {
    for (int row = 1; row < n - 1; ++row) {
      for (int col = 1; col < n - 1; ++col) {
        const size_t idx = static_cast<size_t>(row) * n + col;
        un[idx] = 0.25 * (
            u[static_cast<size_t>(row + 1) * n + col] +
            u[static_cast<size_t>(row - 1) * n + col] +
            u[static_cast<size_t>(row) * n + col + 1] +
            u[static_cast<size_t>(row) * n + col - 1] -
            h2);
      }
    }
    std::swap(u, un);
  }
  return u;
}

int main() {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s\n", prop.name);

  // Warm-up (CUDA context, allocator, kernel caches)
  run_poisson_solver(64, 5, nullptr);

  // Numerical check against the CPU Jacobi reference
  const int n_check = 256;
  const int iters_check = 300;
  JacobiStats check_stats;
  const std::vector<float> gpu_u =
      run_poisson_solver(n_check, iters_check, &check_stats);
  const std::vector<double> cpu_u = cpu_jacobi_reference(n_check, iters_check);
  double max_err = 0.0;
  for (size_t i = 0; i < gpu_u.size(); ++i) {
    max_err = std::max(max_err, std::fabs(gpu_u[i] - cpu_u[i]));
  }
  std::printf("Ellenorzes N=%d, iters=%d: max |GPU-CPU| = %.3e\n", n_check,
              iters_check, max_err);

  std::printf("\nNaiv Jacobi benchmark\n");
  std::printf("%6s %6s %10s %10s %10s %10s\n", "N", "iter", "H2D", "kernel",
              "D2H", "total");
  for (int n : {256, 512, 1024}) {
    JacobiStats stats;
    run_poisson_solver(n, 500, &stats);
    std::printf("%6d %6d %9.2fms %9.2fms %9.2fms %9.2fms\n", n,
                stats.iterations, stats.h2d_ms, stats.kernel_ms,
                stats.d2h_ms, stats.total_ms);
  }

  return EXIT_SUCCESS;
}
