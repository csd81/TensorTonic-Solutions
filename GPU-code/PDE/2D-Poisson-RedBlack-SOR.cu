// 2D-Poisson-RedBlack-SOR.cu -- CUDA C++ port of 2D-Poisson-RedBlack-SOR.py.
//
// Red-Black SOR solver for the 2-D Poisson equation with zero Dirichlet
// boundaries, in-place on a single double-precision grid:
//   red kernel   -> update (i+j) even points from black neighbours,
//   black kernel -> update (i+j) odd points from the fresh red values,
//   omega = 2 / (1 + sin(pi/N)),
//   residual checked every check_every iterations (max |Laplacian - f|).
//
// Validation uses a sequential double-precision red-black SOR reference.
//
// Build:
//   nvcc -O3 -arch=native -o poisson-sor 2D-Poisson-RedBlack-SOR.cu
// Run:
//   ./poisson-sor

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#define BLOCK_X 32
#define BLOCK_Y 8

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

__global__ void sor_red_kernel(double* __restrict__ u,
                               const double* __restrict__ f, double h2,
                               double omega, int n) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row > 0 && row < n - 1 && col > 0 && col < n - 1 &&
      ((row + col) & 1) == 0) {
    const size_t idx = static_cast<size_t>(row) * n + col;
    const double gs = 0.25 * (
        u[static_cast<size_t>(row + 1) * n + col] +
        u[static_cast<size_t>(row - 1) * n + col] +
        u[static_cast<size_t>(row) * n + col + 1] +
        u[static_cast<size_t>(row) * n + col - 1] -
        h2 * f[idx]);
    u[idx] += omega * (gs - u[idx]);
  }
}

__global__ void sor_black_kernel(double* __restrict__ u,
                                 const double* __restrict__ f, double h2,
                                 double omega, int n) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row > 0 && row < n - 1 && col > 0 && col < n - 1 &&
      ((row + col) & 1) == 1) {
    const size_t idx = static_cast<size_t>(row) * n + col;
    const double gs = 0.25 * (
        u[static_cast<size_t>(row + 1) * n + col] +
        u[static_cast<size_t>(row - 1) * n + col] +
        u[static_cast<size_t>(row) * n + col + 1] +
        u[static_cast<size_t>(row) * n + col - 1] -
        h2 * f[idx]);
    u[idx] += omega * (gs - u[idx]);
  }
}

__global__ void residual_max_kernel(const double* __restrict__ u,
                                    const double* __restrict__ f, double h2,
                                    float* __restrict__ out, int n) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row > 0 && row < n - 1 && col > 0 && col < n - 1) {
    const size_t idx = static_cast<size_t>(row) * n + col;
    const double lap = (
        u[static_cast<size_t>(row + 1) * n + col] +
        u[static_cast<size_t>(row - 1) * n + col] +
        u[static_cast<size_t>(row) * n + col + 1] +
        u[static_cast<size_t>(row) * n + col - 1] -
        4.0 * u[idx]) / h2;
    const float diff = static_cast<float>(fabs(lap - f[idx]));
    // Positive floats preserve ordering when compared as unsigned integers.
    atomicMax(reinterpret_cast<unsigned int*>(out), __float_as_uint(diff));
  }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

struct SorStats {
  double h2d_ms = 0.0;
  double kernel_ms = 0.0;
  double d2h_ms = 0.0;
  double total_ms = 0.0;
  int iterations = 0;
  bool converged = false;
  double residual = 0.0;
  double omega = 0.0;
};

static std::vector<double> run_poisson_solver(int n, int max_iters, double tol,
                                              int check_every,
                                              SorStats* stats) {
  if (n < 3) {
    throw std::invalid_argument("n must be at least 3");
  }
  if (check_every < 1) {
    throw std::invalid_argument("check_every must be >= 1");
  }

  const double h = 1.0 / (n - 1);
  const double h2 = h * h;
  const double omega = 2.0 / (1.0 + std::sin(3.14159265358979323846 / n));

  std::vector<double> u_host(static_cast<size_t>(n) * n, 0.0);
  std::vector<double> f_host(static_cast<size_t>(n) * n, 1.0);

  double* d_u = nullptr;
  double* d_f = nullptr;
  float* d_res = nullptr;
  const size_t bytes = static_cast<size_t>(n) * n * sizeof(double);

  SorStats local;
  local.omega = omega;
  const auto h2d_t0 = Clock::now();
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_u), bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_f), bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_res), sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_u, u_host.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_f, f_host.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
  local.h2d_ms = elapsed_ms(h2d_t0);

  const dim3 block(BLOCK_X, BLOCK_Y);
  const dim3 grid((n + BLOCK_X - 1) / BLOCK_X, (n + BLOCK_Y - 1) / BLOCK_Y);

  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));
  CUDA_CHECK(cudaEventRecord(start));

  float h_res = 0.0f;
  for (int it = 0; it < max_iters; ++it) {
    sor_red_kernel<<<grid, block>>>(d_u, d_f, h2, omega, n);
    CUDA_CHECK(cudaGetLastError());
    sor_black_kernel<<<grid, block>>>(d_u, d_f, h2, omega, n);
    CUDA_CHECK(cudaGetLastError());
    local.iterations = it + 1;

    if (local.iterations % check_every == 0 || local.iterations == max_iters) {
      CUDA_CHECK(cudaMemset(d_res, 0, sizeof(float)));
      residual_max_kernel<<<grid, block>>>(d_u, d_f, h2, d_res, n);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(&h_res, d_res, sizeof(float),
                            cudaMemcpyDeviceToHost));
      local.residual = h_res;
      if (local.residual < tol) {
        local.converged = true;
        break;
      }
    }
  }

  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));
  float kernel_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, end));
  local.kernel_ms = kernel_ms;

  std::vector<double> result(static_cast<size_t>(n) * n);
  const auto d2h_t0 = Clock::now();
  CUDA_CHECK(cudaMemcpy(result.data(), d_u, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());
  local.d2h_ms = elapsed_ms(d2h_t0);
  local.total_ms = local.h2d_ms + local.kernel_ms + local.d2h_ms;

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));
  CUDA_CHECK(cudaFree(d_u));
  CUDA_CHECK(cudaFree(d_f));
  CUDA_CHECK(cudaFree(d_res));

  if (stats != nullptr) {
    *stats = local;
  }
  return result;
}

static double max_residual_host(const std::vector<double>& u, int n) {
  const double h = 1.0 / (n - 1);
  const double h2 = h * h;
  double result = 0.0;
  for (int row = 1; row < n - 1; ++row) {
    for (int col = 1; col < n - 1; ++col) {
      const size_t idx = static_cast<size_t>(row) * n + col;
      const double lap = (
          u[static_cast<size_t>(row + 1) * n + col] +
          u[static_cast<size_t>(row - 1) * n + col] +
          u[static_cast<size_t>(row) * n + col + 1] +
          u[static_cast<size_t>(row) * n + col - 1] -
          4.0 * u[idx]) / h2;
      result = std::max(result, std::fabs(lap - 1.0));
    }
  }
  return result;
}

static std::vector<double> cpu_sor_reference(int n, int max_iters,
                                             double tol, int* used_iters) {
  const double h = 1.0 / (n - 1);
  const double h2 = h * h;
  const double omega = 2.0 / (1.0 + std::sin(3.14159265358979323846 / n));

  std::vector<double> u(static_cast<size_t>(n) * n, 0.0);
  std::vector<double> f(static_cast<size_t>(n) * n, 1.0);

  int iters = 0;
  for (int it = 0; it < max_iters; ++it) {
    for (int color = 0; color < 2; ++color) {
      for (int row = 1; row < n - 1; ++row) {
        for (int col = 1; col < n - 1; ++col) {
          if (((row + col) & 1) != color) {
            continue;
          }
          const size_t idx = static_cast<size_t>(row) * n + col;
          const double gs = 0.25 * (
              u[static_cast<size_t>(row + 1) * n + col] +
              u[static_cast<size_t>(row - 1) * n + col] +
              u[static_cast<size_t>(row) * n + col + 1] +
              u[static_cast<size_t>(row) * n + col - 1] -
              h2 * f[idx]);
          u[idx] += omega * (gs - u[idx]);
        }
      }
    }
    iters = it + 1;
    if (iters % 20 == 0 && max_residual_host(u, n) < tol) {
      break;
    }
  }
  if (used_iters != nullptr) {
    *used_iters = iters;
  }
  return u;
}

int main() {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s\n", prop.name);

  // Warm-up (CUDA context, allocator, kernel caches)
  run_poisson_solver(64, 5, 1e-4, 20, nullptr);

  // Numerical check against the sequential SOR reference
  const int n_check = 256;
  SorStats check_stats;
  const std::vector<double> gpu_u =
      run_poisson_solver(n_check, 10000, 1e-5, 20, &check_stats);
  int cpu_iters = 0;
  const std::vector<double> cpu_u =
      cpu_sor_reference(n_check, 10000, 1e-9, &cpu_iters);
  double max_err = 0.0;
  for (size_t i = 0; i < gpu_u.size(); ++i) {
    max_err = std::max(max_err, std::fabs(gpu_u[i] - cpu_u[i]));
  }
  std::printf(
      "Ellenorzes N=%d: GPU iters=%d, res=%.3e | CPU iters=%d, res=%.3e | "
      "max |GPU-CPU|=%.3e\n",
      n_check, check_stats.iterations, check_stats.residual, cpu_iters,
      max_residual_host(cpu_u, n_check), max_err);

  std::printf("\nRed-Black SOR benchmark (optimalis omega)\n");
  std::printf("%6s %8s %7s %12s %10s %10s %6s\n", "N", "omega", "iter",
              "residual", "kernel", "total", "conv");
  for (int n : {256, 512, 1024}) {
    SorStats stats;
    run_poisson_solver(n, 20000, 1e-4, 20, &stats);
    std::printf("%6d %8.5f %7d %12.3e %9.2fms %9.2fms %6s\n", n,
                stats.omega, stats.iterations, stats.residual,
                stats.kernel_ms, stats.total_ms,
                stats.converged ? "true" : "false");
  }

  return EXIT_SUCCESS;
}
