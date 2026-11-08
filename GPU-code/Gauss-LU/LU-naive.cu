// LU-naive.cu -- CUDA C++ port of LU-naive.py.
//
// Unblocked LU factorization without pivoting, stored in place:
//   L = strictly lower triangle with unit diagonal, U = upper triangle.
// Per step k:
//   A[i,k] /= A[k,k]                 (i > k)
//   A[i,j] -= A[i,k] * A[k,j]        (i,j > k)
// The host loop launches one 1-D scale kernel and one 2-D update kernel
// per step. The test uses a diagonally dominant matrix (no pivoting).
//
// Build:
//   nvcc -O3 -arch=native -o LU-naive LU-naive.cu
// Run:
//   ./LU-naive [n]        # default n = 256

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define TPB_1D 256
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
// Kernels
// ---------------------------------------------------------------------------

__global__ void lu_scale_kernel(float* A, int k, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x + k + 1;
  if (i < n) {
    A[static_cast<size_t>(i) * n + k] /= A[static_cast<size_t>(k) * n + k];
  }
}

__global__ void lu_update_kernel(float* A, int k, int n) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x + k + 1;
  const int row = blockIdx.y * blockDim.y + threadIdx.y + k + 1;
  if (row < n && col < n) {
    A[static_cast<size_t>(row) * n + col] -=
        A[static_cast<size_t>(row) * n + k] *
        A[static_cast<size_t>(k) * n + col];
  }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

struct LuStats {
  double h2d_ms = 0.0;
  double kernel_ms = 0.0;
  double d2h_ms = 0.0;
  double total_ms = 0.0;
};

// Runs naive LU on the device; returns the in-place L/U packed matrix.
static std::vector<float> run_naive_lu(const std::vector<float>& A0, int n,
                                       LuStats* stats) {
  const size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
  float* d_A = nullptr;

  LuStats local;
  const auto h2d_t0 = Clock::now();
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), bytes));
  CUDA_CHECK(cudaMemcpy(d_A, A0.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
  local.h2d_ms = elapsed_ms(h2d_t0);

  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));
  CUDA_CHECK(cudaEventRecord(start));

  const dim3 block(BLOCK_X, BLOCK_Y);
  for (int k = 0; k < n; ++k) {
    const int remaining = n - k - 1;
    if (remaining <= 0) {
      continue;
    }

    const int scale_blocks = (remaining + TPB_1D - 1) / TPB_1D;
    lu_scale_kernel<<<scale_blocks, TPB_1D>>>(d_A, k, n);
    CUDA_CHECK(cudaGetLastError());

    const dim3 grid((remaining + BLOCK_X - 1) / BLOCK_X,
                    (remaining + BLOCK_Y - 1) / BLOCK_Y);
    lu_update_kernel<<<grid, block>>>(d_A, k, n);
    CUDA_CHECK(cudaGetLastError());
  }

  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));
  float kernel_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, end));
  local.kernel_ms = kernel_ms;

  const auto d2h_t0 = Clock::now();
  std::vector<float> A_out(static_cast<size_t>(n) * n);
  CUDA_CHECK(cudaMemcpy(A_out.data(), d_A, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());
  local.d2h_ms = elapsed_ms(d2h_t0);
  local.total_ms = local.h2d_ms + local.kernel_ms + local.d2h_ms;

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));
  CUDA_CHECK(cudaFree(d_A));

  if (stats != nullptr) {
    *stats = local;
  }
  return A_out;
}

// ||A0 - L U||_F / ||A0||_F for the packed in-place LU matrix.
static double lu_relative_error(const std::vector<float>& A0,
                                const std::vector<float>& A_lu, int n) {
  double diff2 = 0.0;
  double norm2 = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double lu = 0.0;
      for (int k = 0; k < n; ++k) {
        const double l =
            (i == k) ? 1.0
                     : (i > k ? static_cast<double>(A_lu[static_cast<size_t>(i) * n + k])
                              : 0.0);
        const double u =
            (k <= j) ? static_cast<double>(A_lu[static_cast<size_t>(k) * n + j])
                     : 0.0;
        lu += l * u;
      }
      const double a = static_cast<double>(A0[static_cast<size_t>(i) * n + j]);
      diff2 += (a - lu) * (a - lu);
      norm2 += a * a;
    }
  }
  return std::sqrt(diff2 / norm2);
}

static void cpu_naive_lu(std::vector<double>& A, int n) {
  for (int k = 0; k < n; ++k) {
    const double pivot = A[static_cast<size_t>(k) * n + k];
    for (int i = k + 1; i < n; ++i) {
      A[static_cast<size_t>(i) * n + k] /= pivot;
    }
    for (int i = k + 1; i < n; ++i) {
      const double lik = A[static_cast<size_t>(i) * n + k];
      for (int j = k + 1; j < n; ++j) {
        A[static_cast<size_t>(i) * n + j] -=
            lik * A[static_cast<size_t>(k) * n + j];
      }
    }
  }
}

int main(int argc, char** argv) {
  const int n = (argc > 1) ? std::atoi(argv[1]) : 256;
  if (n < 1) {
    std::fprintf(stderr, "n must be >= 1\n");
    return EXIT_FAILURE;
  }

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s\n", prop.name);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
  std::vector<float> A0(static_cast<size_t>(n) * n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      A0[static_cast<size_t>(i) * n + j] = uniform(gen);
    }
    A0[static_cast<size_t>(i) * n + i] += static_cast<float>(n);
  }

  // Warm-up (CUDA context, allocator, caches); not timed.
  const int warm = std::min(n, 16);
  std::vector<float> A_warm(static_cast<size_t>(warm) * warm);
  for (int i = 0; i < warm; ++i) {
    for (int j = 0; j < warm; ++j) {
      A_warm[static_cast<size_t>(i) * warm + j] =
          A0[static_cast<size_t>(i) * n + j];
    }
  }
  run_naive_lu(A_warm, warm, nullptr);

  LuStats stats;
  const std::vector<float> A_lu = run_naive_lu(A0, n, &stats);
  const double rel_err = lu_relative_error(A0, A_lu, n);

  std::vector<double> cpu_times;
  double cpu_checksum = 0.0;
  for (int r = 0; r < 3; ++r) {
    std::vector<double> A_cpu(A0.begin(), A0.end());
    const auto t0 = Clock::now();
    cpu_naive_lu(A_cpu, n);
    cpu_times.push_back(elapsed_ms(t0));
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
      sum += A_cpu[static_cast<size_t>(i) * n + i];
    }
    cpu_checksum += sum;
  }
  std::sort(cpu_times.begin(), cpu_times.end());
  const double cpu_ms = cpu_times[1];

  std::printf("Naiv LU, n=%d\n", n);
  std::printf("  GPU H2D:     %.2f ms\n", stats.h2d_ms);
  std::printf("  GPU kernel:  %.2f ms\n", stats.kernel_ms);
  std::printf("  GPU D2H:     %.2f ms\n", stats.d2h_ms);
  std::printf("  GPU total:   %.2f ms\n", stats.total_ms);
  std::printf("  CPU naiv:    %.2f ms (median/3)\n", cpu_ms);
  std::printf("  Rel. hiba ||A-LU||_F/||A||_F: %.3e\n", rel_err);
  std::printf("  CPU checksum: %.6e\n", cpu_checksum);

  return EXIT_SUCCESS;
}
