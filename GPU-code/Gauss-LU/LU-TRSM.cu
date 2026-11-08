// LU-TRSM.cu -- CUDA C++ port of LU-TRSM.py.
//
// Blocked LU factorization without pivoting, TILE = 16:
//   1. diagonal tile factorization in shared memory,
//   2. L panel via right triangular solve  L_ik * U_kk = A_ik,
//   3. U panel via left triangular solve   L_kk * U_kj = A_kj,
//   4. trailing GEMM update                A_ij -= L_ik * U_kj.
// L (unit lower) and U (upper) are stored in place in A.
// The test uses a diagonally dominant matrix (no pivoting).
//
// Build:
//   nvcc -O3 -arch=native -o LU-TRSM LU-TRSM.cu
// Run:
//   ./LU-TRSM [n]        # default n = 256

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define TILE 16
#define TILE_PAD (TILE + 1)
#define TPB_1D 256

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

__global__ void diagonal_factorize_kernel(float* A, int k0, int t, int n) {
  __shared__ float s_A[TILE][TILE_PAD];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = k0 + ty;
  const int col = k0 + tx;

  if (row < n && col < n) {
    s_A[ty][tx] = A[static_cast<size_t>(row) * n + col];
  } else {
    s_A[ty][tx] = 0.0f;
  }
  __syncthreads();

  for (int k = 0; k < t; ++k) {
    // Normalize the pivot column (multipliers)
    if (ty > k && tx == k) {
      s_A[ty][k] /= s_A[k][k];
    }
    __syncthreads();

    // Update the inner submatrix
    if (ty > k && tx > k) {
      s_A[ty][tx] -= s_A[ty][k] * s_A[k][tx];
    }
    __syncthreads();
  }

  if (row < n && col < n) {
    A[static_cast<size_t>(row) * n + col] = s_A[ty][tx];
  }
}

__global__ void panel_l_trsm_kernel(float* A, int k0, int t, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x + k0 + t;
  if (i < n) {
    for (int j = 0; j < t; ++j) {
      double s = 0.0;
      for (int p = 0; p < j; ++p) {
        s += static_cast<double>(A[static_cast<size_t>(i) * n + k0 + p]) *
             static_cast<double>(A[static_cast<size_t>(k0 + p) * n + k0 + j]);
      }
      A[static_cast<size_t>(i) * n + k0 + j] = static_cast<float>(
          (static_cast<double>(A[static_cast<size_t>(i) * n + k0 + j]) - s) /
          static_cast<double>(A[static_cast<size_t>(k0 + j) * n + k0 + j]));
    }
  }
}

__global__ void panel_u_trsm_kernel(float* A, int k0, int t, int n) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x + k0 + t;
  if (j < n) {
    for (int i = 0; i < t; ++i) {
      double s = 0.0;
      for (int p = 0; p < i; ++p) {
        s += static_cast<double>(A[static_cast<size_t>(k0 + i) * n + k0 + p]) *
             static_cast<double>(A[static_cast<size_t>(k0 + p) * n + j]);
      }
      A[static_cast<size_t>(k0 + i) * n + j] -= static_cast<float>(s);
    }
  }
}

__global__ void blocked_lu_gemm_kernel(float* A, int k0, int n) {
  __shared__ float s_L[TILE][TILE_PAD];
  __shared__ float s_U[TILE][TILE_PAD];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = (blockIdx.y + 1) * TILE + ty + k0;
  const int col = (blockIdx.x + 1) * TILE + tx + k0;

  if (row < n && (k0 + tx) < n) {
    s_L[ty][tx] = A[static_cast<size_t>(row) * n + k0 + tx];
  } else {
    s_L[ty][tx] = 0.0f;
  }

  if ((k0 + ty) < n && col < n) {
    s_U[ty][tx] = A[static_cast<size_t>(k0 + ty) * n + col];
  } else {
    s_U[ty][tx] = 0.0f;
  }
  __syncthreads();

  if (row < n && col < n) {
    float acc = 0.0f;
    for (int p = 0; p < TILE; ++p) {
      acc += s_L[ty][p] * s_U[p][tx];
    }
    A[static_cast<size_t>(row) * n + col] -= acc;
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

static std::vector<float> run_blocked_lu(const std::vector<float>& A0, int n,
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

  const dim3 block(TILE, TILE);
  for (int k0 = 0; k0 < n; k0 += TILE) {
    const int t = std::min(TILE, n - k0);

    diagonal_factorize_kernel<<<1, block>>>(d_A, k0, t, n);
    CUDA_CHECK(cudaGetLastError());

    if (k0 + t < n) {
      const int rem = n - k0 - t;
      const int panel_blocks = (rem + TPB_1D - 1) / TPB_1D;

      panel_l_trsm_kernel<<<panel_blocks, TPB_1D>>>(d_A, k0, t, n);
      CUDA_CHECK(cudaGetLastError());
      panel_u_trsm_kernel<<<panel_blocks, TPB_1D>>>(d_A, k0, t, n);
      CUDA_CHECK(cudaGetLastError());

      const int tiles = (rem + TILE - 1) / TILE;
      const dim3 grid(tiles, tiles);
      blocked_lu_gemm_kernel<<<grid, block>>>(d_A, k0, n);
      CUDA_CHECK(cudaGetLastError());
    }
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

  std::mt19937 gen(43);
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
  std::vector<float> A0(static_cast<size_t>(n) * n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      A0[static_cast<size_t>(i) * n + j] = uniform(gen);
    }
    A0[static_cast<size_t>(i) * n + i] += static_cast<float>(n);
  }

  // Warm-up (CUDA context, allocator, caches); not timed.
  const int warm = std::min(n, 32);
  std::vector<float> A_warm(static_cast<size_t>(warm) * warm);
  for (int i = 0; i < warm; ++i) {
    for (int j = 0; j < warm; ++j) {
      A_warm[static_cast<size_t>(i) * warm + j] =
          A0[static_cast<size_t>(i) * n + j];
    }
  }
  run_blocked_lu(A_warm, warm, nullptr);

  LuStats stats;
  const std::vector<float> A_lu = run_blocked_lu(A0, n, &stats);
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

  std::printf("Blokkolt LU (TRSM), n=%d, TILE=%d\n", n, TILE);
  std::printf("  GPU H2D:     %.2f ms\n", stats.h2d_ms);
  std::printf("  GPU kernel:  %.2f ms\n", stats.kernel_ms);
  std::printf("  GPU D2H:     %.2f ms\n", stats.d2h_ms);
  std::printf("  GPU total:   %.2f ms\n", stats.total_ms);
  std::printf("  CPU naiv:    %.2f ms (median/3)\n", cpu_ms);
  std::printf("  Rel. hiba ||A-LU||_F/||A||_F: %.3e\n", rel_err);
  std::printf("  CPU checksum: %.6e\n", cpu_checksum);

  return EXIT_SUCCESS;
}
