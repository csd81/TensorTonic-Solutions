// conv2d.cu -- CUDA C++ port of conv2d.py (fixed version).
//
// Naive 2D FIR convolution, one thread per output pixel, zero padding.
// The filter is read reversed, so this implements true convolution:
//   y[row, col] = sum_{ky,kx} x[row + ky - K/2, col + kx - K/2]
//                              * h[K-1-ky, K-1-kx]
// matching scipy.ndimage.convolve(mode="constant", cval=0).
//
// Odd K is required by the reference convention (the benchmark uses K=5).
//
// Build:
//   nvcc -O3 -arch=native -o conv2d conv2d.cu
// (T4: -arch=sm_75, Quadro P2000: -arch=sm_61)
//
// Run:
//   ./conv2d                  # 4096 x 4096, K = 5
//   ./conv2d 1024 1024 7      # custom width height K

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

#define TPB_X 16
#define TPB_Y 16

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

__global__ void conv2d_global_kernel(const float* __restrict__ img_in,
                                     const float* __restrict__ kernel,
                                     float* __restrict__ img_out, int width,
                                     int height, int k_size) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int k_half = k_size / 2;

  if (col < width && row < height) {
    float acc = 0.0f;
    for (int ky = 0; ky < k_size; ++ky) {
      for (int kx = 0; kx < k_size; ++kx) {
        const int img_x = col + (kx - k_half);
        const int img_y = row + (ky - k_half);
        if (img_x >= 0 && img_x < width && img_y >= 0 && img_y < height) {
          // True convolution: reversed filter indices.
          acc += img_in[img_y * width + img_x] *
                 kernel[(k_size - 1 - ky) * k_size + (k_size - 1 - kx)];
        }
      }
    }
    img_out[row * width + col] = acc;
  }
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

static std::vector<float> cpu_conv2d_same(const std::vector<float>& img,
                                          const std::vector<float>& kernel,
                                          int width, int height,
                                          int k_size) {
  const int k_half = k_size / 2;
  std::vector<float> out(static_cast<size_t>(width) * height, 0.0f);

  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      double acc = 0.0;
      for (int ky = 0; ky < k_size; ++ky) {
        for (int kx = 0; kx < k_size; ++kx) {
          const int img_x = col + (kx - k_half);
          const int img_y = row + (ky - k_half);
          if (img_x >= 0 && img_x < width && img_y >= 0 && img_y < height) {
            acc += static_cast<double>(img[img_y * width + img_x]) *
                   static_cast<double>(
                       kernel[(k_size - 1 - ky) * k_size + (k_size - 1 - kx)]);
          }
        }
      }
      out[static_cast<size_t>(row) * width + col] = static_cast<float>(acc);
    }
  }
  return out;
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

// Symmetric binomial (Pascal row K-1) outer-product filter, normalized.
// For K=5 this is the [1,4,6,4,1]^T [1,4,6,4,1] kernel used in Python.
static std::vector<float> make_binomial_filter(int k_size) {
  std::vector<double> b(k_size);
  b[0] = 1.0;
  for (int i = 1; i < k_size; ++i) {
    b[i] = b[i - 1] * static_cast<double>(k_size - i) /
           static_cast<double>(i);
  }

  std::vector<double> tmp(static_cast<size_t>(k_size) * k_size);
  double total = 0.0;
  for (int y = 0; y < k_size; ++y) {
    for (int x = 0; x < k_size; ++x) {
      const double value = b[y] * b[x];
      tmp[static_cast<size_t>(y) * k_size + x] = value;
      total += value;
    }
  }

  std::vector<float> kernel(tmp.size());
  for (size_t i = 0; i < tmp.size(); ++i) {
    kernel[i] = static_cast<float>(tmp[i] / total);
  }
  return kernel;
}

// ---------------------------------------------------------------------------
// Demo / benchmark
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  const int width = (argc > 1) ? std::atoi(argv[1]) : 4096;
  const int height = (argc > 2) ? std::atoi(argv[2]) : 4096;
  const int k_size = (argc > 3) ? std::atoi(argv[3]) : 5;

  if (width < 1 || height < 1) {
    std::fprintf(stderr, "width and height must be >= 1\n");
    return EXIT_FAILURE;
  }
  if (k_size < 1 || (k_size % 2) == 0) {
    std::fprintf(stderr, "k_size must be an odd positive number\n");
    return EXIT_FAILURE;
  }

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  const dim3 block(TPB_X, TPB_Y);
  const dim3 grid((width + TPB_X - 1) / TPB_X,
                  (height + TPB_Y - 1) / TPB_Y);

  std::printf("GPU: %s\n", prop.name);
  std::printf("--- 2D Kepi Konvolucios Benchmark (CUDA C++) ---\n");
  std::printf("Kep merete: %d x %d pixel (%.1f MP)\n", width, height,
              static_cast<double>(width) * height / 1e6);
  std::printf("Szuromag merete: %dx%d\n", k_size, k_size);
  std::printf("Grid: (%u, %u) blokk x (%u, %u) szal\n\n", grid.x, grid.y,
              block.x, block.y);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> uniform(0.0f, 255.0f);
  std::vector<float> image(static_cast<size_t>(width) * height);
  for (float& value : image) {
    value = uniform(gen);
  }
  const std::vector<float> kernel = make_binomial_filter(k_size);

  // ---------------------------------------------------------------------
  // Small asymmetric-filter check (a symmetric filter would hide
  // convolution-vs-correlation mistakes).
  // ---------------------------------------------------------------------
  std::vector<float> test_img(25);
  std::iota(test_img.begin(), test_img.end(), 0.0f);
  std::vector<float> test_kernel(9);
  std::iota(test_kernel.begin(), test_kernel.end(), 0.0f);

  const std::vector<float> test_ref =
      cpu_conv2d_same(test_img, test_kernel, 5, 5, 3);

  float* d_test_img = nullptr;
  float* d_test_kernel = nullptr;
  float* d_test_out = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_test_img),
                        25 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_test_kernel),
                        9 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_test_out),
                        25 * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_test_img, test_img.data(), 25 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_test_kernel, test_kernel.data(), 9 * sizeof(float),
                        cudaMemcpyHostToDevice));
  conv2d_global_kernel<<<1, block>>>(d_test_img, d_test_kernel, d_test_out, 5,
                                     5, 3);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> test_out(25);
  CUDA_CHECK(cudaMemcpy(test_out.data(), d_test_out, 25 * sizeof(float),
                        cudaMemcpyDeviceToHost));
  const float test_err = max_abs_diff(test_ref, test_out);
  std::printf("Aszimmetrikus 3x3 konvolucio-ellenorzes: max hiba = %.2e (%s)\n\n",
              test_err, test_err < 1e-5f ? "OK" : "HIBA");
  CUDA_CHECK(cudaFree(d_test_img));
  CUDA_CHECK(cudaFree(d_test_kernel));
  CUDA_CHECK(cudaFree(d_test_out));

  // ---------------------------------------------------------------------
  // CPU reference (median of 3)
  // ---------------------------------------------------------------------
  std::vector<double> cpu_times;
  std::vector<float> image_cpu;
  for (int r = 0; r < 3; ++r) {
    const auto t0 = Clock::now();
    image_cpu = cpu_conv2d_same(image, kernel, width, height, k_size);
    cpu_times.push_back(elapsed_ms(t0));
  }
  std::sort(cpu_times.begin(), cpu_times.end());
  const double cpu_ms = cpu_times[1];
  std::printf("CPU referencia: %.2f ms (3 futas medianja)\n", cpu_ms);

  // ---------------------------------------------------------------------
  // Host -> Device
  // ---------------------------------------------------------------------
  const size_t image_bytes = static_cast<size_t>(width) * height * sizeof(float);
  const size_t kernel_bytes = static_cast<size_t>(k_size) * k_size * sizeof(float);

  float* d_image_in = nullptr;
  float* d_kernel = nullptr;
  float* d_image_out = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_image_in), image_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_kernel), kernel_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_image_out), image_bytes));

  const auto h2d_t0 = Clock::now();
  CUDA_CHECK(cudaMemcpy(d_image_in, image.data(), image_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_kernel, kernel.data(), kernel_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
  const double h2d_ms = elapsed_ms(h2d_t0);
  std::printf("Host -> Device masolas: %.2f ms\n", h2d_ms);

  const auto launch = [&]() {
    conv2d_global_kernel<<<grid, block>>>(d_image_in, d_kernel, d_image_out,
                                          width, height, k_size);
    CUDA_CHECK(cudaGetLastError());
  };

  const double gpu_ms = median_kernel_ms(launch);

  std::vector<float> image_gpu(static_cast<size_t>(width) * height);
  const auto d2h_t0 = Clock::now();
  CUDA_CHECK(cudaMemcpy(image_gpu.data(), d_image_out, image_bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());
  const double d2h_ms = elapsed_ms(d2h_t0);

  const float max_err = max_abs_diff(image_cpu, image_gpu);

  std::printf("GPU futasido:   %.2f ms (20 futas medianja)\n", gpu_ms);
  std::printf("Device -> Host masolas: %.2f ms\n", d2h_ms);
  std::printf("Max abszolut hiba: %.3e\n", max_err);
  std::printf("Numerikus helyesseg: %s\n", max_err < 1e-3f ? "OK" : "HIBA");
  std::printf("Gyorsulas (Speedup): %.1fx\n", cpu_ms / gpu_ms);

  CUDA_CHECK(cudaFree(d_image_in));
  CUDA_CHECK(cudaFree(d_kernel));
  CUDA_CHECK(cudaFree(d_image_out));

  return EXIT_SUCCESS;
}
