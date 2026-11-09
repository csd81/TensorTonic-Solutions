// RMS normalization backward (challenge 142).
//
// For each row i: rstd_i = 1/sqrt(mean(x^2) + eps), xhat = x*rstd,
// g = dy*weight,
//   dx       = rstd * (g - xhat * mean_j(g*xhat))
//   dweight_j = sum_i dy_ij * xhat_ij
// Benchmark: N = 16,384, C = 4,096 -> x, dy and dx are 268 MB each; one read
// of x and dy plus one write of dx is 804 MB, ~9.8 ms at this GPU's ~82 GB/s.
//
// The starter measures 21.7 ms: it reads x three times and dy twice (1.6 GB)
// and performs one float atomicAdd per element for dweight (67 M atomics,
// which also makes its dweight accumulate ~1e-2 of error through
// cancellation over 16 K rows).
//
// This version:
//
//  1. Keeps the current row of x, dy and weight in registers (COLS columns
//     per thread, COLS = ceil(C/256) <= 16 for C <= 4096); the row is read
//     once, dx is written once, and nothing is re-read.
//  2. Processes R = 16 consecutive rows per block, accumulating dweight's
//     contribution for the block's columns in registers; only C doubles are
//     atomically added per block (4.2 M double atomics instead of 67 M
//     float ones), which also removes the cancellation error of the
//     element-wise atomic accumulation.
//  3. Reduces sumsq and g*xhat with warp shuffles plus a shared combine
//     (with a barrier after the broadcast read, since the reductions repeat
//     for every row).
//
// A generic fallback kernel handles C > 4096 (correctness only; the
// benchmark is C = 4096).
//
// Measured: see README (~10 ms, ~2.1x).

#include <cuda_runtime.h>
#include <math.h>

namespace {

constexpr int TPB = 256;
constexpr int R = 16;

__device__ __forceinline__ float block_reduce_bcast(float v, float* scratch,
                                                    int nwarp) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  if ((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = v;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.0f;
    for (int w = 0; w < nwarp; ++w) t += scratch[w];
    scratch[0] = t;
  }
  __syncthreads();
  const float result = scratch[0];
  __syncthreads();  // the caller may run another reduction immediately
  return result;
}

template <int COLS>
__global__ void __launch_bounds__(TPB)
rms_bwd_kernel(const float* __restrict__ x, const float* __restrict__ weight,
               const float* __restrict__ dy, float* __restrict__ dx,
               double* __restrict__ dweight, int N, int C, float eps) {
  __shared__ float scratch[8];  // one slot per warp
  const int nwarp = TPB >> 5;
  const int col = threadIdx.x;

  float wv[COLS];
#pragma unroll
  for (int c = 0; c < COLS; ++c) {
    const int j = col + c * TPB;
    wv[c] = (j < C) ? weight[j] : 0.0f;
  }
  float dw[COLS];
#pragma unroll
  for (int c = 0; c < COLS; ++c) dw[c] = 0.0f;

  const int row0 = blockIdx.x * R;
  const int row_end = min(N, row0 + R);
  for (int row = row0; row < row_end; ++row) {
    const float* xr = x + (size_t)row * C;
    const float* yr = dy + (size_t)row * C;
    float xv[COLS], yv[COLS];
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      const int j = col + c * TPB;
      if (j < C) {
        xv[c] = xr[j];
        yv[c] = yr[j];
      } else {
        xv[c] = 0.0f;
        yv[c] = 0.0f;
      }
    }

    float ss = 0.0f;
#pragma unroll
    for (int c = 0; c < COLS; ++c) ss = fmaf(xv[c], xv[c], ss);
    const float rstd = rsqrtf(block_reduce_bcast(ss, scratch, nwarp) / (float)C + eps);

    float dot = 0.0f;
#pragma unroll
    for (int c = 0; c < COLS; ++c) dot = fmaf(yv[c] * wv[c], xv[c] * rstd, dot);
    const float mean = block_reduce_bcast(dot, scratch, nwarp) / (float)C;

    float* dxr = dx + (size_t)row * C;
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      const int j = col + c * TPB;
      if (j < C) {
        const float xh = xv[c] * rstd;
        dxr[j] = rstd * (yv[c] * wv[c] - xh * mean);
        dw[c] = fmaf(yv[c], xh, dw[c]);
      }
    }
  }

  // One double atomic per column per block (R rows' worth of contributions).
#pragma unroll
  for (int c = 0; c < COLS; ++c) {
    const int j = col + c * TPB;
    if (j < C) atomicAdd(&dweight[j], (double)dw[c]);
  }
}

// Fallback for C > 4096: one block per row, rows re-read (L1), float partial
// sums reduced per block; dweight accumulated in double per block.
__global__ void rms_bwd_generic_kernel(const float* __restrict__ x,
                                       const float* __restrict__ weight,
                                       const float* __restrict__ dy,
                                       float* __restrict__ dx,
                                       double* __restrict__ dweight, int N,
                                       int C, float eps) {
  __shared__ float scratch[32];
  const int nwarp = (blockDim.x + 31) >> 5;
  const int row = blockIdx.x;
  const float* xr = x + (size_t)row * C;
  const float* yr = dy + (size_t)row * C;
  float* dxr = dx + (size_t)row * C;

  float ss = 0.0f;
  for (int j = threadIdx.x; j < C; j += blockDim.x) ss = fmaf(xr[j], xr[j], ss);
  const float rstd = rsqrtf(block_reduce_bcast(ss, scratch, nwarp) / (float)C + eps);

  float dot = 0.0f;
  for (int j = threadIdx.x; j < C; j += blockDim.x)
    dot = fmaf(yr[j] * weight[j], xr[j] * rstd, dot);
  const float mean = block_reduce_bcast(dot, scratch, nwarp) / (float)C;

  for (int j = threadIdx.x; j < C; j += blockDim.x) {
    const float xh = xr[j] * rstd;
    dxr[j] = rstd * (yr[j] * weight[j] - xh * mean);
    atomicAdd(&dweight[j], (double)(yr[j] * xh));
  }
}

// dweight accumulates in double (an element-wise float atomic sum over 16 K
// rows loses ~1e-2 absolute to cancellation), then converts to float.
__global__ void convert_kernel(const double* __restrict__ src,
                               float* __restrict__ dst, int C) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < C) dst[j] = (float)src[j];
}

double* g_dw = nullptr;
int g_dw_cap = 0;

}  // namespace

extern "C" void solve(const float* x, const float* weight, const float* dy,
                      float* dx, float* dweight, int N, int C, float eps) {
  if (N <= 0 || C <= 0) return;
  if (C > g_dw_cap) {
    if (g_dw) cudaFree(g_dw);
    cudaMalloc(&g_dw, (size_t)C * sizeof(double));
    g_dw_cap = C;
  }
  cudaMemsetAsync(g_dw, 0, (size_t)C * sizeof(double));

  const int row_blocks = (N + R - 1) / R;
  if (C <= 4096) {
    const int cols = (C + TPB - 1) / TPB;  // 1..16
    if (cols <= 1) {
      rms_bwd_kernel<1><<<row_blocks, TPB>>>(x, weight, dy, dx, g_dw, N, C, eps);
    } else if (cols <= 2) {
      rms_bwd_kernel<2><<<row_blocks, TPB>>>(x, weight, dy, dx, g_dw, N, C, eps);
    } else if (cols <= 4) {
      rms_bwd_kernel<4><<<row_blocks, TPB>>>(x, weight, dy, dx, g_dw, N, C, eps);
    } else if (cols <= 8) {
      rms_bwd_kernel<8><<<row_blocks, TPB>>>(x, weight, dy, dx, g_dw, N, C, eps);
    } else {
      rms_bwd_kernel<16><<<row_blocks, TPB>>>(x, weight, dy, dx, g_dw, N, C, eps);
    }
  } else {
    rms_bwd_generic_kernel<<<N, 256>>>(x, weight, dy, dx, g_dw, N, C, eps);
  }
  convert_kernel<<<(C + 255) / 256, 256>>>(g_dw, dweight, C);
}
