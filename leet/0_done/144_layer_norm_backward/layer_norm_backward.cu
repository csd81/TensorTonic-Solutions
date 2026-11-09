// Layer normalization backward (challenge 144).
//
// Given the forward pass's per-row mean and rstd, with
// xhat = (x - mean)*rstd and g = dy*gamma:
//   dx      = rstd * (g - mean_j(g) - xhat * mean_j(g*xhat))
//   dgamma_j = sum_i dy_ij * xhat_ij
//   dbeta_j  = sum_i dy_ij
// Benchmark: N = 16,384, C = 4,096 -> x, dy and dx are 268 MB each; one read
// of x and dy plus one write of dx is 804 MB, ~9.8 ms at this GPU's ~82 GB/s.
//
// The starter measures 24.5 ms: its dx kernel reads x and dy twice, and its
// parameter kernel re-reads x and dy once more with ONE THREAD PER COLUMN
// summing all 16 K rows in a serial float32 loop -- which also leaves up to
// ~1.5e-2 of absolute error on dgamma/dbeta through accumulation error.
//
// This version:
//
//  1. Keeps the current row of x, dy and weight in registers (COLS columns
//     per thread, COLS = ceil(C/256) <= 16 for C <= 4096); the row is read
//     once, dx is written once, and nothing is re-read.
//  2. Processes R = 16 consecutive rows per block, accumulating dgamma and
//     dbeta for the block's columns in registers; only C doubles are
//     atomically added per block per parameter (4.2 M double atomics), which
//     removes the starter's serial-float accumulation error.
//  3. Reduces sum(g) and sum(g*xhat) with warp shuffles plus a shared
//     combine (with a barrier after the broadcast read, since the reductions
//     repeat for every row).
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
ln_bwd_kernel(const float* __restrict__ x, const float* __restrict__ gamma,
              const float* __restrict__ mean, const float* __restrict__ rstd,
              const float* __restrict__ dy, float* __restrict__ dx,
              double* __restrict__ dgamma, double* __restrict__ dbeta,
              int N, int C) {
  __shared__ float scratch[8];  // one slot per warp
  const int nwarp = TPB >> 5;
  const int col = threadIdx.x;

  float wv[COLS];
#pragma unroll
  for (int c = 0; c < COLS; ++c) {
    const int j = col + c * TPB;
    wv[c] = (j < C) ? gamma[j] : 0.0f;
  }
  float dg[COLS], db[COLS];
#pragma unroll
  for (int c = 0; c < COLS; ++c) {
    dg[c] = 0.0f;
    db[c] = 0.0f;
  }

  const int row0 = blockIdx.x * R;
  const int row_end = min(N, row0 + R);
  for (int row = row0; row < row_end; ++row) {
    const float mu = mean[row];
    const float r = rstd[row];
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

    float sg = 0.0f, sgx = 0.0f;
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      const float xh = (xv[c] - mu) * r;
      const float g = yv[c] * wv[c];
      sg += g;
      sgx = fmaf(g, xh, sgx);
    }
    // Two reductions; the helper's trailing barrier makes them safe back to
    // back.
    const float sum_g = block_reduce_bcast(sg, scratch, nwarp);
    const float sum_gx = block_reduce_bcast(sgx, scratch, nwarp);
    const float invC = 1.0f / (float)C;

    float* dxr = dx + (size_t)row * C;
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      const int j = col + c * TPB;
      if (j < C) {
        const float xh = (xv[c] - mu) * r;
        const float g = yv[c] * wv[c];
        dxr[j] = r * (g - invC * sum_g - xh * invC * sum_gx);
        dg[c] = fmaf(yv[c], xh, dg[c]);
        db[c] += yv[c];
      }
    }
  }

  // One double atomic per column per parameter per block.
#pragma unroll
  for (int c = 0; c < COLS; ++c) {
    const int j = col + c * TPB;
    if (j < C) {
      atomicAdd(&dgamma[j], (double)dg[c]);
      atomicAdd(&dbeta[j], (double)db[c]);
    }
  }
}

// Fallback for C > 4096: one block per row, rows re-read (L1).
__global__ void ln_bwd_generic_kernel(const float* __restrict__ x,
                                      const float* __restrict__ gamma,
                                      const float* __restrict__ mean,
                                      const float* __restrict__ rstd,
                                      const float* __restrict__ dy,
                                      float* __restrict__ dx,
                                      double* __restrict__ dgamma,
                                      double* __restrict__ dbeta, int N, int C) {
  __shared__ float scratch[32];
  const int nwarp = (blockDim.x + 31) >> 5;
  const int row = blockIdx.x;
  const float mu = mean[row], r = rstd[row];
  const float* xr = x + (size_t)row * C;
  const float* yr = dy + (size_t)row * C;
  float* dxr = dx + (size_t)row * C;

  float sg = 0.0f, sgx = 0.0f;
  for (int j = threadIdx.x; j < C; j += blockDim.x) {
    const float xh = (xr[j] - mu) * r;
    const float g = yr[j] * gamma[j];
    sg += g;
    sgx = fmaf(g, xh, sgx);
  }
  const float sum_g = block_reduce_bcast(sg, scratch, nwarp);
  const float sum_gx = block_reduce_bcast(sgx, scratch, nwarp);
  const float invC = 1.0f / (float)C;

  for (int j = threadIdx.x; j < C; j += blockDim.x) {
    const float xh = (xr[j] - mu) * r;
    const float g = yr[j] * gamma[j];
    dxr[j] = r * (g - invC * sum_g - xh * invC * sum_gx);
    atomicAdd(&dgamma[j], (double)(yr[j] * xh));
    atomicAdd(&dbeta[j], (double)yr[j]);
  }
}

// dgamma/dbeta accumulate in double, then convert to float.
__global__ void convert2_kernel(const double* __restrict__ s0,
                                const double* __restrict__ s1,
                                float* __restrict__ d0,
                                float* __restrict__ d1, int C) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < C) {
    d0[j] = (float)s0[j];
    d1[j] = (float)s1[j];
  }
}

double* g_dg = nullptr;
double* g_db = nullptr;
int g_dw_cap = 0;

}  // namespace

extern "C" void solve(const float* x, const float* gamma, const float* mean,
                      const float* rstd, const float* dy, float* dx,
                      float* dgamma, float* dbeta, int N, int C) {
  if (N <= 0 || C <= 0) return;
  if (C > g_dw_cap) {
    if (g_dg) {
      cudaFree(g_dg);
      cudaFree(g_db);
    }
    cudaMalloc(&g_dg, (size_t)C * sizeof(double));
    cudaMalloc(&g_db, (size_t)C * sizeof(double));
    g_dw_cap = C;
  }
  cudaMemsetAsync(g_dg, 0, (size_t)C * sizeof(double));
  cudaMemsetAsync(g_db, 0, (size_t)C * sizeof(double));

  const int row_blocks = (N + R - 1) / R;
  if (C <= 4096) {
    const int cols = (C + TPB - 1) / TPB;  // 1..16
    if (cols <= 1) {
      ln_bwd_kernel<1><<<row_blocks, TPB>>>(x, gamma, mean, rstd, dy, dx, g_dg, g_db, N, C);
    } else if (cols <= 2) {
      ln_bwd_kernel<2><<<row_blocks, TPB>>>(x, gamma, mean, rstd, dy, dx, g_dg, g_db, N, C);
    } else if (cols <= 4) {
      ln_bwd_kernel<4><<<row_blocks, TPB>>>(x, gamma, mean, rstd, dy, dx, g_dg, g_db, N, C);
    } else if (cols <= 8) {
      ln_bwd_kernel<8><<<row_blocks, TPB>>>(x, gamma, mean, rstd, dy, dx, g_dg, g_db, N, C);
    } else {
      ln_bwd_kernel<16><<<row_blocks, TPB>>>(x, gamma, mean, rstd, dy, dx, g_dg, g_db, N, C);
    }
  } else {
    ln_bwd_generic_kernel<<<N, 256>>>(x, gamma, mean, rstd, dy, dx, g_dg, g_db, N, C);
  }
  convert2_kernel<<<(C + 255) / 256, 256>>>(g_dg, g_db, dgamma, dbeta, C);
}
