// Gaussian blur (challenge 28).
//
// output[i,j] = sum_{m,n} input[i+m-ph, j+n-pw] * kernel[m,n], zero-padded.
// Benchmark: 512 x 512 image, 7 x 7 kernel -- 12.8 M multiply-adds on 1 MB in
// and 1 MB out, so this is a tiny kernel: a pure float4 copy of the same
// 2 MB measures ~32 us including launch, and that is the floor.
//
// The starter measures 212 us.  It uses a fixed 36x36 shared tile (sized for
// the maximum radius 10 regardless of the actual kernel) and one output per
// thread with a scalar shared load per multiply-add; the runtime kernel-size
// loops also prevent unrolling.
//
// This version:
//
//  1. Sizes the shared tile from the actual halo (BLOCK + 2*ph) x
//     (BLOCKW + 2*pw) via dynamic shared memory, so the tile load cost
//     matches the kernel at hand.
//  2. Computes 4 adjacent output columns per thread.  For each kernel row the
//     thread reads a (4 + KC - 1)-wide window from shared memory once and
//     uses every value for up to 4 multiply-adds.
//  3. Instantiates the kernel for each odd kernel width 3..21 (the challenge
//     constraint).  With KC a compile-time constant the kc loop fully
//     unrolls and the overlapping window reads are recognized as the same
//     address (no interleaved stores to shared), so the compiler keeps the
//     window in registers: ~10 shared loads per 28 multiply-adds instead of
//     the naive 28.  This is the main win: 84 us (same kernel, runtime KC)
//     -> 55 us.  A runtime-KC fallback kernel stays for out-of-contract
//     widths.
//  4. Stores float4 when the row stride and column are 4-aligned (the
//     benchmark case), scalars otherwise.
//
// Measured (see README): ~55 us vs 212 us for the starter, against a ~32 us
// copy floor.

#include <cuda_runtime.h>

namespace {

constexpr int TX = 16;   // threads along columns
constexpr int TY = 16;   // threads along rows
constexpr int CPT = 4;   // output columns per thread
constexpr int BW = TX * CPT;  // 64 output columns per block
constexpr int BH = 16;        // output rows per block

template <int KCT>
__global__ void blur_kernel(const float* __restrict__ input,
                            const float* __restrict__ kernel,
                            float* __restrict__ output, int R, int C, int KR) {
  extern __shared__ float tile[];

  const int ph = KR >> 1, pw = KCT >> 1;
  const int pitch = BW + 2 * pw;
  const int rows = BH + 2 * ph;
  const int r0 = blockIdx.y * BH - ph;
  const int c0 = blockIdx.x * BW - pw;

  // Cooperative tile load with zero padding.  Nested (row, col) loops keep
  // the pitch out of any division: a flat index would need idx/pitch and
  // idx%pitch with a runtime divisor in every iteration.
  for (int i = threadIdx.y; i < rows; i += TY) {
    const int gr = r0 + i;
    for (int j = threadIdx.x; j < pitch; j += TX) {
      const int gc = c0 + j;
      tile[i * pitch + j] = (gr >= 0 && gr < R && gc >= 0 && gc < C)
                                ? input[(size_t)gr * C + gc]
                                : 0.0f;
    }
  }
  __syncthreads();

  const int orow = blockIdx.y * BH + threadIdx.y;
  const int ocol = blockIdx.x * BW + threadIdx.x * CPT;
  if (orow >= R) return;

  float acc[CPT];
#pragma unroll
  for (int c = 0; c < CPT; ++c) acc[c] = 0.0f;

  const int tr = threadIdx.y, tc = threadIdx.x * CPT;
  for (int kr = 0; kr < KR; ++kr) {
    const float* rowp = &tile[(tr + kr) * pitch + tc];
    const float* krow = kernel + kr * KCT;
#pragma unroll
    for (int kc = 0; kc < KCT; ++kc) {
      const float kv = krow[kc];
#pragma unroll
      for (int c = 0; c < CPT; ++c) acc[c] = fmaf(rowp[kc + c], kv, acc[c]);
    }
  }

  const size_t o = (size_t)orow * C + ocol;
  if ((C & 3) == 0 && ocol + CPT <= C) {
#pragma unroll
    for (int c = 0; c < CPT; c += 4) {
      *reinterpret_cast<float4*>(output + o + c) =
          make_float4(acc[c], acc[c + 1], acc[c + 2], acc[c + 3]);
    }
  } else {
#pragma unroll
    for (int c = 0; c < CPT; ++c) {
      if (ocol + c < C) output[o + c] = acc[c];
    }
  }
}

// Fallback for kernel widths outside the challenge's odd 3..21 set: same
// block shape, runtime kernel-width loop.
__global__ void blur_kernel_generic(const float* __restrict__ input,
                                    const float* __restrict__ kernel,
                                    float* __restrict__ output, int R, int C,
                                    int KR, int KC) {
  extern __shared__ float tile[];
  const int ph = KR >> 1, pw = KC >> 1;
  const int pitch = BW + 2 * pw;
  const int rows = BH + 2 * ph;
  const int r0 = blockIdx.y * BH - ph;
  const int c0 = blockIdx.x * BW - pw;
  for (int i = threadIdx.y; i < rows; i += TY) {
    const int gr = r0 + i;
    for (int j = threadIdx.x; j < pitch; j += TX) {
      const int gc = c0 + j;
      tile[i * pitch + j] = (gr >= 0 && gr < R && gc >= 0 && gc < C)
                                ? input[(size_t)gr * C + gc]
                                : 0.0f;
    }
  }
  __syncthreads();
  const int orow = blockIdx.y * BH + threadIdx.y;
  const int ocol = blockIdx.x * BW + threadIdx.x * CPT;
  if (orow >= R) return;
  float acc[CPT];
#pragma unroll
  for (int c = 0; c < CPT; ++c) acc[c] = 0.0f;
  const int tr = threadIdx.y, tc = threadIdx.x * CPT;
  for (int kr = 0; kr < KR; ++kr) {
    const float* rowp = &tile[(tr + kr) * pitch + tc];
    const float* krow = kernel + kr * KC;
#pragma unroll 4
    for (int kc = 0; kc < KC; ++kc) {
      const float kv = krow[kc];
#pragma unroll
      for (int c = 0; c < CPT; ++c) acc[c] = fmaf(rowp[kc + c], kv, acc[c]);
    }
  }
  const size_t o = (size_t)orow * C + ocol;
  if ((C & 3) == 0 && ocol + CPT <= C) {
#pragma unroll
    for (int c = 0; c < CPT; c += 4) {
      *reinterpret_cast<float4*>(output + o + c) =
          make_float4(acc[c], acc[c + 1], acc[c + 2], acc[c + 3]);
    }
  } else {
#pragma unroll
    for (int c = 0; c < CPT; ++c) {
      if (ocol + c < C) output[o + c] = acc[c];
    }
  }
}

}  // namespace

// input, kernel, output are device pointers.
extern "C" void solve(const float* input, const float* kernel, float* output,
                      int input_rows, int input_cols, int kernel_rows,
                      int kernel_cols) {
  if (input_rows <= 0 || input_cols <= 0 || kernel_rows <= 0 ||
      kernel_cols <= 0) {
    return;
  }
  const dim3 block(TX, TY);
  const dim3 grid((input_cols + BW - 1) / BW, (input_rows + BH - 1) / BH);
  const size_t smem =
      (size_t)(BH + 2 * (kernel_rows / 2)) *
      (BW + 2 * (kernel_cols / 2)) * sizeof(float);

  switch (kernel_cols) {
    case 3:
      blur_kernel<3><<<grid, block, smem>>>(input, kernel, output, input_rows,
                                            input_cols, kernel_rows);
      break;
    case 5:
      blur_kernel<5><<<grid, block, smem>>>(input, kernel, output, input_rows,
                                            input_cols, kernel_rows);
      break;
    case 7:
      blur_kernel<7><<<grid, block, smem>>>(input, kernel, output, input_rows,
                                            input_cols, kernel_rows);
      break;
    case 9:
      blur_kernel<9><<<grid, block, smem>>>(input, kernel, output, input_rows,
                                            input_cols, kernel_rows);
      break;
    case 11:
      blur_kernel<11><<<grid, block, smem>>>(input, kernel, output, input_rows,
                                             input_cols, kernel_rows);
      break;
    case 13:
      blur_kernel<13><<<grid, block, smem>>>(input, kernel, output, input_rows,
                                             input_cols, kernel_rows);
      break;
    case 15:
      blur_kernel<15><<<grid, block, smem>>>(input, kernel, output, input_rows,
                                             input_cols, kernel_rows);
      break;
    case 17:
      blur_kernel<17><<<grid, block, smem>>>(input, kernel, output, input_rows,
                                             input_cols, kernel_rows);
      break;
    case 19:
      blur_kernel<19><<<grid, block, smem>>>(input, kernel, output, input_rows,
                                             input_cols, kernel_rows);
      break;
    case 21:
      blur_kernel<21><<<grid, block, smem>>>(input, kernel, output, input_rows,
                                             input_cols, kernel_rows);
      break;
    default:
      blur_kernel_generic<<<grid, block, smem>>>(
          input, kernel, output, input_rows, input_cols, kernel_rows,
          kernel_cols);
      break;
  }
}
