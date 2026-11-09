#include <cuda_runtime.h>

#include <cstdint>

// SWiGLU: split x into halves x1 = x[0..N/2), x2 = x[N/2..N) and compute
// out[i] = SiLU(x1[i]) * x2[i], i.e. halfN outputs.
//
// At the benchmark (N = 100,000, halfN = 50,000) the kernel reads 400 KB and
// writes 200 KB — 600 KB, so it is launch-bound.  On this GPU an empty kernel
// costs 6.7 us and a kernel with the same traffic costs 8.0 us; the starter
// (scalar, IEEE division) costs 12.3 us.  Two things close that gap:
//
//   * one float4 per thread: 4 outputs per thread, two vector loads and one
//     vector store instead of 4 scalar loads/3 stores, and
//   * __fdividef + __expf instead of IEEE division: the division alone was
//     costing ~2.7 us at this size in the SiLU challenge, and the accuracy
//     trade (worst-case relative error ~1e-6) is far inside this challenge's
//     tolerance.
//
// The vector path needs halfN % 4 == 0 (so the x2 half starts 16-byte
// aligned) and 16-byte-aligned pointers; otherwise a scalar kernel runs.

#ifndef THREADS
#define THREADS 256
#endif

__device__ __forceinline__ float silu(float x) {
  return __fdividef(x, 1.0f + __expf(-x));
}

__global__ void swiglu_vec_kernel(const float4* __restrict__ x1,
                                  const float4* __restrict__ x2,
                                  float4* __restrict__ out, int n4) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) {
    const float4 a = __ldg(&x1[i]);
    const float4 b = __ldg(&x2[i]);
    out[i] = make_float4(silu(a.x) * b.x, silu(a.y) * b.y, silu(a.z) * b.z,
                         silu(a.w) * b.w);
  }
}

// General path: any halfN, any pointer alignment.
__global__ void swiglu_scalar_kernel(const float* __restrict__ x1,
                                     const float* __restrict__ x2,
                                     float* __restrict__ out, int halfN) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < halfN) out[i] = silu(__ldg(&x1[i])) * __ldg(&x2[i]);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
  if (N <= 0) return;
  const int halfN = N / 2;
  if (halfN <= 0) return;
  const float* x2 = input + halfN;

  const bool aligned =
      (((reinterpret_cast<uintptr_t>(input) | reinterpret_cast<uintptr_t>(output)) &
        15u) == 0) &&
      ((halfN & 3) == 0);
  if (aligned) {
    const int n4 = halfN >> 2;
    const int blocks = (n4 + THREADS - 1) / THREADS;
    swiglu_vec_kernel<<<blocks, THREADS>>>(
        reinterpret_cast<const float4*>(input),
        reinterpret_cast<const float4*>(x2), reinterpret_cast<float4*>(output),
        n4);
  } else {
    const int blocks = (halfN + THREADS - 1) / THREADS;
    swiglu_scalar_kernel<<<blocks, THREADS>>>(input, x2, output, halfN);
  }
  cudaDeviceSynchronize();
}
