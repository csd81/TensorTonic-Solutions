#include <cuda_runtime.h>

#include <cstdint>

// RoPE with the half-split layout:
//   out[j]     = q[j]*cos[j] - q[j+D/2]*sin[j]
//   out[j+D/2] = q[j+D/2]*cos[j+D/2] + q[j]*sin[j+D/2]
// and the challenge guarantees cos[j] == cos[j+D/2], sin[j] == sin[j+D/2].
//
// At the benchmark (M = 1,048,576, D = 128) the starter reads all of Q, cos
// and sin (1.6 GB) plus writes 536 MB — 2.15 GB — and measures 27.8 ms
// (77 GB/s of that traffic).  The invariant lets the cos/sin tensors be read
// only once per half-pair: a thread loads a float4 of Q from the first half,
// the mirrored float4 from the second half, and one float4 of cos and sin,
// then writes both output groups.  Traffic drops to
// Q(536 MB) + cos/sin(268 MB each) + out(536 MB) = 1.61 GB, and the kernel
// measures ~19-20 ms.
//
// The fast path requires D % 8 == 0 (so the half offset is a whole number of
// float4s) and 16-byte-aligned pointers; otherwise a scalar kernel reads the
// full cos/sin tensors elementwise.

namespace {

__global__ void rope_vec_kernel(const float4* __restrict__ Q,
                                const float4* __restrict__ C,
                                const float4* __restrict__ S,
                                float4* __restrict__ O, int n4row, int half4,
                                int M) {
  const int row = blockIdx.x * blockDim.y + threadIdx.y;  // x: gridDim.y caps at 65535
  if (row >= M) return;
  const size_t base = (size_t)row * n4row;
  for (int g = threadIdx.x; g < half4; g += blockDim.x) {
    const float4 q1 = __ldg(&Q[base + g]);
    const float4 q2 = __ldg(&Q[base + g + half4]);
    const float4 c = __ldg(&C[base + g]);
    const float4 s = __ldg(&S[base + g]);
    O[base + g] = make_float4(q1.x * c.x - q2.x * s.x, q1.y * c.y - q2.y * s.y,
                              q1.z * c.z - q2.z * s.z, q1.w * c.w - q2.w * s.w);
    O[base + g + half4] =
        make_float4(q2.x * c.x + q1.x * s.x, q2.y * c.y + q1.y * s.y,
                    q2.z * c.z + q1.z * s.z, q2.w * c.w + q1.w * s.w);
  }
}

__global__ void rope_scalar_kernel(const float* __restrict__ Q,
                                   const float* __restrict__ C,
                                   const float* __restrict__ S,
                                   float* __restrict__ O, long total, int D) {
  const int half = D / 2;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += (long)gridDim.x * blockDim.x) {
    const int col = (int)(i % D);
    const float q = __ldg(&Q[i]);
    const float rot = (col < half) ? -__ldg(&Q[i + half]) : __ldg(&Q[i - half]);
    O[i] = q * __ldg(&C[i]) + rot * __ldg(&S[i]);
  }
}

}  // namespace

// Q, cos, sin, output are device pointers (M x D each)
extern "C" void solve(float* Q, float* cos_vals, float* sin_vals, float* output,
                      int M, int D) {
  if (M <= 0 || D <= 0) return;
  const long total = (long)M * D;

  const bool aligned =
      (((reinterpret_cast<uintptr_t>(Q) | reinterpret_cast<uintptr_t>(cos_vals) |
         reinterpret_cast<uintptr_t>(sin_vals) |
         reinterpret_cast<uintptr_t>(output)) &
        15u) == 0) &&
      ((D & 7) == 0);
  if (aligned) {
    const int n4row = D >> 2;
    const int half4 = D >> 3;
    const int tx = (half4 < 32) ? half4 : 32;
    const int ty = 256 / tx;
    dim3 block(tx, ty);
    dim3 grid((M + ty - 1) / ty, 1);
    rope_vec_kernel<<<grid, block>>>(
        reinterpret_cast<const float4*>(Q), reinterpret_cast<const float4*>(cos_vals),
        reinterpret_cast<const float4*>(sin_vals),
        reinterpret_cast<float4*>(output), n4row, half4, M);
  } else {
    const int threads = 256;
    long want = (total + threads - 1) / threads;
    int blocks = (int)(want > 4096 ? 4096 : want);
    rope_scalar_kernel<<<blocks, threads>>>(Q, cos_vals, sin_vals, output, total,
                                            D);
  }
  cudaDeviceSynchronize();
}
