#include <cuda_runtime.h>

#include <cstdint>

// 3D "valid" convolution:
//   out[i][j][k] = sum_{d<KD,r<KR,c<KC} in[i+d][j+r][k+c] * k[d][r][c].
//
// A block of 16x16 threads computes a 64x16x4 tile of outputs:
// thread (tx, ty) owns 4 consecutive columns for each of 4 output
// depths (one output row).  The input tile plus halo (RD+KD-1 planes,
// BH+KR-1 rows, BW+KC-1 columns) is staged in shared memory with a
// 16-byte aligned padded column stride; the kernel is staged with each
// row padded to a multiple of four floats (zero filled) so taps load as
// aligned float4 broadcasts.
//
// Per (kernel depth, kernel row) the thread reloads its window for each
// output depth and slides along the columns four taps at a time: one
// float4 load per output depth, one broadcast float4 for four taps, and
// 4*RD*RN FMAs.  A scalar tail handles KC % 4 leftover taps.

#define TX 16
#define TY 16
#define RN 4
#define RD 4
#define BW (TX * RN)   // 64 output columns per block
#define BH (TY)        // 16 output rows per block
#define MAX_KERNEL_PADDED 200  // 5*5 rows of 8 floats

__global__ void conv3d_kernel(const float* __restrict__ input,
                              const float* __restrict__ kernel,
                              float* __restrict__ output, int in_d,
                              int in_r, int in_c, int kd, int kr, int kc,
                              int out_d, int out_r, int out_c) {
  __shared__ __align__(16) float s_kernel[MAX_KERNEL_PADDED];
  extern __shared__ float s_tile[];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * TX + tx;
  const int bx = blockIdx.x * BW;  // first output column
  const int by = blockIdx.y * BH;  // first output row
  const int bz = blockIdx.z * RD;  // first output depth

  const int ks = (kc + 3) & ~3;
  const int planes = RD + kd - 1;
  const int rows = BH + kr - 1;
  const int cols = BW + kc - 1;
  const int stride = (cols + 3) & ~3;

  // Stage the kernel, zero-padding each (d, r) row to a multiple of four.
  const int krows = kd * kr;
  for (int i = tid; i < krows * ks; i += TX * TY) {
    const int r = i / ks;
    const int c = i - r * ks;
    s_kernel[i] = (c < kc) ? kernel[r * kc + c] : 0.0f;
  }

  // Zero only the row padding (columns [cols, stride)); the staging loop
  // below writes every valid column, including the zero fill for elements
  // outside the input, so no barrier is needed between the two.
  for (int i = tid; i < planes * rows; i += TX * TY) {
    const int base = i * stride;
    for (int c = cols; c < stride; ++c) s_tile[base + c] = 0.0f;
  }
  for (int d2 = 0; d2 < planes; ++d2) {
    const int gz = bz + d2;
    const bool z_ok = gz < in_d;
    const size_t gplane = static_cast<size_t>(gz) * in_r * in_c;
    for (int r2 = ty; r2 < rows; r2 += TY) {
      const int gy = by + r2;
      const bool y_ok = z_ok && gy < in_r;
      const int base = (d2 * rows + r2) * stride;
      const size_t grow = gplane + static_cast<size_t>(gy) * in_c;
      for (int c2 = tx; c2 < cols; c2 += TX) {
        const int gx = bx + c2;
        s_tile[base + c2] =
            (y_ok && gx < in_c) ? input[grow + gx] : 0.0f;
      }
    }
  }
  __syncthreads();

  const int ox = bx + tx * RN;
  const int oy = by + ty;
  const int cbase = tx * RN;

  // Accumulators: one row of RN columns for each of RD output depths.
  float a00 = 0.0f;
  float a01 = 0.0f;
  float a02 = 0.0f;
  float a03 = 0.0f;
  float a10 = 0.0f;
  float a11 = 0.0f;
  float a12 = 0.0f;
  float a13 = 0.0f;
  float a20 = 0.0f;
  float a21 = 0.0f;
  float a22 = 0.0f;
  float a23 = 0.0f;
  float a30 = 0.0f;
  float a31 = 0.0f;
  float a32 = 0.0f;
  float a33 = 0.0f;

  for (int dk = 0; dk < kd; ++dk) {
    for (int rk = 0; rk < kr; ++rk) {
      const float* krow = s_kernel + (dk * kr + rk) * ks;
      const int trow = ty + rk;
      const float* w0 =
          s_tile + ((0 + dk) * rows + trow) * stride + cbase;
      float4 X0 = *reinterpret_cast<const float4*>(w0);
      const float4* p0 = reinterpret_cast<const float4*>(w0);
      const float* w1 =
          s_tile + ((1 + dk) * rows + trow) * stride + cbase;
      float4 X1 = *reinterpret_cast<const float4*>(w1);
      const float4* p1 = reinterpret_cast<const float4*>(w1);
      const float* w2 =
          s_tile + ((2 + dk) * rows + trow) * stride + cbase;
      float4 X2 = *reinterpret_cast<const float4*>(w2);
      const float4* p2 = reinterpret_cast<const float4*>(w2);
      const float* w3 =
          s_tile + ((3 + dk) * rows + trow) * stride + cbase;
      float4 X3 = *reinterpret_cast<const float4*>(w3);
      const float4* p3 = reinterpret_cast<const float4*>(w3);

      int n = 0;
      for (; n + 4 <= kc; n += 4) {
        const float4 Y0 = p0[(n >> 2) + 1];
        const float4 Y1 = p1[(n >> 2) + 1];
        const float4 Y2 = p2[(n >> 2) + 1];
        const float4 Y3 = p3[(n >> 2) + 1];
        const float4 kk = *reinterpret_cast<const float4*>(krow + n);
        const float k0 = kk.x;
        const float k1 = kk.y;
        const float k2 = kk.z;
        const float k3 = kk.w;
        // Tap n+0.
        a00 += X0.x * k0; a01 += X0.y * k0;
        a02 += X0.z * k0; a03 += X0.w * k0;
        a10 += X1.x * k0; a11 += X1.y * k0;
        a12 += X1.z * k0; a13 += X1.w * k0;
        a20 += X2.x * k0; a21 += X2.y * k0;
        a22 += X2.z * k0; a23 += X2.w * k0;
        a30 += X3.x * k0; a31 += X3.y * k0;
        a32 += X3.z * k0; a33 += X3.w * k0;
        // Tap n+1.
        a00 += X0.y * k1; a01 += X0.z * k1;
        a02 += X0.w * k1; a03 += Y0.x * k1;
        a10 += X1.y * k1; a11 += X1.z * k1;
        a12 += X1.w * k1; a13 += Y1.x * k1;
        a20 += X2.y * k1; a21 += X2.z * k1;
        a22 += X2.w * k1; a23 += Y2.x * k1;
        a30 += X3.y * k1; a31 += X3.z * k1;
        a32 += X3.w * k1; a33 += Y3.x * k1;
        // Tap n+2.
        a00 += X0.z * k2; a01 += X0.w * k2;
        a02 += Y0.x * k2; a03 += Y0.y * k2;
        a10 += X1.z * k2; a11 += X1.w * k2;
        a12 += Y1.x * k2; a13 += Y1.y * k2;
        a20 += X2.z * k2; a21 += X2.w * k2;
        a22 += Y2.x * k2; a23 += Y2.y * k2;
        a30 += X3.z * k2; a31 += X3.w * k2;
        a32 += Y3.x * k2; a33 += Y3.y * k2;
        // Tap n+3.
        a00 += X0.w * k3; a01 += Y0.x * k3;
        a02 += Y0.y * k3; a03 += Y0.z * k3;
        a10 += X1.w * k3; a11 += Y1.x * k3;
        a12 += Y1.y * k3; a13 += Y1.z * k3;
        a20 += X2.w * k3; a21 += Y2.x * k3;
        a22 += Y2.y * k3; a23 += Y2.z * k3;
        a30 += X3.w * k3; a31 += Y3.x * k3;
        a32 += Y3.y * k3; a33 += Y3.z * k3;
        X0 = Y0;
        X1 = Y1;
        X2 = Y2;
        X3 = Y3;
      }
      for (; n < kc; ++n) {  // scalar tail, KC % 4 taps
        const float kk = krow[n];
        a00 += X0.x * kk; a01 += X0.y * kk; a02 += X0.z * kk; a03 += X0.w * kk;
        X0.x = X0.y; X0.y = X0.z; X0.z = X0.w;
        X0.w = w0[n + 4];
        a10 += X1.x * kk; a11 += X1.y * kk; a12 += X1.z * kk; a13 += X1.w * kk;
        X1.x = X1.y; X1.y = X1.z; X1.z = X1.w;
        X1.w = w1[n + 4];
        a20 += X2.x * kk; a21 += X2.y * kk; a22 += X2.z * kk; a23 += X2.w * kk;
        X2.x = X2.y; X2.y = X2.z; X2.z = X2.w;
        X2.w = w2[n + 4];
        a30 += X3.x * kk; a31 += X3.y * kk; a32 += X3.z * kk; a33 += X3.w * kk;
        X3.x = X3.y; X3.y = X3.z; X3.z = X3.w;
        X3.w = w3[n + 4];
      }
    }
  }

  // Masked stores.
  if (bz + 0 < out_d && oy < out_r) {
    float* ap0 = output +
        (static_cast<size_t>(bz + 0) * out_r + oy) * out_c + ox;
    if (ox + 0 < out_c) ap0[0] = a00;
    if (ox + 1 < out_c) ap0[1] = a01;
    if (ox + 2 < out_c) ap0[2] = a02;
    if (ox + 3 < out_c) ap0[3] = a03;
  }
  if (bz + 1 < out_d && oy < out_r) {
    float* ap1 = output +
        (static_cast<size_t>(bz + 1) * out_r + oy) * out_c + ox;
    if (ox + 0 < out_c) ap1[0] = a10;
    if (ox + 1 < out_c) ap1[1] = a11;
    if (ox + 2 < out_c) ap1[2] = a12;
    if (ox + 3 < out_c) ap1[3] = a13;
  }
  if (bz + 2 < out_d && oy < out_r) {
    float* ap2 = output +
        (static_cast<size_t>(bz + 2) * out_r + oy) * out_c + ox;
    if (ox + 0 < out_c) ap2[0] = a20;
    if (ox + 1 < out_c) ap2[1] = a21;
    if (ox + 2 < out_c) ap2[2] = a22;
    if (ox + 3 < out_c) ap2[3] = a23;
  }
  if (bz + 3 < out_d && oy < out_r) {
    float* ap3 = output +
        (static_cast<size_t>(bz + 3) * out_r + oy) * out_c + ox;
    if (ox + 0 < out_c) ap3[0] = a30;
    if (ox + 1 < out_c) ap3[1] = a31;
    if (ox + 2 < out_c) ap3[2] = a32;
    if (ox + 3 < out_c) ap3[3] = a33;
  }
}

// input, kernel, output are device pointers.
extern "C" void solve(const float* input, const float* kernel, float* output,
                      int input_depth, int input_rows, int input_cols,
                      int kernel_depth, int kernel_rows, int kernel_cols) {
  if (input == nullptr || kernel == nullptr || output == nullptr) return;
  if (input_depth <= 0 || input_rows <= 0 || input_cols <= 0 ||
      kernel_depth <= 0 || kernel_rows <= 0 || kernel_cols <= 0) {
    return;
  }
  if (kernel_depth > 5 || kernel_rows > 5 || kernel_cols > 5) return;
  const int out_d = input_depth - kernel_depth + 1;
  const int out_r = input_rows - kernel_rows + 1;
  const int out_c = input_cols - kernel_cols + 1;
  if (out_d <= 0 || out_r <= 0 || out_c <= 0) return;

  const int ks = (kernel_cols + 3) & ~3;
  const int planes = RD + kernel_depth - 1;
  const int rows = BH + kernel_rows - 1;
  const int cols = BW + kernel_cols - 1;
  const int stride = (cols + 3) & ~3;
  const size_t shared_bytes = (static_cast<size_t>(planes) * rows * stride + 8 +
                               static_cast<size_t>(kernel_depth) *
                                   kernel_rows * ks) *
                              sizeof(float);

  const dim3 block(TX, TY);
  const dim3 grid((out_c + BW - 1) / BW, (out_r + BH - 1) / BH,
                  (out_d + RD - 1) / RD);
  conv3d_kernel<<<grid, block, shared_bytes>>>(
      input, kernel, output, input_depth, input_rows, input_cols,
      kernel_depth, kernel_rows, kernel_cols, out_d, out_r, out_c);
  cudaDeviceSynchronize();
}
