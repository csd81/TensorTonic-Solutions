#include <cuda_runtime.h>

#include <cstdint>

// 2D "valid" convolution: out[i][j] = sum_{m<KR,n<KC} in[i+m][j+n]*k[m][n].
//
// A block of 32x8 threads produces a 128x16 tile of outputs; each
// thread owns 2 rows x 4 columns.  The input tile plus halo is staged in
// shared memory with a 16-byte aligned padded stride, and the kernel is
// staged with each row padded to a multiple of four floats (zero filled) so
// taps load as aligned float4 broadcasts.
//
// Per kernel row m the thread keeps its window in registers and walks the
// kernel columns four at a time: RM float4 loads bring the next four input
// columns, one broadcast float4 brings four taps, and 4*RM*RN FMAs are
// issued.  The window then advances by four columns.

#define TX 32
#define TY 8
#define RM 2
#define RN 4
#define BW (TX * RN)          // 128 output columns per block
#define BH (TY * RM)          // 16 output rows per block
#define MAX_KERNEL_ELEMS 992  // 31 rows of 32 floats (padded)

__global__ void conv2d_valid_kernel(const float* __restrict__ input,
                                    const float* __restrict__ kernel,
                                    float* __restrict__ output, int input_rows,
                                    int input_cols, int kernel_rows,
                                    int kernel_cols, int output_rows,
                                    int output_cols) {
  __shared__ float s_kernel[MAX_KERNEL_ELEMS];
  extern __shared__ float s_input[];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * TX + tx;
  const int block_out_x = blockIdx.x * BW;
  const int block_out_y = blockIdx.y * BH;

  const int tile_w = BW + kernel_cols - 1;
  const int stride = (tile_w + 3) & ~3;  // 16-byte aligned rows
  const int tile_h = BH + kernel_rows - 1;

  // Stage the kernel, padding each row to a multiple of four floats with
  // zeros so the tap loads can use aligned float4 broadcasts.
  const int ks = (kernel_cols + 3) & ~3;
  const int k_padded = kernel_rows * ks;
  for (int i = tid; i < k_padded; i += TX * TY) {
    const int r = i / ks;
    const int c = i - r * ks;
    s_kernel[i] = (c < kernel_cols) ? kernel[r * kernel_cols + c] : 0.0f;
  }

  // Stage the input tile with halo; out-of-range elements are zeroed.
  for (int ly = ty; ly < tile_h; ly += TY) {
    const int gy = block_out_y + ly;
    const int base = ly * stride;
    const bool row_ok = gy < input_rows;
    for (int lx = tx; lx < tile_w; lx += TX) {
      const int gx = block_out_x + lx;
      s_input[base + lx] =
          (row_ok && gx < input_cols) ? input[gy * input_cols + gx] : 0.0f;
    }
  }
  __syncthreads();

  const int oy = block_out_y + ty * RM;
  const int ox = block_out_x + tx * RN;
  const int ly0 = ty * RM;
  const int lx0 = tx * RN;

  float a00 = 0.0f;
  float a01 = 0.0f;
  float a02 = 0.0f;
  float a03 = 0.0f;
  float a10 = 0.0f;
  float a11 = 0.0f;
  float a12 = 0.0f;
  float a13 = 0.0f;

  for (int m = 0; m < kernel_rows; ++m) {
    const float* row0 = s_input + (ly0 + 0 + m) * stride + lx0;
    float4 X0 = *reinterpret_cast<const float4*>(row0);
    const float4* p0 = reinterpret_cast<const float4*>(row0);
    const float* row1 = s_input + (ly0 + 1 + m) * stride + lx0;
    float4 X1 = *reinterpret_cast<const float4*>(row1);
    const float4* p1 = reinterpret_cast<const float4*>(row1);
    const float* krow = s_kernel + m * ks;

    for (int n = 0; n < ks; n += 4) {
      const float4 Y0 = p0[(n >> 2) + 1];
      const float4 Y1 = p1[(n >> 2) + 1];
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
      // Tap n+1.
      a00 += X0.y * k1; a01 += X0.z * k1;
      a02 += X0.w * k1; a03 += Y0.x * k1;
      a10 += X1.y * k1; a11 += X1.z * k1;
      a12 += X1.w * k1; a13 += Y1.x * k1;
      // Tap n+2.
      a00 += X0.z * k2; a01 += X0.w * k2;
      a02 += Y0.x * k2; a03 += Y0.y * k2;
      a10 += X1.z * k2; a11 += X1.w * k2;
      a12 += Y1.x * k2; a13 += Y1.y * k2;
      // Tap n+3.
      a00 += X0.w * k3; a01 += Y0.x * k3;
      a02 += Y0.y * k3; a03 += Y0.z * k3;
      a10 += X1.w * k3; a11 += Y1.x * k3;
      a12 += Y1.y * k3; a13 += Y1.z * k3;
      X0 = Y0;
      X1 = Y1;
    }
  }

  // Masked stores.
  if (oy + 0 < output_rows) {
    float* out = output + static_cast<size_t>(oy + 0) * output_cols + ox;
    if (ox + 0 < output_cols) out[0] = a00;
    if (ox + 1 < output_cols) out[1] = a01;
    if (ox + 2 < output_cols) out[2] = a02;
    if (ox + 3 < output_cols) out[3] = a03;
  }
  if (oy + 1 < output_rows) {
    float* out = output + static_cast<size_t>(oy + 1) * output_cols + ox;
    if (ox + 0 < output_cols) out[0] = a10;
    if (ox + 1 < output_cols) out[1] = a11;
    if (ox + 2 < output_cols) out[2] = a12;
    if (ox + 3 < output_cols) out[3] = a13;
  }
}

// input, kernel, output are device pointers.
extern "C" void solve(const float* input, const float* kernel, float* output,
                      int input_rows, int input_cols, int kernel_rows,
                      int kernel_cols) {
  if (input == nullptr || kernel == nullptr || output == nullptr) return;
  if (input_rows <= 0 || input_cols <= 0 || kernel_rows <= 0 ||
      kernel_cols <= 0) {
    return;
  }
  if (kernel_rows > 31 || kernel_cols > 31) return;
  const int output_rows = input_rows - kernel_rows + 1;
  const int output_cols = input_cols - kernel_cols + 1;
  if (output_rows <= 0 || output_cols <= 0) return;

  const int tile_w = BW + kernel_cols - 1;
  const int stride = (tile_w + 3) & ~3;
  const int tile_h = BH + kernel_rows - 1;
  const size_t shared_bytes =
      (static_cast<size_t>(stride) * tile_h + 16) * sizeof(float);

  const dim3 block(TX, TY);
  const dim3 grid((output_cols + BW - 1) / BW, (output_rows + BH - 1) / BH);
  conv2d_valid_kernel<<<grid, block, shared_bytes>>>(
      input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols,
      output_rows, output_cols);
  cudaDeviceSynchronize();
}
