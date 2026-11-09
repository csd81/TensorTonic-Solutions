#include <cuda_runtime.h>

// Naive in-place RGBA inversion: one GPU thread per pixel.  Each thread does
// three direct byte loads and three direct byte stores from/to global memory;
// the alpha byte is never touched.  No vector types, no packing, one
// straightforward kernel launch.
__global__ void color_inversion_naive_kernel(unsigned char* image, int pixels) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < pixels) {
    unsigned char* px = image + 4 * i;
    px[0] = static_cast<unsigned char>(255 - px[0]);
    px[1] = static_cast<unsigned char>(255 - px[1]);
    px[2] = static_cast<unsigned char>(255 - px[2]);
  }
}

// image is a device pointer (i.e. pointer to memory on the GPU).
extern "C" void solve(unsigned char* image, int width, int height) {
  if (image == nullptr || width <= 0 || height <= 0) return;
  const int pixels = width * height;
  const int threads = 256;
  const int blocks = (pixels + threads - 1) / threads;
  color_inversion_naive_kernel<<<blocks, threads>>>(image, pixels);
  cudaDeviceSynchronize();
}
