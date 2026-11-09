#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Device-side lookup table for FP4 E2M1 decoding.
// Maps a 4-bit nibble index (0x0 to 0xF) to its actual float32 value.
__constant__ float FP4_LUT[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Device-side function to decode an FP8 E4M3 (uint8_t bit pattern) into a float32.
// Format: 1 sign bit, 4 exponent bits, 3 mantissa bits.
// Since we only need standard decoding for the block scales, we follow the FP8 E4M3 spec.
__device__ __forceinline__ float decode_e4m3(uint8_t bits) {
    if (bits == 0x7F || bits == 0xFF) return 0.0f; // NaN/Inf handled as 0 or clipped in inference
    
    int sign = (bits >> 7) & 1;
    int exp = (bits >> 3) & 0x0F;
    int mant = bits & 0x07;
    
    float val;
    if (exp == 0) {
        // Subnormal numbers
        val = (float)mant * (1.0f / 128.0f) * 0.0625f; // mant * 2^-3 * 2^-4
    } else {
        // Normal numbers
        // 1.f + mant/8.f * 2^(exp - 7)
        float m = 1.0f + (float)mant * 0.125f;
        // Fast pow2 calculation for 2^(exp - 7)
        val = m * __powf(2.0f, (float)exp - 7.0f);
    }
    
    return sign ? -val : val;
}

// Tiled GEMM Kernel for NVFP4 Inference
__global__ void nvfp4_gemm_kernel(
    const uint8_t* __restrict__ x_q, 
    const uint8_t* __restrict__ x_scales, 
    const uint8_t* __restrict__ w_q, 
    const uint8_t* __restrict__ w_scales, 
    float alpha, 
    __half* __restrict__ y, 
    int M, int N, int K) 
{
    // Compute the global row and column for the output matrix y
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= M || col >= N) return;
    
    float accum = 0.0f;
    
    // Loop over the reduction dimension K. 
    // Each byte contains two 4-bit values (high nibble: even index, low nibble: odd index).
    // Therefore, iterating through K bytes translates to K * 2 elements. 
    // Since K in the signature is the actual number of elements, the loop runs up to K / 2 bytes.
    int num_bytes_k = K / 2;
    
    for (int byte_k = 0; byte_k < num_bytes_k; ++byte_k) {
        // 1. Fetch packed bytes for activations and weights
        uint8_t x_byte = x_q[row * num_bytes_k + byte_k];
        uint8_t w_byte = w_q[col * num_bytes_k + byte_k]; // w_q is N x K (row-major)
        
        // 2. Unpack individual FP4 values (high nibble = even, low nibble = odd)
        uint8_t x_high = (x_byte >> 4) & 0x0F;
        uint8_t x_low  = x_byte & 0x0F;
        uint8_t w_high = (w_byte >> 4) & 0x0F;
        uint8_t w_low  = w_byte & 0x0F;
        
        // 3. Decode FP4 values using constant memory lookup table
        float x_val_even = FP4_LUT[x_high];
        float x_val_odd  = FP4_LUT[x_low];
        float w_val_even = FP4_LUT[w_high];
        float w_val_odd  = FP4_LUT[w_low];
        
        // 4. Track block scales. Block size is 16 elements (which equals 8 packed bytes).
        int element_idx_even = byte_k * 2;
        int element_idx_odd  = element_idx_even + 1;
        
        uint8_t x_scale_byte = x_scales[row * (K / 16) + (element_idx_even / 16)];
        uint8_t w_scale_byte = w_scales[col * (K / 16) + (element_idx_even / 16)];
        
        float x_scale = decode_e4m3(x_scale_byte);
        float w_scale = decode_e4m3(w_scale_byte);
        
        // 5. Apply scale and multiply-accumulate
        accum += (x_val_even * x_scale) * (w_val_even * w_scale);
        accum += (x_val_odd * x_scale) * (w_val_odd * w_scale);
    }
    
    // Final tensor scaling by alpha and write back as float16
    y[row * N + col] = __float2half(alpha * accum);
}

// Host solving wrapper function
extern "C" void solve(
    const uint8_t* x_q, const uint8_t* x_scales, const uint8_t* w_q,
    const uint8_t* w_scales, float alpha, __half* y, int M, int N, int K) 
{
    // Define 2D thread block configuration (optimal for standard matrix coordinates)
    dim3 threadsPerBlock(16, 16);
    
    // Grid configuration dynamically scaling based on output matrix dimensions
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
                   
    nvfp4_gemm_kernel<<<numBlocks, threadsPerBlock>>>(
        x_q, x_scales, w_q, w_scales, alpha, y, M, N, K
    );
    
    // Sync to guarantee error catching during development or benchmarking
    cudaDeviceSynchronize();
}
