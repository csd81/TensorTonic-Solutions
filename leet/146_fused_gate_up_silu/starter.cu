#include <cuda_runtime.h>
#include <math.h>

// CUDA Kernel configuration
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void fused_gate_up_silu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ W_gate_up,
    float* __restrict__ output,
    int M, int H, int I) 
{
    // Block row and column indicators
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * (BN / TN) + tx;

    // Shared memory allocation for tiling x and W_gate_up
    __shared__ float s_x[BM][BK];
    __shared__ float s_W_gate[BK][BN];
    __shared__ float s_W_up[BK][BN];

    // Thread-level register accumulators
    float accum_gate[TM][TN] = {0.0f};
    float accum_up[TM][TN]   = {0.0f};

    // Shared memory load routing indices
    // For x: BM * BK elements = 1024 elements total per block. 256 threads -> 4 elements/thread
    int s_x_row = tid / (BK / 4);
    int s_x_col = (tid % (BK / 4)) * 4;

    // For W: BK * BN elements = 1024 elements total per block. 256 threads -> 4 elements/thread
    int s_W_row = tid / (BN / 4);
    int s_W_col = (tid % (BN / 4)) * 4;

    // Loop over the inner dimension H in chunks of BK
    for (int k_offset = 0; k_offset < H; k_offset += BK) {
        
        // Coalesced collaborative loading of x into shared memory
        int global_x_row = by * BM + s_x_row;
        int global_x_col = k_offset + s_col_offset_x; // where s_col_offset_x matches s_x_col
        
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int r = s_x_row + (i * (256 / (BK / 4))); // stride threads over rows if needed
            // Safer explicit indexing map:
            int thread_r = (tid * 4 + i) / BK;
            int thread_c = (tid * 4 + i) % BK;
            int g_r = by * BM + thread_r;
            int g_c = k_offset + thread_c;
            
            if (g_r < M && g_c < H) {
                s_x[thread_r][thread_c] = x[g_r * H + g_c];
            } else {
                s_x[thread_r][thread_c] = 0.0f;
            }
        }

        // Coalesced collaborative loading of W_gate and W_up into shared memory
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int thread_r = (tid * 4 + i) / BN;
            int thread_c = (tid * 4 + i) % BN;
            int g_r = k_offset + thread_r;
            int g_c = bx * BN + thread_c;

            if (g_r < H && g_c < I) {
                // W_gate is located in columns [0, I)
                s_W_gate[thread_r][thread_c] = W_gate_up[g_r * (2 * I) + g_c];
                // W_up is located in columns [I, 2 * I)
                s_W_up[thread_r][thread_c]   = W_gate_up[g_r * (2 * I) + I + g_c];
            } else {
                s_W_gate[thread_r][thread_c] = 0.0f;
                s_W_up[thread_r][thread_c]   = 0.0f;
            }
        }

        __syncthreads();

        // Compute outer-product updates on the register tiles
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                float x_val = s_x[ty * TM + m][k];
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    accum_gate[m][n] += x_val * s_W_gate[k][tx * TN + n];
                    accum_up[m][n]   += x_val * s_W_up[k][tx * TN + n];
                }
            }
        }

        __syncthreads();
    }

    // Epilogue Phase: Fuse SiLU formulation & element-wise multiplication
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        int global_row = by * BM + ty * TM + m;
        if (global_row >= M) continue;

        #pragma unroll
        for (int n = 0; n < TN; ++n) {
            int global_col = bx * BN + tx * TN + n;
            if (global_col >= I) continue;

            float g = accum_gate[m][n];
            float u = accum_up[m][n];

            // SiLU activation: g / (1 + e^-g)
            float silu_g = g / (1.0f + __expf(-g));
            
            // Fused element-wise multiplication write back to output
            output[global_row * I + global_col] = silu_g * u;
        }
    }
}

extern "C" void solve(const float* x, const float* W_gate_up, float* output, int M, int H, int I) {
    // Configure thread block to dimensions matching execution tile sub-segments
    // 16x16 threads = 256 threads per block
    dim3 threadsPerBlock(BN / TN, BM / TM); 
    
    // Grid configuration computing execution boundary constraints safely
    dim3 numBlocks((I + BN - 1) / BN, (M + BM - 1) / BM);

    // Launch the fused execution kernel
    fused_gate_up_silu_kernel<<<numBlocks, threadsPerBlock>>>(x, W_gate_up, output, M, H, I);
}
