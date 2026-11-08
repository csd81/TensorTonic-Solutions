#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>

__global__ void online_softmax_kernel(
    const void* input,
    void* output,
    int rows,
    int cols,
    long long input_row_stride,
    long long input_col_stride,
    long long output_row_stride,
    long long output_col_stride,
    int dtype_code
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) return;

    // Allocate shared memory for block-wide reduction:
    // smem_m holds maximums, smem_l holds exponential sums
    extern __shared__ float smem[];
    float* smem_m = smem;
    float* smem_l = smem + blockDim.x;

    // Initialize local thread summaries
    float m_local = -CUDART_INF_F;
    float l_local = 0.0f;

    // 1st Pass: Compute local max and local sum of exponentials
    for (int col = tid; col < cols; col += blockDim.x) {
        float x;
        long long in_idx = row * input_row_stride + col * input_col_stride;
        
        if (dtype_code == 0) {
            x = ((const float*)input)[in_idx];
        } else if (dtype_code == 1) {
            x = __half2float(((const __half*)input)[in_idx]);
        } else {
            x = __bfloat162float(((const __nv_bfloat16*)input)[in_idx]);
        }

        // Online Softmax update rule
        float m_new = max(m_local, x);
        float l_new = 0.0f;
        
        if (m_local > -CUDART_INF_F) {
            l_new += expf(m_local - m_new) * l_local;
        }
        l_new += expf(x - m_new);
        
        m_local = m_new;
        l_local = l_new;
    }

    // Write thread-local summaries into shared memory
    smem_m[tid] = m_local;
    smem_l[tid] = l_local;
    __syncthreads();

    // Block-wide tree reduction to find the global row max and sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float m1 = smem_m[tid];
            float l1 = smem_l[tid];
            float m2 = smem_m[tid + stride];
            float l2 = smem_l[tid + stride];

            float m_new = max(m1, m2);
            float l_new = 0.0f;

            if (m1 > -CUDART_INF_F) l_new += expf(m1 - m_new) * l1;
            if (m2 > -CUDART_INF_F) l_new += expf(m2 - m_new) * l2;

            smem_m[tid] = m_new;
            smem_l[tid] = l_new;
        }
        __syncthreads();
    }

    // Thread 0 now has the global max and sum for the row. 
    // All threads read from index 0 safely (shared memory is synchronous here)
    float global_m = smem_m[0];
    float global_l = smem_l[0];

    // 2nd Pass: Normalize and write out probabilities
    for (int col = tid; col < cols; col += blockDim.x) {
        float x;
        long long in_idx = row * input_row_stride + col * input_col_stride;
        long long out_idx = row * output_row_stride + col * output_col_stride;

        if (dtype_code == 0) {
            x = ((const float*)input)[in_idx];
        } else if (dtype_code == 1) {
            x = __half2float(((const __half*)input)[in_idx]);
        } else {
            x = __bfloat162float(((const __nv_bfloat16*)input)[in_idx]);
        }

        // Apply formula y_j = exp(x_j - m) / l
        float y = expf(x - global_m) / global_l;

        // Store back in original dtype format
        if (dtype_code == 0) {
            ((float*)output)[out_idx] = y;
        } else if (dtype_code == 1) {
            ((__half*)output)[out_idx] = __float2half(y);
        } else {
            ((__nv_bfloat16*)output)[out_idx] = __float2bfloat16(y);
        }
    }
}

// Ensure the solve function interfaces with C properly
extern "C" void solve(
    const void* input,
    void* output,
    int rows,
    int cols,
    long long input_row_stride,
    long long input_col_stride,
    long long output_row_stride,
    long long output_col_stride,
    int block_size,
    int dtype_code
) {
    if (
        rows < 0 ||
        cols < 0 ||
        block_size < 32 ||
        block_size > 1024 ||
        (block_size & (block_size - 1)) != 0 || // Ensures it is a power of 2
        dtype_code < 0 ||
        dtype_code > 2
    ) {
        cudaDeviceSynchronize();
        return;
    }
    if (rows == 0 || cols == 0) {
        cudaDeviceSynchronize();
        return;
    }
    
    // Allocate 2 floats per thread (one for m, one for l)
    size_t shared_bytes = 2 * block_size * sizeof(float);
    
    online_softmax_kernel<<<rows, block_size, shared_bytes>>>(
        input,
        output,
        rows,
        cols,
        input_row_stride,
        input_col_stride,
        output_row_stride,
        output_col_stride,
        dtype_code
    );
    cudaDeviceSynchronize();
}