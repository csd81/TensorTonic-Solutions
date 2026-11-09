#include <cuda_runtime.h>

// Naive baseline for challenge 105, kept as the starter's shape: one block
// per (batch, group), but every element recomputes its (channel, hw) indices
// with runtime divisions and the loads are scalar.  Measures 2.66 ms at the
// benchmark shape (75.8 GB/s of the two-reads-plus-write traffic; the wall is
// ~82 GB/s and the shipped kernel reaches it).

// Coalesced block reduction for computing sum
__device__ __forceinline__ float blockReduceSum(float val) {
    static __shared__ float shared[32]; 
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    // Warp-level parallel reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // Final warp reduction over the block's partial sums
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) {
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }
    return val;
}

__global__ void group_norm_naive_kernel(const float* __restrict__ X, 
                                  const float* __restrict__ gamma, 
                                  const float* __restrict__ beta, 
                                  float* __restrict__ Y, 
                                  int N, int C, int H, int W, int G, float eps) {
    // Each block manages one specific (batch, group) pair
    int n = blockIdx.x / G;
    int g = blockIdx.x % G;

    int channels_per_group = C / G;
    int hw_size = H * W;
    int num_elements = channels_per_group * hw_size;

    // Compute parallel global sum and sum of squares
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;

    for (int i = threadIdx.x; i < num_elements; i += blockDim.x) {
        int c_local = i / hw_size;
        int hw_idx = i % hw_size;
        int c_global = g * channels_per_group + c_local;
        
        // Target linear tracking coordinate in memory
        int total_idx = ((n * C + c_global) * H * W) + hw_idx;
        float val = X[total_idx];

        local_sum += val;
        local_sq_sum += val * val;
    }

    // Accumulate total sum and square sums across the block
    float total_sum = blockReduceSum(local_sum);
    float total_sq_sum = blockReduceSum(local_sq_sum);

    // Shared tracking variables for block-level broadcast
    __shared__ float mean;
    __shared__ float inv_std;

    if (threadIdx.x == 0) {
        mean = total_sum / num_elements;
        float variance = (total_sq_sum / num_elements) - (mean * mean);
        if (variance < 0.0f) variance = 0.0f; // Handle numerical noise
        inv_std = rsqrtf(variance + eps);
    }
    __syncthreads();

    // Normalize, scale, and shift each localized channel element
    for (int i = threadIdx.x; i < num_elements; i += blockDim.x) {
        int c_local = i / hw_size;
        int hw_idx = i % hw_size;
        int c_global = g * channels_per_group + c_local;
        
        int total_idx = ((n * C + c_global) * H * W) + hw_idx;

        float g_val = gamma[c_global];
        float b_val = beta[c_global];

        Y[total_idx] = g_val * ((X[total_idx] - mean) * inv_std) + b_val;
    }
}

extern "C" void solve(const float* X, const float* gamma, const float* beta, float* Y, int N, int C,
                      int H, int W, int G, float eps) {
    // Number of total operational block components is N * G
    int num_blocks = N * G;
    
    // 256 threads offer excellent performance and balance occupancy limits
    int num_threads = 256; 

    group_norm_naive_kernel<<<num_blocks, num_threads>>>(X, gamma, beta, Y, N, C, H, W, G, eps);
}
