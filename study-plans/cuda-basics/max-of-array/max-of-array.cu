#include <cuda_runtime.h>
#include <float.h>

// Helper function to reduce maximum values across a warp (32 threads)
__device__ inline float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        // Exchange variables across warp lanes without shared memory
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Custom atomicMax for floating point numbers utilizing atomicCAS
__device__ void atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    
    // Test-and-test-and-set approach: only attempt atomic update if our value is larger.
    // This dramatically reduces serialization contention.
    while (val > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(val));
        
        // If the value returned by CAS matches our assumed value, the swap succeeded
        if (assumed == old) {
            break;
        }
    }
}

__global__ void max_kernel(const float* input, float* result, int N) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    // Each thread reads one element; pad out-of-bounds with -FLT_MAX
    float val = (gid < N) ? input[gid] : -FLT_MAX;
    
    // Step 1: Thread-level values are reduced down to 1 value per warp
    val = warpReduceMax(val);
    
    // Step 2: Store each warp's maximum into shared memory
    __shared__ float warp_max[32]; // Accommodates up to 1024 threads (32 warps)
    
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    if (lane_id == 0) {
        warp_max[warp_id] = val;
    }
    
    // Synchronize to ensure all warps have written their max values
    __syncthreads();
    
    // Step 3: The first warp executes one final reduction of the partial warp maximums
    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        val = (lane_id < num_warps) ? warp_max[lane_id] : -FLT_MAX;
        
        val = warpReduceMax(val);
        
        // Thread 0 updates the global result for the entire block
        if (lane_id == 0) {
            atomicMaxFloat(result, val);
        }
    }
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    // Set the baseline scalar to -FLT_MAX before the reduction starts
    float neg_inf = -FLT_MAX;
    cudaMemcpy(result, &neg_inf, sizeof(float), cudaMemcpyHostToDevice);
    
    max_kernel<<<blocks, threads>>>(input, result, N);
    
    cudaDeviceSynchronize();
}