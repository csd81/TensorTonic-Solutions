#include <cuda_runtime.h>
#include <float.h>

// Helper device function to implement float atomicMin via atomicCAS
__device__ void atomicMinFloat(float* addr, float value) {
    int* address_as_int = (int*)addr;
    int old = *address_as_int, assumed;
    
    do {
        assumed = old;
        // If the current minimum is already lower or equal, do nothing
        if (__int_as_float(assumed) <= value) {
            break;
        }
        // Attempt to atomically swap in the new minimum
        old = atomicCAS(address_as_int, assumed, __float_as_int(value));
    } while (assumed != old);
}

__global__ void init_result(float* result) {
    result[0] = FLT_MAX;
}

__global__ void min_kernel(const float* input, float* result, int N) {
    // 256 elements per block as dictated by the launcher configuration
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 1. Initial coalesced load from global memory into shared memory. 
    // Threads mapping out of bounds load FLT_MAX (the identity for min).
    sdata[tid] = (i < N) ? input[i] : FLT_MAX;
    __syncthreads();
    
    // 2. Tree reduction in shared memory down to the last warp (32 elements).
    // The loop handles s = 128, 64, 32. It stops when s = 16.
    for (unsigned int s = blockDim.x / 2; s > 16; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    // 3. Optimized Warp Shuffle for the final 32 elements.
    // Removes the tail of __syncthreads() calls and shared-memory round-trips.
    if (tid < 32) {
        float val = sdata[tid];
        
        // Execute the final 5 steps of the reduction directly in registers
        val = fminf(val, __shfl_down_sync(0xffffffff, val, 16));
        val = fminf(val, __shfl_down_sync(0xffffffff, val, 8));
        val = fminf(val, __shfl_down_sync(0xffffffff, val, 4));
        val = fminf(val, __shfl_down_sync(0xffffffff, val, 2));
        val = fminf(val, __shfl_down_sync(0xffffffff, val, 1));
        
        // 4. Thread 0 (the winner) folds the block minimum into global memory
        if (tid == 0) {
            atomicMinFloat(result, val);
        }
    }
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    init_result<<<1, 1>>>(result);
    min_kernel<<<blocks, threads>>>(input, result, N);
    
    cudaDeviceSynchronize();
}