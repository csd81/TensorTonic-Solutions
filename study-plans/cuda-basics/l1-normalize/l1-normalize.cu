#include <cuda_runtime.h>
#include <math.h>

// Phase 1: Reduce the array to a single L1 norm (sum of absolute values)
__global__ void l1_reduce_kernel(const float* input, float* scratch_sum, int N) {
    float sum = 0.0f;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 1. Grid-stride loop: Each thread accumulates multiple elements
    for (int i = tid; i < N; i += stride) {
        sum += fabsf(input[i]);
    }

    // 2. Warp-level reduction
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // 3. Block-level reduction using shared memory
    __shared__ float shared_sums[32]; // Accommodates up to 1024 threads per block
    int lane = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;

    if (lane == 0) {
        shared_sums[warpId] = sum;
    }
    __syncthreads();

    // 4. Final reduction by the first warp
    if (warpId == 0) {
        // Read partial sums from shared memory; pad with 0.0f for empty warps
        sum = (lane < (blockDim.x / 32)) ? shared_sums[lane] : 0.0f;
        
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(mask, sum, offset);
        }
        
        // 5. One single atomic add per block to the global accumulator
        if (lane == 0) {
            atomicAdd(scratch_sum, sum);
        }
    }
}

// Phase 2: Map the global reciprocal across the array
__global__ void l1_map_kernel(const float* input, float* output, const float* scratch_sum, int N) {
    // Broadcast the denominator to all threads and convert to a multiply
    float inv_sum = 1.0f / (*scratch_sum);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride map
    for (int i = tid; i < N; i += stride) {
        output[i] = input[i] * inv_sum;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    // Allocate and zero out the global scratch accumulator
    float* d_sum;
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));

    int threads = 256;
    
    // Dynamically size the grid to saturate the GPU without excessive atomics
    int deviceId;
    cudaGetDevice(&deviceId);
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId);
    
    // Target ~32 blocks per SM for excellent latency hiding
    int blocks = numSMs * 32; 
    
    // Clamp blocks if N is small
    if (blocks > (N + threads - 1) / threads) {
        blocks = (N + threads - 1) / threads;
    }

    // Launch Phase 1: Reduction
    l1_reduce_kernel<<<blocks, threads>>>(input, d_sum, N);
    
    // Launch Phase 2: Map 
    // The kernel boundary acts as an implicit global barrier
    l1_map_kernel<<<blocks, threads>>>(input, output, d_sum, N);

    cudaDeviceSynchronize();
    cudaFree(d_sum);
}