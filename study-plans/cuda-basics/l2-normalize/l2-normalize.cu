#include <cuda_runtime.h>
#include <math.h>

__global__ void reduce_sq_sum(const float* input, float* sumv, int N) {
    // Shared memory for block-level reduction
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Grid-stride loop to handle array sizes larger than the grid
    float partial_sum = 0.0f;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        float val = input[i];
        partial_sum += val * val;
    }
    
    // Store each thread's partial sum in shared memory
    sdata[tid] = partial_sum;
    __syncthreads();

    // Perform contiguous tree reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 of the block adds the block's total to the global sum
    if (tid == 0) {
        atomicAdd(sumv, sdata[0]);
    }
}

__global__ void divide_by_sqrt(const float* input, float* output, const float* sumv, int N) {
    // Calculate reciprocal square root once and broadcast it across the threads.
    // rsqrtf evaluates 1 / sqrt(x) in a single fast hardware instruction.
    float factor = rsqrtf(*sumv);
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Grid-stride loop matching the scaling array
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        output[i] = input[i] * factor;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    float* d_sum;
    // (void**) cast safely handles NVCC stricter C++ types
    cudaMalloc((void**)&d_sum, sizeof(float)); 
    cudaMemset(d_sum, 0, sizeof(float));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Launch Phase 1
    reduce_sq_sum<<<blocks, threads>>>(input, d_sum, N);
    
    // Launch Phase 2
    // Implicit execution barrier: Kernel 2 executes only after Kernel 1 finishes
    // on the default stream, so no intermediate cudaDeviceSynchronize is needed.
    divide_by_sqrt<<<blocks, threads>>>(input, output, d_sum, N);

    // Final Sync before exiting
    cudaDeviceSynchronize();
    cudaFree(d_sum);
}