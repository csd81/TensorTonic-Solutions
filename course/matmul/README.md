This C++ / CUDA program performs a standard matrix multiplication (\(C = A \times B\)) using both the CPU and the GPU, benchmarking their execution times to calculate the performance speedup.
Here is a step-by-step breakdown of how the code works:

1. Configuration & Constants (Lines 6–9)

cpp
#define M 256          // Number of rows in A and C
#define K 512          // Number of columns in A and rows in B
#define N 256          // Number of columns in B and C
#define BLOCK_SIZE 32  // Threads per block dimension (32x32)
Use code with caution.
• Matrix Dimensions: Matrix \(A\) is \(256 \times 512\), Matrix \(B\) is \(512 \times 256\), resulting in a Matrix \(C\) of size \(256 \times 256\).
• Block Size: Sets up a 2D block grid where each thread block contains \(32 \times 32 = 1024\) threads (the maximum limit for modern NVIDIA GPUs).

2. CPU Implementation (Lines 28–39)

cpp
void matmul_cpu(float *A, float *B, float *C, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}
Use code with caution.
• This is a standard naive \(O(N^3)\) triple-nested loop running sequentially on a single CPU core.
• It computes each element C[i][j] by taking the dot product of row i from matrix \(A\) and column j from matrix \(B\).
• Since multi-dimensional arrays are flattened into 1D memory pools, it maps index positions using the formula (row * width) + column.

3. GPU Kernel Implementation (Lines 41–52)

cpp
__global__ void matmul_gpu(float *A, float *B, float *C, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int l = 0; l < k; l++) {
            sum += A[row * k + l] * B[l * n + col];
        }
        C[row * n + col] = sum;
    }
}
Use code with caution.
• __global__: Mark tells the compiler that this function is a kernel executed on the GPU, callable from the CPU.
• Parallel Architecture: Instead of using two outer loops like the CPU, the GPU launches thousands of threads simultaneously. Each individual thread computes exactly one element of the output matrix \(C\).
• Index Mapping: row and col compute which global output coordinate this specific thread is responsible for.
• Boundary Check (if): Ensures threads that fall outside the bounds of the matrix (if the dimensions aren't a clean multiple of BLOCK_SIZE) do not perform illegal memory operations.

4. The Main Workflow (Lines 54–146)

The main() function coordinates host-to-device operations:
1. Memory Allocation:
	• Allocates RAM on the CPU (Host) using standard malloc.
	• Allocates VRAM on the GPU (Device) using cudaMalloc (d_A, d_B, d_C).
2. Initialization: Fills matrices \(A\) and \(B\) with random floating-point values.
3. Data Transfer: Copies the input matrices from host memory over to device memory using cudaMemcpy(..., cudaMemcpyHostToDevice).
4. Grid Execution Configuration:cpp
dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
Use code with caution.
This creates enough blocks of \(32 \times 32\) threads to cover the entire output matrix surface area.
5. Warm-up Loop: Runs both implementations 3 times without tracking time to ensure cache and GPU clocks are primed.
6. Benchmarking: Loops both functions 20 times using a monotonic high-resolution clock timer to find the average time. cudaDeviceSynchronize() is explicitly called during the GPU benchmark to stop the CPU timer from stopping prematurely before the GPU finishes executing its asynchronous queue.
7. Performance Printout: Displays average times in microseconds and outputs the relative speed increase factor (Speedup).
8. Memory Cleanup: Deallocates RAM via free() and VRAM via cudaFree().
Would you like to know how to optimize this GPU kernel using Shared Memory (Tiling) to dramatically increase its speed further?
