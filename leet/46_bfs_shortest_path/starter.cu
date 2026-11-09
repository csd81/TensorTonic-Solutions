#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Macro for 2D array indexing in flat memory
#define GRID(r, c) grid[(r) * cols + (c)]

/**
 * CUDA Kernel that performs one step/iteration of the parallel BFS.
 * 
 * It scans the grid for cells matching the 'current_dist'. For each matching cell,
 * it looks at its 4-way neighbors. If a neighbor is free (0) and unvisited,
 * it updates that neighbor's distance to 'current_dist + 1'.
 */
__global__ void bfs_kernel(const int* grid, int* dist, int rows, int cols, int current_dist, bool* d_changed) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r >= rows || c >= cols) return;

    // Process neighbors if the current cell was reached in the previous step
    if (dist[r * cols + c] == current_dist) {
        int dr[] = {-1, 1, 0, 0};
        int dc[] = {0, 0, -1, 1};

        for (int i = 0; i < 4; ++i) {
            int nr = r + dr[i];
            int nc = c + dc[i];

            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {
                int neighbor_idx = nr * cols + nc;
                
                // If neighbor is free and hasn't been visited yet
                if (grid[neighbor_idx] == 0 && dist[neighbor_idx] == -1) {
                    dist[neighbor_idx] = current_dist + 1;
                    *d_changed = true; // Signal that at least one update occurred
                }
            }
        }
    }
}

/**
 * Solves the shortest path on an unweighted 2D grid using parallel BFS in CUDA.
 * 
 * @param grid      Device pointer to the input grid (0 = free, 1 = obstacle)
 * @param result    Device pointer where the final shortest path distance must be stored
 * @param rows      Number of rows in the grid
 * @param cols      Number of columns in the grid
 * @param start_row Starting row coordinate
 * @param start_col Starting column coordinate
 * @param end_row   Ending row coordinate
 * @param end_col   Ending column coordinate
 */
extern "C" void solve(const int* grid, int* result, int rows, int cols, int start_row,
                      int start_col, int end_row, int end_col) {
    // Edge case: Start and end positions are identical
    if (start_row == end_row && start_col == end_col) {
        int zero = 0;
        cudaMemcpy(result, &zero, sizeof(int), cudaMemcpyHostToDevice);
        return;
    }

    size_t grid_size = rows * cols * sizeof(int);
    int* d_dist = nullptr;
    cudaMalloc(&d_dist, grid_size);

    // Initialize all distances to -1 (unvisited) on the host, then copy to device
    int* h_dist = (int*)malloc(grid_size);
    for (int i = 0; i < rows * cols; ++i) {
        h_dist[i] = -1;
    }
    h_dist[start_row * cols + start_col] = 0; // Distance to start is 0
    cudaMemcpy(d_dist, h_dist, grid_size, cudaMemcpyHostToDevice);
    free(h_dist);

    // Allocate convergence flag on device
    bool* d_changed = nullptr;
    cudaMalloc(&d_changed, sizeof(bool));

    // Execution configuration for 2D grid processing
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

    int current_dist = 0;
    bool h_changed = true;

    // Run BFS level-by-level
    while (h_changed) {
        h_changed = false;
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

        bfs_kernel<<<gridSize, blockSize>>>(grid, d_dist, rows, cols, current_dist, d_changed);
        
        // Synchronize and check if any new cells were found/updated
        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
        
        // Optimization check: If destination is already reached, we can break early
        int h_end_dist;
        cudaMemcpy(&h_end_dist, &d_dist[end_row * cols + end_col], sizeof(int), cudaMemcpyDeviceToHost);
        if (h_end_dist != -1) {
            break;
        }

        current_dist++;
    }

    // Copy the final distance of the destination cell to the result pointer
    cudaMemcpy(result, &d_dist[end_row * cols + end_col], sizeof(int), cudaMemcpyDeviceToDevice);

    // Clean up device allocations
    cudaFree(d_dist);
    cudaFree(d_changed);
}
