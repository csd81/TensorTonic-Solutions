#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define BLOCK_SIZE 256
#define MAX_K 128

// A helper structure to store distance and index together
struct Candidate {
    int dist;
    int idx;
};

// Device helper function to maintain a max-heap of size k
__device__ void insert_max_heap(Candidate* heap, int& heap_size, int k, int dist, int idx) {
    if (heap_size < k) {
        heap[heap_size] = {dist, idx};
        // Bubble up
        int i = heap_size;
        heap_size++;
        while (i > 0) {
            int p = (i - 1) / 2;
            if (heap[i].dist > heap[p].dist || (heap[i].dist == heap[p].dist && heap[i].idx > heap[p].idx)) {
                Candidate tmp = heap[i]; heap[i] = heap[p]; heap[p] = tmp;
                i = p;
            } else {
                break;
            }
        }
    } else {
        // Compare with root (which contains the largest distance/index in our top-k)
        if (dist < heap[0].dist || (dist == heap[0].dist && idx < heap[0].idx)) {
            heap[0] = {dist, idx};
            // Bubble down
            int i = 0;
            while (2 * i + 1 < k) {
                int left = 2 * i + 1;
                int right = 2 * i + 2;
                int largest = i;
                
                if (heap[left].dist > heap[largest].dist || (heap[left].dist == heap[largest].dist && heap[left].idx > heap[largest].idx)) {
                    largest = left;
                }
                if (right < k && (heap[right].dist > heap[largest].dist || (heap[right].dist == heap[largest].dist && heap[right].idx > heap[largest].idx))) {
                    largest = right;
                }
                
                if (largest != i) {
                    Candidate tmp = heap[i]; heap[i] = heap[largest]; heap[largest] = tmp;
                    i = largest;
                } else {
                    break;
                }
            }
        }
    }
}

__global__ void compute_and_local_select(const int* __restrict__ codes, const int* __restrict__ query, 
                                          unsigned long long* __restrict__ block_out_keys, int N, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load query to shared memory for fast retrieval across the block
    __shared__ int s_query[8];
    if (threadIdx.x < 8) {
        s_query[threadIdx.x] = query[threadIdx.x];
    }
    __syncthreads();

    // Local heap to track thread's top-k candidates
    Candidate local_heap[MAX_K];
    int heap_size = 0;

    // Process elements stride-by-stride
    for (int i = idx; i < N; i += gridDim.x * blockDim.x) {
        int dist = 0;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            dist += __popc(s_query[w] ^ codes[8 * i + w]);
        }
        insert_max_heap(local_heap, heap_size, k, dist, i);
    }

    // Now reduce all thread heaps inside the block to a shared pool, or write to global memory.
    // For simplicity and high scalability to handle extreme N safely:
    // Write out each thread's local heap directly to block results
    int base_out_idx = (blockIdx.x * blockDim.x + threadIdx.x) * k;
    for (int i = 0; i < heap_size; ++i) {
        unsigned long long packed_key = ((unsigned long long)local_heap[i].dist << 32) | (unsigned int)local_heap[i].idx;
        block_out_keys[base_out_idx + i] = packed_key;
    }
    // Pad out remaining if this thread found fewer than k elements (only happens if N < grid * block * k)
    for (int i = heap_size; i < k; ++i) {
        block_out_keys[base_out_idx + i] = 0xFFFFFFFFFFFFFFFFULL;
    }
}

// Bitonic Sort helper for the global final merge pass
__device__ inline void swap_keys(unsigned long long &a, unsigned long long &b) {
    if (a > b) {
        unsigned long long tmp = a; a = b; b = tmp;
    }
}

__global__ void final_global_merge(const unsigned long long* __restrict__ total_candidates, int* __restrict__ output, int total_elements, int k) {
    // Single block final gather or small multi-block depending on total size.
    // Given MAX_K = 128, a single block can sort the final narrowed down space easily using shared memory.
    __shared__ unsigned long long s_keys[2048]; // Adjust based on dynamic grid requirements

    int tid = threadIdx.x;
    Candidate local_heap[MAX_K];
    int heap_size = 0;

    // Each thread in this final collector reads a chunk of the block outputs
    for (int i = tid; i < total_elements; i += blockDim.x) {
        unsigned long long packed = total_candidates[i];
        if (packed != 0xFFFFFFFFFFFFFFFFULL) {
            int dist = (int)(packed >> 32);
            int idx = (int)(packed & 0xFFFFFFFFU);
            insert_max_heap(local_heap, heap_size, k, dist, idx);
        }
    }

    // Every thread now pushes its top candidates to a shared memory pool
    // To sort completely via a single block bitonic sort or sequential priority queue merge:
    // We collect all into shared memory
    extern __shared__ unsigned long long shared_pool[];
    
    // Linear insertion into shared array or bitonic sort
    // For exact sorting of the final answer up to k=128:
    // Thread 0 collects everything from thread heaps sequentially into a global/shared heap, 
    // or we sort globally using a parallel sorting network.
    
    // Quick alternative: Thread 0 merges the small number of results sequentially
    if (tid == 0) {
        Candidate final_heap[MAX_K];
        int final_size = 0;
        
        // This is safe and performant because the number of threads is typically small or we only loop up to k
        for (int t = 0; t < blockDim.x; ++t) {
            int base = t * k;
            // Since we can't look inside other thread registers directly safely without warp shuffles,
            // we have threads write their heaps to shared memory first:
        }
    }
    
    // Correct parallel method using shared memory:
    int base_shmem = tid * k;
    for (int i = 0; i < k; ++i) {
        if (i < heap_size) {
            unsigned long long packed_key = ((unsigned long long)local_heap[i].dist << 32) | (unsigned int)local_heap[i].idx;
            shared_pool[base_shmem + i] = packed_key;
        } else {
            shared_pool[base_shmem + i] = 0xFFFFFFFFFFFFFFFFULL;
        }
    }
    __syncthreads();

    // Now thread 0 can execute a single final pass over the shared_pool to extract the absolute top k
    if (tid == 0) {
        Candidate absolute_heap[MAX_K];
        int absolute_size = 0;
        int num_items = blockDim.x * k;
        
        for (int i = 0; i < num_items; ++i) {
            unsigned long long packed = shared_pool[i];
            if (packed != 0xFFFFFFFFFFFFFFFFULL) {
                int dist = (int)(packed >> 32);
                int idx = (int)(packed & 0xFFFFFFFFU);
                insert_max_heap(absolute_heap, absolute_size, k, dist, idx);
            }
        }

        // Sort the absolute max-heap ascending by extracting elements or using standard bubble sort
        for (int i = 0; i < k - 1; ++i) {
            for (int j = 0; j < k - i - 1; ++j) {
                if (absolute_heap[j].dist > absolute_heap[j+1].dist || 
                   (absolute_heap[j].dist == absolute_heap[j+1].dist && absolute_heap[j].idx > absolute_heap[j+1].idx)) {
                    Candidate tmp = absolute_heap[j];
                    absolute_heap[j] = absolute_heap[j+1];
                    absolute_heap[j+1] = tmp;
                }
            }
        }

        // Write to output array
        for (int i = 0; i < k; ++i) {
            output[i] = absolute_heap[i].idx;
        }
    }
}

extern "C" void solve(const int* codes, const int* query, int* output, int N, int k) {
    // Tailor grid dimensions for the performance evaluation size (N=4,000,000)
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = 512; 
    
    unsigned long long* d_block_out_keys;
    int total_candidates = blocksPerGrid * threadsPerBlock * k;
    cudaMalloc(&d_block_out_keys, total_candidates * sizeof(unsigned long long));

    // Pass 1: Compute distance and locate block-level candidate spaces
    compute_and_local_select<<<blocksPerGrid, threadsPerBlock>>>(codes, query, d_block_out_keys, N, k);
    cudaDeviceSynchronize();

    // Pass 2: Merge the block-level candidate lists globally into a final sorted top-k result
    size_t shared_mem_size = threadsPerBlock * k * sizeof(unsigned long long);
    final_global_merge<<<1, threadsPerBlock, shared_mem_size>>>(d_block_out_keys, output, total_candidates, k);
    cudaDeviceSynchronize();

    cudaFree(d_block_out_keys);
}
