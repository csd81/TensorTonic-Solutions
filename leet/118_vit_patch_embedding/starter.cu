#include <cuda_runtime.h>

// CUDA Kernel for ViT Patch Embedding Stem
__global__ void patch_embedding_kernel(
    const float* __restrict__ images,
    const float* __restrict__ patch_weight,
    const float* __restrict__ patch_bias,
    const float* __restrict__ cls_token,
    const float* __restrict__ pos_embed,
    float* __restrict__ output,
    int B, int C, int H, int W, int P, int D, int g_w, int N) 
{
    // target token index within the sequence (0 to N, where 0 is CLS token)
    int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // target dimension index
    int d = blockIdx.y * blockDim.y + threadIdx.y;
    // target batch index
    int b = blockIdx.z;

    if (b >= B || d >= D || seq_idx > N) return;

    float val = 0.0f;

    if (seq_idx == 0) {
        // Row 0 is the CLS Token + Positional Embedding 0
        val = cls_token[d] + pos_embed[d];
    } else {
        // Patch tokens (index 1 to N mapped to patch n = seq_idx - 1)
        int n = seq_idx - 1;
        int p_y = n / g_w;
        int p_x = n % g_w;
        
        // Calculate base coordinates in the image
        int row_start = p_y * P;
        int col_start = p_x * P;

        // Start with the patch bias for this dimension
        val = patch_bias[d];

        // Perform 2D convolution over the patch region across all channels
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < P; ++i) {
                int img_row = row_start + i;
                // Precompute base indices for memory indexing
                int img_base_idx = ((b * C + c) * H + img_row) * W + col_start;
                int weight_base_idx = ((d * C + c) * P + i) * P;

                for (int j = 0; j < P; ++j) {
                    float img_pixel = images[img_base_idx + j];
                    float w = patch_weight[weight_base_idx + j];
                    val += img_pixel * w;
                }
            }
        }
        // Add corresponding positional embedding
        val += pos_embed[seq_idx * D + d];
    }

    // Write directly in-place to row-major output: (B, N+1, D)
    int out_idx = (b * (N + 1) + seq_idx) * D + d;
    output[out_idx] = val;
}

extern "C" void solve(const float* images, const float* patch_weight, const float* patch_bias,
                      const float* cls_token, const float* pos_embed, float* output, int B, int C,
                      int H, int W, int P, int D) 
{
    int g_h = H / P;
    int g_w = W / P;
    int N = g_h * g_w;
    int num_sequences = N + 1; // Including the CLS Token

    // Configure 2D thread block size
    dim3 threadsPerBlock(16, 16, 1);
    
    // Configure 3D grid size: x maps to tokens, y maps to D, z maps to Batch
    dim3 numBlocks(
        (num_sequences + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (D + threadsPerBlock.y - 1) / threadsPerBlock.y,
        B
    );

    // Launch the fused execution kernel
    patch_embedding_kernel<<<numBlocks, threadsPerBlock>>>(
        images, patch_weight, patch_bias, cls_token, pos_embed, output,
        B, C, H, W, P, D, g_w, N
    );
}
