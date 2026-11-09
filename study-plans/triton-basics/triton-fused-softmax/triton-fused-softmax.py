import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(x_ptr, out_ptr, x_row_stride, out_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # Map this program instance to its specific row index
    row_idx = tl.program_id(axis=0)
    
    # Locate the starting point of the row in global memory
    row_start_ptr_x = x_ptr + row_idx * x_row_stride
    row_start_ptr_out = out_ptr + row_idx * out_row_stride
    
    # Compute column offsets within the block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Create the column tail boundary mask
    mask = col_offsets < n_cols
    
    # 1. Load row values into registers. 
    # Use -inf for out-of-bounds columns so they lose max comparisons.
    x = tl.load(row_start_ptr_x + col_offsets, mask=mask, other=-float('inf'))
    
    # 2. Subtract row max for numerical stability (prevents float32 overflow)
    row_max = tl.max(x, axis=0)
    x_shifted = x - row_max
    
    # 3. Exponentiate the shifted logits
    numerator = tl.exp(x_shifted)
    
    # 4. Sum the exponents across the row to find the normalization denominator
    # Masked lanes evaluate to exp(-inf) = 0.0, contributing nothing to the sum.
    denominator = tl.sum(numerator, axis=0)
    
    # 5. Normalize to compute the final softmax probabilities
    out = numerator / denominator
    
    # Store the results back to the output row in global memory using the mask
    tl.store(row_start_ptr_out + col_offsets, out, mask=mask)

def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch softmax_kernel with one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    softmax_kernel[grid](
        x, out, x.stride(0), out.stride(0), N, BLOCK_SIZE=BLOCK_SIZE,
    )
