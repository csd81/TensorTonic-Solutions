import torch
import triton
import triton.language as tl

@triton.jit
def max_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This problem uses a single-program reduction, so program_id is 0
    # and offsets span the full padded BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Create the boundary mask for elements within the array length
    mask = offsets < n
    
    # Load elements. Masked lanes must use -inf so they lose all comparisons.
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Perform the register reduction across the entire block block axis
    result = tl.max(x, axis=0)
    
    # Store the single scalar value directly into out[0]
    tl.store(out_ptr, result)

def solve(x: torch.torch.Tensor, out: torch.Tensor) -> None:
    """Launch max_kernel on the provided tensor with a single-program reduction."""
    n = x.numel()
    BLOCK_SIZE = triton.next_power_of_2(n)
    grid = (1,)
    max_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
