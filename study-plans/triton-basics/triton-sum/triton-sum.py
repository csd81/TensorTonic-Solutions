import torch
import triton
import triton.language as tl


@triton.jit
def sum_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Determine which tile/block of elements this program instance is processing
    pid = tl.program_id(axis=0)
    
    # Compute the global memory offsets for this block's elements
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create a boundary mask to prevent out-of-bounds memory accesses
    mask = offsets < n
    
    # Load data from global memory; out-of-bounds lanes default to 0.0 so they don't impact the sum
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Reduce the elements in registers along the single block dimension
    block_sum = tl.sum(x, axis=0)
    
    # Atomically add this block's partial sum to the final output scalar
    tl.atomic_add(out_ptr, block_sum)



def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch sum_kernel on the provided tensors."""
    n = x.numel()
    out.zero_()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    sum_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)