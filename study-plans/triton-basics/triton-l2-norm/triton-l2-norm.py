import torch
import triton
import triton.language as tl


@triton.jit
def l2_norm_kernel(x_ptr, sumsq_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Map this program instance to its slice of the 1D tensor
    pid = tl.program_id(axis=0)
    
    # Compute the global memory offsets for this block's elements
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create a boundary mask to prevent out-of-bounds reads
    mask = offsets < n
    
    # Load input values. Tail lanes must default to 0.0 so their squares add nothing.
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Square the elements in registers before performing the reduction
    sq = x * x
    
    # Reduce the squared values across the entire block dimension
    block_sumsq = tl.sum(sq, axis=0)
    
    # Atomically accumulate this program's partial sum of squares into the buffer
    tl.atomic_add(sumsq_ptr, block_sumsq)



def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch l2_norm_kernel and finalize the square root."""
    n = x.numel()
    sumsq_buf = torch.zeros(1, device='cuda', dtype=torch.float32)
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    l2_norm_kernel[grid](x, sumsq_buf, n, BLOCK_SIZE=BLOCK_SIZE)
    out.copy_(torch.sqrt(sumsq_buf))