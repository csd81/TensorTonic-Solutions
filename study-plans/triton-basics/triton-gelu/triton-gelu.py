import torch
import triton
import triton.language as tl

@triton.jit
def gelu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Determine the program's spatial coordinate
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to prevent out-of-bounds memory access
    mask = offsets < n
    
    # Load input vector from global memory
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Exact GELU computation: 0.5 * x * (1 + erf(x * 0.7071067811865476))
    out = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    
    # Store the result back to global memory
    tl.store(out_ptr + offsets, out, mask=mask)

def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch gelu_kernel: out = 0.5 * x * (1 + erf(x / sqrt(2)))."""
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    gelu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
