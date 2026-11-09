import torch
import triton
import triton.language as tl


@triton.jit
def silu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Identify which block of elements this program instance is responsible for
    pid = tl.program_id(axis=0)
    
    # Compute the elements' memory offsets for this block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to safely handle inputs where N is not a multiple of BLOCK_SIZE
    mask = offsets < n
    
    # Load input data using the mask
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute SiLU in a single fused expression using Triton's exponential function
    out = x / (1.0 + tl.exp(-x))
    
    # Store the calculated values back to memory using the mask
    tl.store(out_ptr + offsets, out, mask=mask)



def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch silu_kernel: out = x / (1 + exp(-x))."""
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    silu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)