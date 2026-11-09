import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Calculate the block offset for this specific program instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle boundary conditions safely
    mask = offsets < n
    
    # Load the elements from memory, masking out-of-bounds positions
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Apply the ReLU operation: out = max(x, 0)
    output = tl.maximum(x, 0.0)
    
    # Store the result back into the output tensor
    tl.store(out_ptr + offsets, output, mask=mask)

def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch relu kernel; out = max(x, 0)."""
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    relu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
