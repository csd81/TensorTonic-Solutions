import torch
import triton
import triton.language as tl


@triton.jit
def vectorized_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Identify which block this program instance is responsible for
    pid = tl.program_id(axis=0)
    
    # Compute the memory offsets for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle arrays whose sizes are not perfectly divisible by BLOCK_SIZE
    mask = offsets < n
    
    # Load the vectors from global memory using the offsets and mask
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Perform the elementwise addition
    output = x + y
    
    # Store the result back into global memory
    tl.store(out_ptr + offsets, output, mask=mask)



def solve(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> None:
    """Launch vectorized_add_kernel with a large BLOCK_SIZE for long arrays."""
    n = x.numel()
    BLOCK_SIZE = 4096
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    vectorized_add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)