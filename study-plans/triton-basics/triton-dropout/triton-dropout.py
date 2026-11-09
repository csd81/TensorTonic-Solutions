import torch
import triton
import triton.language as tl


@triton.jit
def dropout_kernel(
    x_ptr, mask_ptr, out_ptr,
    n, p,
    BLOCK_SIZE: tl.constexpr,
):
    # 1. Identify which block this program instance handles
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # 2. Generate element offsets for the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 3. Create a guard mask to prevent out-of-bounds memory access
    mask_boundary = offsets < n
    
    # 4. Load the input values and the externally provided 0/1 mask
    x = tl.load(x_ptr + offsets, mask=mask_boundary)
    m = tl.load(mask_ptr + offsets, mask=mask_boundary)
    
    # 5. Compute inverted scaling dropout formula
    scale = 1.0 / (1.0 - p)
    output = x * m * scale
    
    # 6. Store the results back into global memory
    tl.store(out_ptr + offsets, output, mask=mask_boundary)



def solve(x: torch.Tensor, mask: torch.Tensor, out: torch.Tensor, p: float) -> None:
    """Launch the dropout kernel: 1D grid over the input vector."""
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    dropout_kernel[grid](
        x, mask, out,
        n, p,
        BLOCK_SIZE=BLOCK_SIZE,
    )