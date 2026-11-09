import torch
import triton
import triton.language as tl



@triton.jit
def mean_var_kernel(x_ptr, sum_ptr, sumsq_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Map this program instance to its slice of the 1D tensor
    pid = tl.program_id(axis=0)
    
    # Compute the elements' global memory offsets for this block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create a boundary mask to prevent out-of-bounds reads
    mask = offsets < n
    
    # Load input values. Masked elements default to 0.0 to prevent changing the sums.
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Two-statistic fusion: compute both block reductions in registers
    block_sum = tl.sum(x, axis=0)
    block_sumsq = tl.sum(x * x, axis=0)
    
    # Atomically accumulate both partial statistics into the scratch buffers
    tl.atomic_add(sum_ptr, block_sum)
    tl.atomic_add(sumsq_ptr, block_sumsq)


def solve(x: torch.Tensor, mean_out: torch.Tensor, var_out: torch.Tensor) -> None:
    """Launch mean_var_kernel and finalize mean and variance."""
    n = x.numel()
    sum_buf = torch.zeros(1, device='cuda', dtype=torch.float32)
    sumsq_buf = torch.zeros(1, device='cuda', dtype=torch.float32)
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    mean_var_kernel[grid](x, sum_buf, sumsq_buf, n, BLOCK_SIZE=BLOCK_SIZE)
    mean = sum_buf / n
    var = sumsq_buf / n - mean * mean
    mean_out.copy_(mean)
    var_out.copy_(var)