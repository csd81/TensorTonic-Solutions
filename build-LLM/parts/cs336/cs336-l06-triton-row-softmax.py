import torch
import triton
import triton.language as tl

@triton.jit
def row_softmax_kernel(x_ptr, out_ptr, row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # Map 1D program ID to the row index
    row_idx = tl.program_id(0)
    
    # Generate column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Compute memory pointers for the current row
    row_ptr = x_ptr + row_idx * row_stride + col_offsets
    # FIXED: out is contiguous, so its row stride is exactly n_cols
    out_row_ptr = out_ptr + row_idx * n_cols + col_offsets
    
    # Load data with masking (pad out-of-bound elements with -inf)
    logits = tl.load(row_ptr, mask=mask, other=-float("inf")).to(tl.float32)
    
    # Numerically stable softmax: subtract max value along the row
    max_val = tl.max(logits, axis=0)
    shifted_logits = logits - max_val
    
    # Exponentiate and sum online
    numerators = tl.exp(shifted_logits)
    sum_exp = tl.sum(numerators, axis=0)
    
    # Compute final softmax probabilities
    probabilities = numerators / sum_exp
    
    # Store the results back to global memory
    tl.store(out_row_ptr, probabilities, mask=mask)

def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError("x must be a nonempty-width matrix")
    if x.shape != out.shape or x.device != out.device or x.dtype != out.dtype:
        raise ValueError("x and out must have matching shape, device, and dtype")
    if not x.is_cuda or x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("x must be a supported CUDA tensor")
    if x.stride(1) != 1 or not out.is_contiguous():
        raise ValueError("input columns and output must be contiguous")
    if x.shape[1] > 65536:
        raise ValueError("row width exceeds the supported limit")
    if x.shape[0] == 0:
        return
    kernel_x = x
    kernel_out = out
    if x.dtype == torch.bfloat16:
        kernel_x = x.to(torch.float32)
        kernel_out = torch.empty_like(x, dtype=torch.float32)
    block_size = triton.next_power_of_2(x.shape[1])
    grid = (x.shape[0],)
    row_softmax_kernel[grid](
        kernel_x,
        kernel_out,
        kernel_x.stride(0),
        x.shape[1],
        BLOCK_SIZE=block_size,
    )
    if kernel_out is not out:
        out.copy_(kernel_out)
