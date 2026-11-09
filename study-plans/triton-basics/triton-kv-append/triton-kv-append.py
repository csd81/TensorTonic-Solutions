import torch
import triton
import triton.language as tl


@triton.jit
def kv_append_kernel(
    k_new_ptr, v_new_ptr, k_cache_ptr, v_cache_ptr,
    pos, D,
    BLOCK_SIZE: tl.constexpr,
):
    # 1. Identify the 1D program ID along the D dimension
    pid_d = tl.program_id(axis=0)
    
    # 2. Compute the offset range for this tile
    offsets = pid_d * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 3. Create a mask to prevent out-of-bounds memory access if D is not a multiple of BLOCK_SIZE
    mask = offsets < D
    
    # 4. Compute 1D pointers for the incoming 1D token tensors k_new and v_new
    k_new_offsets = k_new_ptr + offsets
    v_new_offsets = v_new_ptr + offsets
    
    # 5. Load the new key and value vector tokens using the mask
    k_val = tl.load(k_new_offsets, mask=mask, other=0.0)
    v_val = tl.load(v_new_offsets, mask=mask, other=0.0)
    
    # 6. Compute 2D row/column offsets for the pre-allocated global cache matrices
    # Row offset targets the specific time-step 'pos'. Column offset spans the current block.
    k_cache_offsets = k_cache_ptr + pos * D + offsets
    v_cache_offsets = v_cache_ptr + pos * D + offsets
    
    # 7. Write the values into row 'pos' of the pre-allocated caches in place
    tl.store(k_cache_offsets, k_val, mask=mask)
    tl.store(v_cache_offsets, v_val, mask=mask)



def solve(
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    pos: int,
) -> None:
    """Write k_new and v_new into row `pos` of k_cache and v_cache in place."""
    D = k_new.numel()
    BLOCK_SIZE = 1024
    grid = ((D + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    kv_append_kernel[grid](
        k_new, v_new, k_cache, v_cache,
        pos, D,
        BLOCK_SIZE=BLOCK_SIZE,
    )