import torch

def allocate_kv_blocks(
    seq_lengths: list,
    block_size: int,
    free_block_ids: list,
) -> tuple:
    """
    Allocates physical KV-cache blocks for a list of sequences following the PagedAttention paradigm.
    
    Returns:
        (block_table, blocks_used, remaining_free_blocks): A tuple of three integer PyTorch tensors.
    """
    # 1. Compute the number of blocks required for each sequence
    # n_i = ceil(L_i / P)
    blocks_used_list = [(l + block_size - 1) // block_size for l in seq_lengths]
    total_blocks_needed = sum(blocks_used_list)
    
    # 2. Check if the free pool has sufficient capacity
    if total_blocks_needed > len(free_block_ids):
        raise RuntimeError("Insufficient free blocks available for allocation.")
        
    N = len(seq_lengths)
    W = max(blocks_used_list) if N > 0 else 0
    
    # 3. Initialize the block table with padding value of -1
    block_table_meta = []
    
    # Track the current index in the free_block_ids list
    free_idx = 0
    
    for num_blocks in blocks_used_list:
        # Extract the exact block IDs needed for this sequence
        assigned_ids = free_block_ids[free_idx : free_idx + num_blocks]
        free_idx += num_blocks
        
        # Pad with -1 up to width W
        padded_row = assigned_ids + [-1] * (W - num_blocks)
        block_table_meta.append(padded_row)
        
    # 4. Extract remaining free block IDs
    remaining_ids = free_block_ids[free_idx:]
    
    # 5. Convert lists to integer PyTorch tensors
    block_table_tensor = torch.tensor(block_table_meta, dtype=torch.int64)
    blocks_used_tensor = torch.tensor(blocks_used_list, dtype=torch.int64)
    remaining_free_tensor = torch.tensor(remaining_ids, dtype=torch.int64)
    
    return block_table_tensor, blocks_used_tensor, remaining_free_tensor
