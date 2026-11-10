import torch

def match_prefix_cache(
    request_token_ids: list,
    cached_token_blocks: list,
    cached_physical_block_ids: list,
    block_size: int,
) -> tuple:
    """
    Finds the cached candidate sharing the longest whole-block prefix with a new request.
    
    Returns a tuple containing:
      - reusable_token_count: A scalar integer PyTorch tensor.
      - reusable_physical_block_ids: A 1D integer PyTorch tensor of physical IDs.
    """
    # 1. Break the request token IDs into complete blocks
    num_request_blocks = len(request_token_ids) // block_size
    request_blocks = []
    for i in range(num_request_blocks):
        start_idx = i * block_size
        end_idx = start_idx + block_size
        request_blocks.append(request_token_ids[start_idx:end_idx])
        
    max_matched_blocks = 0
    best_candidate_idx = -1
    
    # 2. Iterate through each candidate to find the longest matching prefix
    for cand_idx, cand_blocks in enumerate(cached_token_blocks):
        matched_blocks = 0
        # Check consecutive blocks starting from index 0
        max_possible_check = min(len(request_blocks), len(cand_blocks))
        
        for b_idx in range(max_possible_check):
            if request_blocks[b_idx] == cand_blocks[b_idx]:
                matched_blocks += 1
            else:
                break  # Stop at the first mismatch
        
        # Tie-breaker: Equal-length matches favor the lowest candidate index (inherent in >)
        if matched_blocks > max_matched_blocks:
            max_matched_blocks = matched_blocks
            best_candidate_idx = cand_idx

    # 3. Construct the output PyTorch tensors
    if max_matched_blocks == 0 or best_candidate_idx == -1:
        reusable_tokens = torch.tensor(0, dtype=torch.long)
        reusable_block_ids = torch.tensor([], dtype=torch.long)
    else:
        reusable_tokens = torch.tensor(max_matched_blocks * block_size, dtype=torch.long)
        # Extract the matching subset of physical block IDs for the winning candidate
        matching_ids = cached_physical_block_ids[best_candidate_idx][:max_matched_blocks]
        reusable_block_ids = torch.tensor(matching_ids, dtype=torch.long)
        
    return (reusable_tokens, reusable_block_ids)
