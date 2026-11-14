import numpy as np

def apply_mlm_mask(
    token_ids: np.ndarray, 
    mask_positions: np.ndarray, 
    replace_probs: np.ndarray, 
    random_tokens: np.ndarray, 
    mask_token_id: int = 103
) -> dict:
    """
    Returns masked_ids and labels as int64 arrays in a dictionary.
    """
    # Initialize masked_ids as a copy of token_ids and labels with -100
    masked_ids = token_ids.copy().astype(np.int64)
    labels = np.full_like(token_ids, -100, dtype=np.int64)
    
    # Store original token IDs at the selected mask positions
    labels[mask_positions] = token_ids[mask_positions]
    
    # Extract values at mask_positions
    probs = replace_probs[mask_positions]
    orig_tokens = token_ids[mask_positions]
    rand_tokens = random_tokens[mask_positions]
    
    # Apply 80-10-10 rule on selected positions:
    # 1. r_i < 0.8 -> mask_token_id (80%)
    # 2. 0.8 <= r_i < 0.9 -> random token (10%)
    # 3. r_i >= 0.9 -> keep original token_ids (10%)
    new_masked_vals = np.where(
        probs < 0.8,
        mask_token_id,
        np.where(probs < 0.9, rand_tokens, orig_tokens)
    ).astype(np.int64)
    
    masked_ids[mask_positions] = new_masked_vals
    
    return {
        "masked_ids": masked_ids,
        "labels": labels
    }
