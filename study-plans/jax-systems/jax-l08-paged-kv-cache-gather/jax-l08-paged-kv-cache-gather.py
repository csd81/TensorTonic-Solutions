import jax
import jax.numpy as jnp

def paged_kv_gather(k_pages: jax.Array, v_pages: jax.Array, block_tables: jax.Array, seq_lens: jax.Array) -> dict:
    """
    Returns a dict with keys and values of shape (B, max_blocks * page_size, num_heads, head_dim) 
    and a bool mask of shape (B, max_blocks * page_size).
    """
    B, max_blocks = block_tables.shape
    num_pages, page_size, num_heads, head_dim = k_pages.shape
    L = max_blocks * page_size
    
    # 1. Generate position (j) indices for the target sequence length L
    j = jnp.arange(L)[None, :]  # Shape: (1, L)
    
    # 2. Compute the corresponding block and slot indices for each position
    block_idx = j // page_size  # Shape: (1, L)
    slot_idx = j % page_size    # Shape: (1, L)
    
    # 3. Retrieve the page indices from the block table for every batch element
    b_idx = jnp.arange(B)[:, None]  # Shape: (B, 1)
    page_idx = block_tables[b_idx, block_idx]  # Shape: (B, L)
    
    # 4. Safely clamp negative/invalid page indices to prevent out-of-bounds indexing
    safe_page_idx = jnp.where(page_idx >= 0, page_idx, 0)
    slot_idx_broadcast = jnp.broadcast_to(slot_idx, (B, L))
    
    # 5. Extract keys and values using JAX advanced indexing
    gathered_keys = k_pages[safe_page_idx, slot_idx_broadcast]    # Shape: (B, L, num_heads, head_dim)
    gathered_values = v_pages[safe_page_idx, slot_idx_broadcast]  # Shape: (B, L, num_heads, head_dim)
    
    # 6. Build the sequence validity mask
    mask = j < seq_lens[:, None]  # Shape: (B, L)
    
    # 7. Zero-out positions that fall outside the actual sequence length
    keys = jnp.where(mask[:, :, None, None], gathered_keys, 0)
    values = jnp.where(mask[:, :, None, None], gathered_values, 0)
    
    return {
        "keys": keys,
        "values": values,
        "mask": mask
    }
