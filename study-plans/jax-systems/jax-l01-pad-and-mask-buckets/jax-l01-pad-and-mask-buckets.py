import jax
import jax.numpy as jnp

@jax.jit(static_argnames=['L'])
def _pad_to_bucket_compiled(tokens: jax.Array, lengths: jax.Array, pad_id: int, L: int) -> dict:
    # Compute start offsets for each sequence: s_b = sum_{c=0}^{b-1} lengths[c]
    offsets = jnp.cumsum(lengths) - lengths
    
    # Create a 2D grid of absolute flat indices of shape (B, L)
    idx = offsets[:, None] + jnp.arange(L)[None, :]
    
    # Create the boolean validity mask based on sequence lengths
    mask = jnp.arange(L)[None, :] < lengths[:, None]
    
    # Clip indices to prevent out-of-bounds errors on tokens (values outside the mask are ignored anyway)
    clipped_idx = jnp.clip(idx, 0, tokens.shape[0] - 1)
    
    # Gather the values and substitute pad_id where mask is False
    gathered_tokens = tokens[clipped_idx]
    output_tokens = jnp.where(mask, gathered_tokens, pad_id)
    
    return {"tokens": output_tokens, "mask": mask}

def pad_to_bucket(tokens: jax.Array, lengths: jax.Array, buckets: list[int], pad_id: int) -> dict:
    """
    Returns a dict with tokens, an int32 JAX array of shape (B, L), and mask, a bool JAX array of shape (B, L).
    """
    # 1. Pull the maximum sequence length to the host
    max_len = int(jnp.max(lengths))
    
    # 2. Find the smallest bucket length that can fit the longest sequence
    L = min(b for b in buckets if b >= max_len)
    
    # 3. Call the module-level jitted function with L as a static argument
    return _pad_to_bucket_compiled(tokens, lengths, pad_id, L=L)
