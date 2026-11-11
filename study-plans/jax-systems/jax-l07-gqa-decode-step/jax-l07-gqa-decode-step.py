import jax
import jax.numpy as jnp

def gqa_decode_step(query: jax.Array, key_cache: jax.Array, value_cache: jax.Array, lengths: jax.Array) -> jax.Array:
    """
    Returns the float32 attention output of shape (B, Hq, D) for the new token.
    """
    B, Hq, D = query.shape
    _, S, Hkv, _ = key_cache.shape
    G = Hq // Hkv
    
    # Cast caches to float32 as required
    key_cache = key_cache.astype(jnp.float32)
    value_cache = value_cache.astype(jnp.float32)
    
    # Split query heads (Hq) into group counts (Hkv) and heads per group (G)
    q = query.reshape(B, Hkv, G, D)
    
    # Transpose caches to align heads first: (B, Hkv, S, D)
    k = jnp.transpose(key_cache, (0, 2, 1, 3))
    v = jnp.transpose(value_cache, (0, 2, 1, 3))
    
    # Compute attention scores using broadcasting: (B, Hkv, G, S)
    scores = jnp.sum(q[:, :, :, None, :] * k[:, :, None, :, :], axis=-1) / jnp.sqrt(D)
    
    # Create sequence-length boolean masks
    slots = jnp.arange(S)
    mask = slots[None, :] < lengths[:, None]  # (B, S)
    mask = mask[:, None, None, :]  # (B, 1, 1, S)
    
    # Apply mask with a large negative value before softmax
    scores = jnp.where(mask, scores, -1e9)
    weights = jax.nn.softmax(scores, axis=-1)
    
    # Compute the weighted sum over value vectors: (B, Hkv, G, D)
    out = jnp.sum(weights[:, :, :, :, None] * v[:, :, None, :, :], axis=-2)
    
    # Merge Hkv and G back to match the original (B, Hq, D) query format
    return out.reshape(B, Hq, D)
