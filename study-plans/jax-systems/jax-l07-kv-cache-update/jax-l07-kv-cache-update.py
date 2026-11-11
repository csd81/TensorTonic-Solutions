import jax
import jax.numpy as jnp
from jax import lax

def kv_cache_update(key_cache: jax.Array, value_cache: jax.Array, new_keys: jax.Array, new_values: jax.Array, position: jax.Array) -> dict:
    """
    Returns a dict with key_cache and value_cache (same shapes and dtypes as the inputs) and length, an int32 scalar.
    """
    # Cast incoming float32 tokens to match the cache's dtype (float32 or bfloat16)
    new_keys_cast = new_keys.astype(key_cache.dtype)
    new_values_cast = new_values.astype(value_cache.dtype)
    
    # Start indices for each dimension: (Batch, Sequence, Heads, Features)
    start_indices = (0, position, 0, 0)
    
    # Update the caches dynamically along the sequence dimension
    updated_key_cache = lax.dynamic_update_slice(key_cache, new_keys_cast, start_indices)
    updated_value_cache = lax.dynamic_update_slice(value_cache, new_values_cast, start_indices)
    
    # Compute the new total filled sequence length (position + L)
    L = new_keys.shape[1]
    length = jnp.array(position + L, dtype=jnp.int32)
    
    return {
        "key_cache": updated_key_cache,
        "value_cache": updated_value_cache,
        "length": length
    }
