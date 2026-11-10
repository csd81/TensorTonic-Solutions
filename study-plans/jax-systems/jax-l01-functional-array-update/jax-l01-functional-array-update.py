import jax
import jax.numpy as jnp

def weighted_histogram(counts: jax.Array, indices: jax.Array, weights: jax.Array) -> jax.Array:
    """
    Returns a new float32 array of shape (K,) holding counts plus every weight added into the bin named by its index, with out-of-range indices ignored.
    """
    K = counts.shape[0]
    
    # 1. Create a boolean mask identifying indices within the valid range [0, K - 1]
    is_valid = (indices >= 0) & (indices < K)
    
    # 2. Replace out-of-bounds indices with a safe index (0) so JAX doesn't error or clip unpredictably
    safe_indices = jnp.where(is_valid, indices, 0)
    
    # 3. Force the weights of invalid updates to 0.0 so they don't alter the safe index's bin
    safe_weights = jnp.where(is_valid, weights, 0.0)
    
    # 4. Use non-mutating array updates to accumulate the weights into the histogram
    return counts.at[safe_indices].add(safe_weights)
