import jax
import jax.numpy as jnp

def masked_mean(x: jax.Array, mask: jax.Array, axis: int) -> jax.Array:
    """
    Returns a float32 array with the shape of x without the given axis, holding the mean of the valid entries.
    """
    # 1. Replace invalid entries (including potential NaNs/Infs) with 0.0
    safe_x = jnp.where(mask, x, 0.0)
    
    # 2. Sum the valid entries along the specified axis
    total_sum = jnp.sum(safe_x, axis=axis)
    
    # 3. Count the number of valid entries along the axis
    valid_count = jnp.sum(mask, axis=axis)
    
    # 4. Perform a safe division, replacing 0 counts with 1 to avoid NaN/division-by-zero
    # Since total_sum is 0.0 when valid_count is 0, dividing by 1.0 correctly yields 0.0
    safe_count = jnp.where(valid_count > 0, valid_count, 1.0)
    
    return total_sum / safe_count
