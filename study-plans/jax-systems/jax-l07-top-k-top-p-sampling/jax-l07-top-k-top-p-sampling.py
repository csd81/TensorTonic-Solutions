import jax
import jax.numpy as jnp
from jax import lax

def sample_top_k_top_p(logits: jax.Array, uniform: jax.Array, temperature: float, top_k: int, top_p: float) -> jax.Array:
    """
    Returns an int32 array of shape (B,) holding one sampled token index per row.
    """
    # 1. Scale logits by temperature
    scaled_logits = logits / temperature
    
    # 2. Top-K Filtering
    # Find the threshold value for top_k elements in each row
    # jnp.top_k returns values and indices; the last value in the top_k sorted row is our threshold
    top_k_vals, _ = jax.lax.top_k(scaled_logits, top_k)
    kth_vals = top_k_vals[:, -1:]  # Shape (B, 1)
    
    # Mask out non-top-k logits by setting them to -inf
    top_k_mask = scaled_logits >= kth_vals
    logits_masked_k = jnp.where(top_k_mask, scaled_logits, -jnp.inf)
    
    # 3. Softmax over the kept tokens
    probs = jax.nn.softmax(logits_masked_k, axis=-1)
    
    # 4. Top-P (Nucleus) Filtering
    # Sort probabilities in descending order to calculate running cumulative probability
    sort_indices = jnp.argsort(probs, axis=-1)[:, ::-1]
    sorted_probs = jnp.take_along_axis(probs, sort_indices, axis=-1)
    
    # Cumulative sum of sorted tokens shifted to find the total probability *before* each token
    cumsum_probs = jnp.cumsum(sorted_probs, axis=-1)
    prior_cumsum = cumsum_probs - sorted_probs
    
    # A token stays if the prior cumsum is strictly less than top_p
    nucleus_mask_sorted = prior_cumsum < top_p
    
    # Scatter the mask back to the original vocabulary index order
    # (Inverse permutation of sort_indices)
    inv_sort_indices = jnp.argsort(sort_indices, axis=-1)
    nucleus_mask = jnp.take_along_axis(nucleus_mask_sorted, inv_sort_indices, axis=-1)
    
    # 5. Renormalize nucleus probabilities
    final_probs = jnp.where(nucleus_mask & top_k_mask, probs, 0.0)
    final_probs = final_probs / jnp.sum(final_probs, axis=-1, keepdims=True)
    
    # 6. Inverse-CDF Sampling along vocabulary index order (0, 1, ..., V-1)
    running_cdf = jnp.cumsum(final_probs, axis=-1)
    
    # Find the smallest index where running_cdf > uniform
    # Strict comparison guarantees 0-probability elements are not chosen
    is_greater = running_cdf > uniform[:, None]
    
    # Map True to index and False to a large number (V), then take the minimum
    v_indices = jnp.arange(logits.shape[-1], dtype=jnp.int32)
    selected_indices = jnp.where(is_greater, v_indices, logits.shape[-1])
    
    return jnp.min(selected_indices, axis=-1)
