import jax
import jax.numpy as jnp

def beam_search_step(beam_scores: jax.Array, log_probs: jax.Array, histories: jax.Array, cache: dict) -> dict:
    """
    Returns a dict with scores (B, K) float32, tokens and parents (B, K) int32, 
    histories (B, K, L + 1) int32, and cache reordered by parent.
    """
    B, K, V = log_probs.shape
    L = histories.shape[-1]
    
    # 1. Compute total scores for all K x V candidates per batch row
    candidate_scores = beam_scores[:, :, None] + log_probs
    flat_scores = candidate_scores.reshape(B, K * V)
    
    # 2. Select the top K candidates
    scores, topk_indices = jax.lax.top_k(flat_scores, K)
    
    # 3. Recover parent beam indices and the chosen tokens
    parents = topk_indices // V
    tokens = topk_indices % V
    
    # 4. Update the histories by gathering from parents and appending new tokens
    parents_hist = jnp.broadcast_to(parents[:, :, None], (B, K, L))
    gathered_histories = jnp.take_along_axis(histories, parents_hist, axis=1)
    new_histories = jnp.concatenate([gathered_histories, tokens[:, :, None]], axis=-1)
    
    # 5. Reorder the cache arrays according to the selected parent beams
    new_cache = {}
    for key, arr in cache.items():
        ndim = arr.ndim
        # Match the rank of the cache array by appending trailing dimensions
        parents_expanded = parents
        for _ in range(ndim - 2):
            parents_expanded = parents_expanded[:, :, None]
            
        parents_broadcasted = jnp.broadcast_to(parents_expanded, arr.shape)
        new_cache[key] = jnp.take_along_axis(arr, parents_broadcasted, axis=1)
        
    return {
        "scores": scores,
        "tokens": tokens,
        "parents": parents,
        "histories": new_histories,
        "cache": new_cache
    }
