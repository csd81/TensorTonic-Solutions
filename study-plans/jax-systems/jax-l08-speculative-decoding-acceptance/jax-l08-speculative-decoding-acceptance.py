import jax
import jax.numpy as jnp

def speculative_accept(
    draft_tokens: jax.Array, 
    draft_probs: jax.Array, 
    target_probs: jax.Array, 
    accept_uniforms: jax.Array, 
    sample_uniforms: jax.Array
) -> dict:
    """
    Decides which draft tokens one speculative decoding step keeps, using the target 
    and draft probabilities with the supplied uniform noise, and picks the correction 
    token from the residual distribution.
    """
    B, G = draft_tokens.shape
    
    # 1. Extract target and draft probabilities for the selected draft tokens
    batch_idx = jnp.arange(B)[:, None]
    pos_idx = jnp.arange(G)[None, :]
    q_xi = draft_probs[batch_idx, pos_idx, draft_tokens]
    p_xi = target_probs[batch_idx, pos_idx, draft_tokens]
    
    # 2. Determine individual acceptances and compute the prefix sequence match
    accepted_mask = accept_uniforms < (p_xi / q_xi)
    prefix_accepted = jnp.cumprod(accepted_mask.astype(jnp.int32), axis=-1)
    num_accepted = jnp.sum(prefix_accepted, axis=-1)
    
    # 3. Formulate the fallback distribution for all possible index options (0 to G)
    res = jnp.maximum(target_probs[:, :G, :] - draft_probs, 0.0)
    res = res / jnp.sum(res, axis=-1, keepdims=True)
    
    res_G = target_probs[:, G:G+1, :]
    res_G = res_G / jnp.sum(res_G, axis=-1, keepdims=True)
    
    R = jnp.concatenate([res, res_G], axis=1)
    
    # 4. Dynamically select the exact fallback distribution based on num_accepted
    r_selected = jnp.take_along_axis(R, num_accepted[:, None, None], axis=1)[:, 0, :]
    
    # 5. Sample the replacement token using cumulative probabilities
    r_cum = jnp.cumsum(r_selected, axis=-1)
    y = jnp.sum((r_cum <= sample_uniforms[:, None]).astype(jnp.int32), axis=-1)
    
    # 6. Assemble the resulting tokens array with masking layout
    idx = jnp.arange(G + 1)
    draft_padded = jnp.pad(draft_tokens, ((0, 0), (0, 1)), constant_values=0)
    
    mask_accepted = idx[None, :] < num_accepted[:, None]
    mask_y = idx[None, :] == num_accepted[:, None]
    
    tokens = jnp.where(mask_accepted, draft_padded, -1)
    tokens = jnp.where(mask_y, y[:, None], tokens)
    
    return {
        "num_accepted": num_accepted.astype(jnp.int32), 
        "tokens": tokens.astype(jnp.int32)
    }
