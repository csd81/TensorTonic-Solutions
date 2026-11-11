import jax
import jax.numpy as jnp
from jax import lax

def blockwise_attention(q: jax.Array, k: jax.Array, v: jax.Array, block_size: int, causal: bool) -> jax.Array:
    """
    Returns the float32 attention output of shape (Tq, d).
    """
    Tq, d_dim = q.shape
    Tk, _ = k.shape
    num_blocks = Tk // block_size
    
    # Reshape k and v into blocks of shape (num_blocks, block_size, d_dim)
    k_blocks = k.reshape(num_blocks, block_size, d_dim)
    v_blocks = v.reshape(num_blocks, block_size, d_dim)
    
    # Initialize online softmax statistics:
    # m: running max initialized to -inf, shape (Tq, 1)
    # d: running denominator sum initialized to 0, shape (Tq, 1)
    # o: running output numerator initialized to 0, shape (Tq, d_dim)
    init_m = jnp.full((Tq, 1), -jnp.inf, dtype=jnp.float32)
    init_d = jnp.zeros((Tq, 1), dtype=jnp.float32)
    init_o = jnp.zeros((Tq, d_dim), dtype=jnp.float32)
    
    initial_carry = (init_m, init_d, init_o)
    
    # Pre-calculate query positions for causal masking
    q_idx = jnp.arange(Tq)[:, None] # Shape (Tq, 1)
    
    def scan_fn(carry, block_idx_and_kv):
        block_idx, (k_block, v_block) = block_idx_and_kv
        m_old, d_old, o_old = carry
        
        # Compute raw dot product scores for this block: (Tq, block_size)
        scores = jnp.matmul(q, k_block.T) / jnp.sqrt(d_dim)
        
        # Apply causal mask if required
        if causal:
            # Absolute key positions for this block
            k_idx = block_idx * block_size + jnp.arange(block_size)[None, :] # Shape (1, block_size)
            mask = q_idx < k_idx
            scores = jnp.where(mask, -jnp.inf, scores)
            
        # 1. Local max along key axis for this block
        m_block = jnp.max(scores, axis=-1, keepdims=True)
        
        # 2. Update global max
        m_new = jnp.maximum(m_old, m_block)
        
        # To avoid exp(inf - inf) NaN if a row is completely masked out:
        m_new = jnp.where(m_new == -jnp.inf, -jnp.inf, m_new)
        
        # 3. Rescale factors
        alpha = jnp.exp(m_old - m_new)
        # Handle the -inf case cleanly
        alpha = jnp.where(m_old == -jnp.inf, 0.0, alpha)
        
        # 4. Compute rescaled local exponentials
        p_block = jnp.exp(scores - m_new)
        
        # 5. Update denominator sum and output numerator
        d_new = alpha * d_old + jnp.sum(p_block, axis=-1, keepdims=True)
        o_new = alpha * o_old + jnp.matmul(p_block, v_block)
        
        return (m_new, d_new, o_new), None

    # Pack block indices along with k and v blocks
    block_indices = jnp.arange(num_blocks)
    xs = (block_indices, (k_blocks, v_blocks))
    
    # Run the scan over key/value blocks
    (final_m, final_d, final_o), _ = lax.scan(scan_fn, initial_carry, xs)
    
    # Final normalization step: O = o_num / d_den
    return final_o / final_d
