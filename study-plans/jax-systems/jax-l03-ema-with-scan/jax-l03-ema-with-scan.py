import jax
import jax.numpy as jnp

def ema_scan(values: jax.Array, decay: jax.Array) -> dict:
    """
    Returns a dict with final_state of shape (D,) and corrected of shape (T, D), both float32.
    """
    # Grab the dimension D from the values array
    D = values.shape[1]
    
    # Initialize the carry components: 
    # - m_0: raw running average state of shape (D,) initialized to 0
    # - t_0: step count tracker initialized to 1.0
    init_m = jnp.zeros((D,), dtype=jnp.float32)
    init_t = jnp.array(1.0, dtype=jnp.float32)
    init_carry = (init_m, init_t)
    
    def scan_fn(carry, x_t):
        m_prev, t = carry
        
        # Compute the raw moving average at step t
        m_t = decay * m_prev + (1.0 - decay) * x_t
        
        # Compute the bias-corrected value
        m_hat_t = m_t / (1.0 - jnp.power(decay, t))
        
        # Advance the carry to the next step
        next_carry = (m_t, t + 1.0)
        
        return next_carry, m_hat_t

    # Scan over the sequence along the time axis (T)
    final_carry, corrected = jax.lax.scan(scan_fn, init_carry, values)
    
    # Extract the final raw average m_T from the carry
    final_state, _ = final_carry
    
    return {
        "final_state": final_state,
        "corrected": corrected
    }
