import jax
import jax.numpy as jnp

@jax.jit(static_argnums=1)
def per_head_rms_norm(x: jax.Array, num_heads: int) -> dict:
    """
    Returns a dict with heads of shape (B, num_heads, T, D // num_heads) and merged of shape (B, T, D), both float32.
    """
    B, T, D = x.shape
    D_h = D // num_heads
    
    # Split the final dimension D into (num_heads, D_h)
    x_split = x.reshape(B, T, num_heads, D_h)
    
    # Compute Root Mean Square (RMS) normalization over the head width dimension (D_h)
    # Mean of squares -> sqrt(ms + eps) -> divide
    ms = jnp.mean(jnp.square(x_split), axis=-1, keepdims=True)
    x_norm = x_split / jnp.sqrt(ms + 1e-6)
    
    # Transpose to get (B, num_heads, T, D_h) for the 'heads' output
    heads = jnp.transpose(x_norm, (0, 2, 1, 3))
    
    # Flatten the head dimensions back into the original feature width D
    merged = x_norm.reshape(B, T, D)
    
    return {"heads": heads, "merged": merged}
