import jax
import jax.numpy as jnp

def apply_rope(q: jax.Array, k: jax.Array, positions: jax.Array, base: float) -> dict:
    """
    Returns a dict with keys q and k holding the rotated float32 arrays, same shapes as the inputs.
    """
    D = q.shape[-1]
    half = D // 2
    
    # 1. Compute theta frequencies for the first half of the dimensions
    i = jnp.arange(half, dtype=jnp.float32)
    theta = base ** (-2 * i / D)
    
    # 2. Compute angles per token position: shape (T, half)
    angles = positions.astype(jnp.float32)[:, None] * theta[None, :]
    
    # 3. Add a head dimension to match q and k shapes: shape (T, 1, half)
    cos_a = jnp.cos(angles)[:, None, :]
    sin_a = jnp.sin(angles)[:, None, :]
    
    # 4. Split vectors into paired halves
    q_first, q_second = q[..., :half], q[..., half:]
    k_first, k_second = k[..., :half], k[..., half:]
    
    # 5. Apply the rotation transformation matrix equations
    q_first_rot = q_first * cos_a - q_second * sin_a
    q_second_rot = q_second * cos_a + q_first * sin_a
    
    k_first_rot = k_first * cos_a - k_second * sin_a
    k_second_rot = k_second * cos_a + k_first * sin_a
    
    # 6. Recombine the halves along the hidden dimension axis
    q_rot = jnp.concatenate([q_first_rot, q_second_rot], axis=-1)
    k_rot = jnp.concatenate([k_first_rot, k_second_rot], axis=-1)
    
    return {"q": q_rot, "k": k_rot}
