import jax
import jax.numpy as jnp

def spectral_norm(w: jax.Array, v0: jax.Array, num_iters: jax.Array) -> dict:
    """
    Returns a dict with sigma of shape (), u of shape (M,) and v of shape (N,), all float32.
    """
    # 1. Initialize and normalize the starting vector v
    v_init = v0 / jnp.linalg.norm(v0)
    
    # Create a dummy u vector of shape (M,) to carry through the loop state
    u_init = jnp.zeros(w.shape[0], dtype=jnp.float32)
    
    # 2. Define the body of the loop for power iterations
    def body_fun(i, val):
        u, v = val
        
        # Update u: project v forward and normalize
        u_next = jnp.dot(w, v)
        u_next = u_next / jnp.linalg.norm(u_next)
        
        # Update v: project u backward and normalize
        v_next = jnp.dot(w.T, u_next)
        v_next = v_next / jnp.linalg.norm(v_next)
        
        return (u_next, v_next)
    
    # 3. Run the loop using jax.lax.fori_loop to handle the traced int32 scalar
    _, v_final = jax.lax.fori_loop(0, num_iters, body_fun, (u_init, v_init))
    
    # 4. Final step: compute the largest singular value (sigma) and final u vector
    wv = jnp.dot(w, v_final)
    sigma = jnp.linalg.norm(wv)
    u_final = wv / sigma
    
    return {
        "sigma": sigma,
        "u": u_final,
        "v": v_final
    }
