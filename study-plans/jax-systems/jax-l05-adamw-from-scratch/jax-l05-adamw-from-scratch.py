from typing import Any
import jax
import jax.numpy as jnp

def adamw_step(
    params: Any, 
    grads: Any, 
    state: dict, 
    learning_rate: float, 
    b1: float, 
    b2: float, 
    eps: float, 
    weight_decay: float
) -> dict:
    """
    Returns a dict with the updated params and the new state holding step, mu and nu.
    """
    # 1. Update step counter
    step = state["step"] + 1
    
    # 2. Update moment updates across the trees
    # Every leaf is cleanly cast to float32 elementwise
    mu = jax.tree.map(
        lambda m, g: jnp.astype(b1 * m + (1.0 - b1) * g, jnp.float32), 
        state["mu"], 
        grads
    )
    nu = jax.tree.map(
        lambda v, g: jnp.astype(b2 * v + (1.0 - b2) * jnp.square(g), jnp.float32), 
        state["nu"], 
        grads
    )
    
    # 3. Compute bias corrections scalars
    mu_correction = 1.0 - b1 ** step
    nu_correction = 1.0 - b2 ** step

    # 4. Define parameter update step
    def update_param(p, m, v):
        p = jnp.astype(p, jnp.float32)
        m_hat = m / mu_correction
        v_hat = v / nu_correction
        return p - learning_rate * (m_hat / (jnp.sqrt(v_hat) + eps) + weight_decay * p)

    new_params = jax.tree.map(update_param, params, mu, nu)
    
    return {
        "params": new_params,
        "state": {
            "step": jnp.astype(step, jnp.int32),
            "mu": mu,
            "nu": nu
        }
    }
