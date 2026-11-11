from typing import Any
import jax
import jax.numpy as jnp

def momentum_sgd_step(params: Any, grads: Any, velocity: Any, learning_rate: float, momentum: float) -> tuple:
    """
    Returns a tuple (new_params, new_velocity) of pytrees with the same structure as params.
    """
    # Compute the new velocity for each leaf across the three structural trees
    new_velocity = jax.tree.map(
        lambda v, g: momentum * v + g, 
        velocity, 
        grads
    )
    
    # Compute the new parameters using the updated velocity
    new_params = jax.tree.map(
        lambda p, v_new: p - learning_rate * v_new, 
        params, 
        new_velocity
    )
    
    return new_params, new_velocity
