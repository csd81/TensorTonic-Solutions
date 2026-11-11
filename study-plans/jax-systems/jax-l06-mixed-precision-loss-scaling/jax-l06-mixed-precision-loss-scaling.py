import jax
import jax.numpy as jnp

def mixed_precision_step(params: dict, x: jax.Array, y: jax.Array, loss_scale: jax.Array, good_steps: jax.Array, lr: jax.Array, growth_interval: int) -> dict:
    """
    Returns a dict with the unscaled float32 loss, the updated float32 params, the new loss_scale, the new int32 good_steps and the bool grads_finite flag.
    """
    # 1. Define the forward pass for unscaled loss
    def loss_fn(p):
        w_bf = p['w'].astype(jnp.bfloat16)
        b_bf = p['b'].astype(jnp.bfloat16)
        x_bf = x.astype(jnp.bfloat16)
        
        y_hat_bf = jnp.matmul(x_bf, w_bf) + b_bf
        y_hat = y_hat_bf.astype(jnp.float32)
        
        return jnp.mean((y_hat - y) ** 2)

    # 2. Compute the unscaled loss and gradients of the scaled loss
    loss = loss_fn(params)
    scaled_loss_fn = lambda p: loss_scale * loss_fn(p)
    scaled_grads = jax.grad(scaled_loss_fn)(params)
    
    # 3. Unscale gradients and check for finiteness
    grads = jax.tree_util.tree_map(lambda g: g / loss_scale, scaled_grads)
    
    grads_finite = True
    for g in jax.tree_util.tree_leaves(grads):
        grads_finite = jnp.logical_and(grads_finite, jnp.all(jnp.isfinite(g)))

    # 4. Compute conditional state transitions
    # Finite branch updates
    new_good_steps_finite = good_steps + 1
    grow = (new_good_steps_finite == growth_interval)
    next_good_steps_finite = jnp.where(grow, 0, new_good_steps_finite)
    next_loss_scale_finite = jnp.where(grow, loss_scale * 2.0, loss_scale)
    next_params_finite = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
    
    # Non-finite branch updates
    next_loss_scale_nan = jnp.maximum(loss_scale / 2.0, 1.0)
    
    # Select outcomes based on grads_finite status using traced where operations
    final_good_steps = jnp.where(grads_finite, next_good_steps_finite, 0)
    final_loss_scale = jnp.where(grads_finite, next_loss_scale_finite, next_loss_scale_nan)
    final_params = jax.tree_util.tree_map(
        lambda pf, pn: jnp.where(grads_finite, pf, pn), 
        next_params_finite, 
        params
    )

    return {
        "loss": loss,
        "params": final_params,
        "loss_scale": final_loss_scale,
        "good_steps": final_good_steps.astype(jnp.int32),
        "grads_finite": grads_finite
    }
