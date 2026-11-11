import jax
import jax.numpy as jnp

def per_example_clipped_gradient(params: dict, x: jax.Array, y: jax.Array, max_norm: float) -> dict:
    """
    Returns a dict with the averaged clipped grads (keys w and b) and the float32 per-example norms.
    """
    # 1. Define loss function for a single example
    def single_loss(params, xi, yi):
        pred = jnp.dot(xi, params["w"]) + params["b"]
        return 0.5 * (pred - yi) ** 2

    # 2. Get the gradient function for a single example
    grad_fn = jax.grad(single_loss, argnums=0)

    # 3. Vectorize across the batch dimension (N) to compute per-example gradients
    per_example_grads = jax.vmap(grad_fn, in_axes=(None, 0, 0))(params, x, y)

    # 4. Compute the combined L2 norm for each example across all parameter groups (w and b)
    # Square the entries of w and b individually, sum them, and take the square root
    w_sq_sum = jnp.sum(per_example_grads["w"] ** 2, axis=1)
    b_sq_sum = per_example_grads["b"] ** 2
    norms = jnp.sqrt(w_sq_sum + b_sq_sum)

    # 5. Compute clipping scale factors safely: factor = max(1, norm / max_norm)
    # Using jnp.maximum ensures we avoid division-by-zero or NaNs if a gradient is 0
    scale_factors = jnp.maximum(1.0, norms / max_norm)

    # 6. Apply clipping and compute the mean over the batch (axis=0)
    # Reshape scale_factors for 'w' to allow proper broadcasting against (N, D)
    clipped_w = per_example_grads["w"] / scale_factors[:, None]
    clipped_b = per_example_grads["b"] / scale_factors

    mean_grads = {
        "w": jnp.mean(clipped_w, axis=0),
        "b": jnp.mean(clipped_b, axis=0)
    }

    return {
        "grads": mean_grads,
        "norms": norms
    }
