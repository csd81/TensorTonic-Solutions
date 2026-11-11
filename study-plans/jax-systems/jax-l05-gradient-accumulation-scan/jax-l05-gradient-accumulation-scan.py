import jax
import jax.numpy as jnp

def accumulate_gradients(params: dict, x: jax.Array, y: jax.Array, num_microbatches: int) -> dict:
    """
    Returns a dict with the full-batch mean loss under loss and its gradient dict under grads.
    """
    B, D = x.shape
    K = y.shape[1]
    S = B // num_microbatches
    
    # Reshape full batch to (num_microbatches, microbatch_size, feature_dim)
    xs = x.reshape(num_microbatches, S, D)
    ys = y.reshape(num_microbatches, S, K)
    
    # Compute the scaled microbatch loss so that summing across microbatches yields the full-batch mean loss
    def microbatch_loss_fn(p, x_mb, y_mb):
        h = jnp.tanh(jnp.matmul(x_mb, p["w1"]) + p["b1"])
        o = jnp.matmul(h, p["w2"]) + p["b2"]
        return jnp.sum((o - y_mb) ** 2) / B

    def scan_body(carry, microbatch):
        carry_loss, carry_grads = carry
        x_mb, y_mb = microbatch
        
        # Call value_and_grad inside the scan body to prevent memory scaling with full-batch size
        loss_mb, grads_mb = jax.value_and_grad(microbatch_loss_fn)(params, x_mb, y_mb)
        
        # Accumulate loss and gradients across steps
        new_loss = carry_loss + loss_mb
        new_grads = jax.tree_util.tree_map(lambda c, g: c + g, carry_grads, grads_mb)
        
        return (new_loss, new_grads), None

    # Initialize accumulation containers with zeros matching the structures
    init_loss = jnp.array(0.0, dtype=jnp.float32)
    init_grads = jax.tree_util.tree_map(jnp.zeros_like, params)
    
    (total_loss, total_grads), _ = jax.lax.scan(
        scan_body, 
        (init_loss, init_grads), 
        (xs, ys), 
        length=num_microbatches
    )
    
    return {"loss": total_loss, "grads": total_grads}
