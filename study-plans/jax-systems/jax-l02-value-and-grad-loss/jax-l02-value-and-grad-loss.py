import jax
import jax.numpy as jnp

def logistic_value_and_grad(params: dict, x: jax.Array, y: jax.Array, l2: float) -> dict:
    """
    Returns a dict with the float32 scalar loss and a grads dict with keys w and b.
    """
    def loss_fn(p):
        w = p["w"]
        b = p["b"]
        
        # Calculate logits: z = x @ w + b
        logits = jnp.dot(x, w) + b
        
        # Numerically stable log(1 + exp(z)) - y * z
        # jnp.logaddexp(0.0, logits) computes log(e^0 + e^logits) stably
        loss_entries = jnp.logaddexp(0.0, logits) - y * logits
        
        # Total loss = mean cross-entropy + L2 regularization penalty
        total_loss = jnp.mean(loss_entries) + 0.5 * l2 * jnp.sum(w ** 2)
        return total_loss

    # Use jax.value_and_grad to get both the forward loss value and structural gradients
    loss_val, grads = jax.value_and_grad(loss_fn)(params)
    
    return {
        "loss": loss_val,
        "grads": grads
    }
