import jax
import jax.numpy as jnp

def hessian_vector_product(w: jax.Array, v: jax.Array, x: jax.Array, y: jax.Array, alpha: float) -> jax.Array:
    """
    Returns the float32 Hessian-vector product of shape (D,).
    """
    def loss_fn(w_inner):
        logits = jnp.dot(x, w_inner)
        
        # Safe, numerically stable softplus eliminates the inf/nan issues
        loss_terms = jax.nn.softplus(logits) - y * logits
        
        return jnp.mean(loss_terms) + (alpha / 4.0) * jnp.sum(w_inner ** 4)

    grad_fn = jax.grad(loss_fn)
    _, hvp = jax.jvp(grad_fn, (w,), (v,))
    
    return hvp
