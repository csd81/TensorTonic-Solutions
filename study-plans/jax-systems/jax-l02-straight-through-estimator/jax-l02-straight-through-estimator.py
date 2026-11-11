import jax
import jax.numpy as jnp

def ste_fake_quantize(x: jax.Array, scale: float, qmin: int, qmax: int, cotangent: jax.Array) -> dict:
    """
    Returns a dict with float32 values and grads, both shaped like x.
    """
    # 1. Define the primal function with custom VJP
    @jax.custom_vjp
    def _quantize(x_arr):
        scaled_x = x_arr / scale
        rounded = jnp.round(scaled_x)
        clipped = jnp.clip(rounded, qmin, qmax)
        return scale * clipped

    # 2. Define the forward pass rule
    def _quantize_fwd(x_arr):
        # Return the output and the residual context needed for the backward pass
        return _quantize(x_arr), x_arr

    # 3. Define the backward pass rule
    def _quantize_bwd(res, ct):
        x_arr = res
        scaled_x = x_arr / scale
        # Create a mask where qmin <= x/scale <= qmax
        mask = (scaled_x >= qmin) & (scaled_x <= qmax)
        # Apply the mask to the upstream cotangent
        grad_x = ct * mask
        return (grad_x.astype(jnp.float32),)

    # Register the fwd/bwd rules
    _quantize.defvjp(_quantize_fwd, _quantize_bwd)

    # 4. Evaluate the forward values and compute the gradients using jax.vjp
    values, vjp_fun = jax.vjp(_quantize, x)
    grads, = vjp_fun(cotangent)

    return {
        "values": values.astype(jnp.float32),
        "grads": grads.astype(jnp.float32)
    }
