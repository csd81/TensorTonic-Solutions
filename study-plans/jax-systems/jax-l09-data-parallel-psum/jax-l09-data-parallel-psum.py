import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import AxisType, PartitionSpec as P

def data_parallel_step(w: jax.Array, b: jax.Array, x: jax.Array, y: jax.Array, lr: float) -> dict:
    """
    Returns a dict with w (D, K) and b (K,) after one gradient step, and loss, 
    the float32 scalar mean loss before the step.
    """
    # 1. Build the 1D mesh named 'data' over the 8 devices
    mesh = jax.make_mesh((8,), ("data",), axis_types=(AxisType.Auto,))
    
    # 2. Define the local per-device shard loss function
    def shard_loss(w_local, b_local, x_local, y_local):
        residual = x_local @ w_local + b_local - y_local
        # Sum over outputs and divide by the local batch size N/8
        return jnp.sum(residual ** 2) / x_local.shape[0]

    # 3. Define the per-device logic containing the communication primitives
    def per_device(w_shard, b_shard, x_shard, y_shard):
        # Cast replicated inputs to varying so they can be differentiated per-device
        w_shard, b_shard = lax.pcast((w_shard, b_shard), "data", to="varying")
        
        loss_val, (grad_w, grad_b) = jax.value_and_grad(shard_loss, argnums=(0, 1))(
            w_shard, b_shard, x_shard, y_shard
        )
        
        # All-reduce via pmean across the 'data' axis
        return (
            lax.pmean(loss_val, "data"), 
            lax.pmean(grad_w, "data"), 
            lax.pmean(grad_b, "data")
        )

    # 4. Bind the function to the mesh using jax.shard_map functionally
    step = jax.shard_map(
        per_device,
        mesh=mesh,
        in_specs=(P(), P(), P("data"), P("data")),
        out_specs=(P(), P(), P()),
    )
    
    # 5. Run the map and apply synchronized gradients
    loss, grad_w, grad_b = step(w, b, x, y)
    
    return {
        "w": w - lr * grad_w, 
        "b": b - lr * grad_b, 
        "loss": loss
    }
