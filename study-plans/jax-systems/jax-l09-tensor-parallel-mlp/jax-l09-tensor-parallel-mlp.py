import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

def tensor_parallel_mlp(x: jax.Array, w1: jax.Array, b1: jax.Array, w2: jax.Array, b2: jax.Array) -> dict:
    """
    Returns a dict with out, the float32 (B, O) output on every device, and hidden, the float32 (B, H) activations split by columns across the 8 devices.
    """
    # 1. Create a 1D mesh named 'model' spanning all 8 devices
    devices = jax.devices()
    mesh = Mesh(devices, ('model',))
    
    # 2. Define the localized computation per device shard
    def _shard_fn(x_shard, w1_shard, b1_shard, w2_shard, b2_shard):
        # x_shard: (B, D) [Replicated]
        # w1_shard: (D, H/8) [Split on axis 1]
        # b1_shard: (H/8,) [Split on axis 0]
        h_shard = jax.nn.relu(jnp.dot(x_shard, w1_shard) + b1_shard) # (B, H/8)
        
        # w2_shard: (H/8, O) [Split on axis 0]
        partial_out = jnp.dot(h_shard, w2_shard) # (B, O)
        
        # Sum the partial matrix multiplication products across all devices
        out_replicated = lax.psum(partial_out, axis_name='model') + b2_shard
        
        return {'out': out_replicated, 'hidden': h_shard}

    # 3. Map the sharded function over the mesh with appropriate PartitionSpecs
    return shard_map(
        _shard_fn,
        mesh=mesh,
        in_specs=(
            P(),               # x: replicated
            P(None, 'model'),  # w1: split along columns (hidden dimension)
            P('model',),       # b1: split along hidden dimension
            P('model', None),  # w2: split along rows (hidden dimension)
            P()                # b2: replicated
        ),
        out_specs={
            'out': P(),              # out: fully replicated
            'hidden': P(None, 'model') # hidden: column-sharded
        }
    )(x, w1, b1, w2, b2)
