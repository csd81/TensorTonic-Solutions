import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import AxisType, PartitionSpec as P

def ring_all_reduce(x):
    """
    Returns: float32 array of shape (8, C) whose rows all equal the sum of x's rows, one row per device
    """
    n = 8
    mesh = jax.make_mesh((n,), ("ring",), axis_types=(AxisType.Auto,))
    to_next = [(device, (device + 1) % n) for device in range(n)]

    def per_device(row):
        chunks = row.reshape(n, -1)
        me = lax.axis_index("ring")
        travelling = chunks[me]
        for step in range(n - 1):
            arrived = lax.ppermute(travelling, "ring", to_next)
            travelling = arrived + chunks[(me - step - 1) % n]
        gathered = chunks.at[(me + 1) % n].set(travelling)
        for step in range(n - 1):
            travelling = lax.ppermute(travelling, "ring", to_next)
            gathered = gathered.at[(me - step) % n].set(travelling)
        return gathered.reshape(1, -1)

    return jax.shard_map(per_device, mesh=mesh, in_specs=P("ring"), out_specs=P("ring"))(x)