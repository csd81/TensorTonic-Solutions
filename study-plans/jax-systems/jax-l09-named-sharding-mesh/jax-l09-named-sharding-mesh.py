import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
import numpy as np

def shard_layout(x: jax.Array, mesh_shape: tuple, axis_names: tuple, spec) -> dict:
    """
    Returns a dict with array (x placed on the mesh with spec), and shard_shape and shard_start, 
    int32 arrays of shape (8, x.ndim) with one row per device id.
    """
    # Use the official shorthand to make the mesh directly
    mesh = jax.make_mesh(mesh_shape, axis_names)
    
    # Create the named sharding and place the array on the mesh
    sharding = NamedSharding(mesh, spec)
    sharded_array = jax.device_put(x, sharding)
    
    # Pre-allocate tracking arrays for all 8 devices
    ndim = x.ndim
    shard_shape_np = np.zeros((8, ndim), dtype=np.int32)
    shard_start_np = np.zeros((8, ndim), dtype=np.int32)
    
    # Extract structural indices from each addressable shard
    for shard in sharded_array.addressable_shards:
        d_id = shard.device.id
        
        # Unpack slices safely using the length of each dimension
        for i, (sl, size) in enumerate(zip(shard.index, x.shape)):
            start, stop, step = sl.indices(size)
            shard_start_np[d_id, i] = start
            shard_shape_np[d_id, i] = stop - start
            
    return {
        "array": sharded_array,
        "shard_shape": jnp.array(shard_shape_np, dtype=jnp.int32),
        "shard_start": jnp.array(shard_start_np, dtype=jnp.int32)
    }
