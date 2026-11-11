from typing import Any
import jax
import jax.numpy as jnp

def clip_by_global_norm(grads: Any, max_norm: float) -> dict:
    """
    Returns a dict with the clipped gradient pytree under clipped and the float32 global norm under global_norm.
    """
    # 1. Flatten the pytree to access all leaves
    leaves, treedef = jax.tree_util.tree_flatten(grads)
    
    # 2. Compute the sum of squares across all leaves in float32
    squared_sum = sum(jnp.sum(jnp.square(leaf.astype(jnp.float32))) for leaf in leaves)
    global_norm = jnp.sqrt(squared_sum)
    
    # 3. Calculate the scale factor safely under jax.jit
    # s = max_norm / max(global_norm, max_norm)
    scale = jnp.minimum(1.0, max_norm / jnp.maximum(global_norm, 1e-12))
    
    # 4. Apply the scale factor to each leaf and restore its original dtype
    def scale_leaf(leaf):
        clipped_leaf = leaf.astype(jnp.float32) * scale
        return clipped_leaf.astype(leaf.dtype)
        
    clipped_grads = jax.tree_util.tree_map(scale_leaf, grads)
    
    return {
        "clipped": clipped_grads,
        "global_norm": global_norm
    }
