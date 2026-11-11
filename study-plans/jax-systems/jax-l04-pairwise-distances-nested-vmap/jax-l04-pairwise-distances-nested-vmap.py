import jax
import jax.numpy as jnp

def pairwise_metrics(a: jax.Array, b: jax.Array) -> dict:
    """
    Returns a dict with sq_dist and cosine, both float32 arrays of shape (N, M).
    """
    # 1. Define a core function that computes metrics for a single pair of 1D vectors
    def single_pair_metrics(x, y):
        # Cast inputs to float32 to ensure correct return dtype
        x = x.astype(jnp.float32)
        y = y.astype(jnp.float32)
        
        # Squared Euclidean distance
        sq_dist = jnp.sum(jnp.square(x - y))
        
        # Cosine similarity calculations
        dot_prod = jnp.dot(x, y)
        norm_x = jnp.linalg.norm(x)
        norm_y = jnp.linalg.norm(y)
        denom = jnp.maximum(norm_x * norm_y, 1e-8)
        cosine = dot_prod / denom
        
        return {"sq_dist": sq_dist, "cosine": cosine}

    # 2. Vectorize over the rows of b (inner vmap)
    # in_axes=(None, 0) keeps 'a' row fixed and loops over rows of 'b'
    vmap_b = jax.vmap(single_pair_metrics, in_axes=(None, 0))
    
    # 3. Vectorize over the rows of a (outer vmap)
    # in_axes=(0, None) loops over rows of 'a' and treats the inner mapping of 'b' as static
    vmap_all = jax.vmap(vmap_b, in_axes=(0, None))
    
    # Execute the nested vmap
    return vmap_all(a, b)
