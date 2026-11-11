import jax
import jax.numpy as jnp

def dropout_masks(seed: int, num_layers: int, shape: list[int], keep_prob: float) -> jax.Array:
    """
    Returns a bool array of shape (num_layers, followed by the entries of shape) holding one dropout keep mask per layer.
    """
    # 1. Initialize the running key using the provided seed
    running_key = jax.random.key(seed)
    
    masks = []
    
    # 2. Loop through each layer in order
    for _ in range(num_layers):
        # Split the running key into two keys
        running_key, layer_key = jax.random.split(running_key, 2)
        
        # Draw the layer's Bernoulli keep mask using the layer key
        mask = jax.random.bernoulli(layer_key, p=keep_prob, shape=shape)
        masks.append(mask)
        
    # 3. Stack the masks along the first axis (axis=0) to match the expected output format
    return jnp.stack(masks, axis=0)
