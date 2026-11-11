import jax
import jax.numpy as jnp

def int8_quantized_matmul(x: jax.Array, w: jax.Array) -> dict:
    """
    Computes an int8 quantized matrix multiplication with per-row scales for x 
    and per-column scales for w, accumulating products in int32.
    """
    # 1. Compute per-row scales for x (along D_in axis)
    x_max = jnp.max(jnp.abs(x), axis=1)
    x_scales = jnp.where(x_max == 0.0, 1.0, x_max / 127.0)
    
    # 2. Compute per-column scales for w (along D_in axis)
    w_max = jnp.max(jnp.abs(w), axis=0)
    w_scales = jnp.where(w_max == 0.0, 1.0, w_max / 127.0)
    
    # 3. Quantize x and w to int8 codes
    # Use jnp.round and clip to range [-127, 127]
    x_q = jnp.clip(jnp.round(x / x_scales[:, None]), -127, 127).astype(jnp.int8)
    w_q = jnp.clip(jnp.round(w / w_scales[None, :]), -127, 127).astype(jnp.int8)
    
    # 4. Perform the integer matrix multiplication accumulating in int32
    a = jnp.matmul(x_q, w_q, preferred_element_type=jnp.int32)
    
    # 5. Dequantize using the outer product of the scales
    # Use broadcasting via outer product instead of matrix multiplication
    y = a.astype(jnp.float32) * (x_scales[:, None] * w_scales[None, :])
    
    return {
        "w_q": w_q,
        "w_scales": w_scales,
        "x_q": x_q,
        "x_scales": x_scales,
        "y": y
    }
