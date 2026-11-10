import numpy as np

def jvp_vjp_2layer(W1: list, W2: list, x: list, x_dot: list, v_bar: list) -> dict:
    """
    Returns network output, JVP, VJP, and multiply counts.
    """
    W1_np = np.array(W1, dtype=float)
    W2_np = np.array(W2, dtype=float)
    x_np = np.array(x, dtype=float)
    x_dot_np = np.array(x_dot, dtype=float)
    v_bar_np = np.array(v_bar, dtype=float)
    
    # Dimensions
    m, d = W1_np.shape
    p, _ = W2_np.shape
    
    # Forward Pass
    z1 = W1_np @ x_np
    a1 = np.maximum(0, z1)
    f_x = W2_np @ a1
    
    # ReLU derivative (zero at zero as requested)
    relu_grad = (z1 > 0).astype(float)
    
    # JVP (Forward mode)
    z1_dot = W1_np @ x_dot_np
    a1_dot = relu_grad * z1_dot
    jvp_res = W2_np @ a1_dot
    
    # VJP (Backward mode)
    a1_bar = W2_np.T @ v_bar_np
    z1_bar = a1_bar * relu_grad
    vjp_res = W1_np.T @ z1_bar
    
    # Flops calculation
    jvp_flops = m * d + m + p * m
    vjp_flops = p * m + m + m * d
    full_jacobian_jvp_flops = d * jvp_flops
    full_jacobian_vjp_flops = p * vjp_flops
    
    return {
        "f": f_x.tolist(),
        "jvp_result": jvp_res.tolist(),
        "vjp_result": vjp_res.tolist(),
        "jvp_flops": jvp_flops,
        "vjp_flops": vjp_flops,
        "full_jacobian_jvp_flops": full_jacobian_jvp_flops,
        "full_jacobian_vjp_flops": full_jacobian_vjp_flops
    }
