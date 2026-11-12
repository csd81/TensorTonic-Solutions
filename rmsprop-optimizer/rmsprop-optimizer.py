import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    """
    w_arr = np.asarray(w, dtype=float)
    g_arr = np.asarray(g, dtype=float)
    s_arr = np.asarray(s, dtype=float)

    # Step 1: Update Running Average
    new_s = beta * s_arr + (1.0 - beta) * (g_arr ** 2)

    # Step 2: Parameter Update
    new_w = w_arr - (lr / (np.sqrt(new_s) + eps)) * g_arr

    return (
        [round(val, 6) for val in new_w.flatten()] if w_arr.ndim == 1 else new_w.round(6).tolist(),
        [round(val, 6) for val in new_s.flatten()] if s_arr.ndim == 1 else new_s.round(6).tolist(),
    )
