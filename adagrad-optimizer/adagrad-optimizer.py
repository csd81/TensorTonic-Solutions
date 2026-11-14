import numpy as np

def adagrad_step(w: list, g: list, G: list, lr: float = 0.01, eps: float = 1e-8) -> dict:
    """
    Returns a dictionary with new_w and new_G.
    """
    w = np.asarray(w, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)

    new_G = G + np.square(g)
    new_w = w - lr * g / np.sqrt(new_G + eps)

    return {"new_w": new_w, "new_G": new_G}
