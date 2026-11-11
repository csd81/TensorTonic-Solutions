import numpy as np


def epsilon_greedy(q_values: list, epsilon: float, seed: int = 0) -> int:
    """
    Returns the action index as an integer.
    """
    rng = np.random.default_rng(seed)
    q_arr = np.asarray(q_values)

    u = rng.random()
    if u < epsilon:
        action = rng.integers(0, len(q_arr))
    else:
        action = np.argmax(q_arr)

    return int(action)
