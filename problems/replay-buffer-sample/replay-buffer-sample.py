import numpy as np

def replay_buffer_sample(buffer: list, batch_size: int, seed: int) -> list:
    """
    Returns a deterministic sample of transitions.
    """
    np.random.seed(seed)
    indices = np.random.choice(len(buffer), size=batch_size, replace=False)
    sorted_indices = np.sort(indices)
    return [buffer[i] for i in sorted_indices]
