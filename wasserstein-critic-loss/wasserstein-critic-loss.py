import numpy as np

def wasserstein_critic_loss(real_scores: list, fake_scores: list) -> float:
    """
    Returns the loss as a float.
    """
    real_mean = np.mean(np.asarray(real_scores, dtype=float))
    fake_mean = np.mean(np.asarray(fake_scores, dtype=float))
    return float(fake_mean - real_mean)
