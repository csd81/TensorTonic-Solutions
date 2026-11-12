import numpy as np

def mix_root_dirichlet_noise(
    priors: np.ndarray,
    noise: np.ndarray,
    legal_mask: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """
    Returns the mixed float64 prior array in the original action order.
    """
    # Initialize the output array with exact zeros using float64
    mixed_priors = np.zeros_like(priors, dtype=np.float64)
    
    # Mask out illegal actions from the inputs
    legal_priors = priors * legal_mask
    legal_noise = noise * legal_mask
    
    # Separately normalize priors and noise over legal actions
    p_hat = legal_priors / np.sum(legal_priors)
    eta_hat = legal_noise / np.sum(legal_noise)
    
    # Apply the mixture formula for the legal actions
    mixed_priors[legal_mask] = (1.0 - epsilon) * p_hat[legal_mask] + epsilon * eta_hat[legal_mask]
    
    return mixed_priors
