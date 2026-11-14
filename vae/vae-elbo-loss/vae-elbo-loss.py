import numpy as np

def vae_loss(x: np.ndarray, reconstruction: np.ndarray,
             mu: np.ndarray, log_var: np.ndarray) -> dict:
    """
    Returns total_loss, reconstruction_loss, and kl_loss as Python floats.
    """
    recon_per_sample = np.sum((x - reconstruction) ** 2, axis=1)
    reconstruction_loss = float(np.mean(recon_per_sample))

    kl_per_sample = -0.5 * np.sum(1.0 + log_var - mu**2 - np.exp(log_var), axis=1)
    kl_loss = float(np.mean(kl_per_sample))

    total_loss = float(reconstruction_loss + kl_loss)

    return {
        "total_loss": total_loss,
        "reconstruction_loss": reconstruction_loss,
        "kl_loss": kl_loss,
    }
