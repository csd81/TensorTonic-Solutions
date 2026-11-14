import numpy as np


def gan_losses(real_probs: np.ndarray, fake_probs: np.ndarray) -> dict:
    """
    Returns discriminator_loss and generator_loss as Python floats.
    """
    eps = 1e-8

    # Clip probabilities to [1e-8, 1 - 1e-8] for numerical stability
    real_probs_clipped = np.clip(real_probs, eps, 1.0 - eps)
    fake_probs_clipped = np.clip(fake_probs, eps, 1.0 - eps)

    # Discriminator loss: -1/N * sum(log(p_real) + log(1 - p_fake))
    discriminator_loss = -np.mean(
        np.log(real_probs_clipped) + np.log(1.0 - fake_probs_clipped)
    )

    # Non-saturating generator loss: -1/N * sum(log(p_fake))
    generator_loss = -np.mean(np.log(fake_probs_clipped))

    return {
        "discriminator_loss": float(discriminator_loss),
        "generator_loss": float(generator_loss),
    }
