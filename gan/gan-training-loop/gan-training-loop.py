import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def train_discriminator_step(
    real_data: np.ndarray,
    fake_data: np.ndarray,
    D_W: np.ndarray,
    learning_rate: float,
) -> dict:
    """
    Returns updated discriminator weights and the pre-update loss.
    """
    X_real = np.asarray(real_data, dtype=np.float64)
    X_fake = np.asarray(fake_data, dtype=np.float64)
    W = np.asarray(D_W, dtype=np.float64)

    batch_size = X_real.shape[0]

    # 1. Forward pass: unclipped probabilities for real and fake batches
    p_real = _sigmoid(X_real @ W)
    p_fake = _sigmoid(X_fake @ W)

    # 2. Pre-update discriminator loss (clip probabilities only for logs)
    eps = 1e-12
    p_real_safe = np.clip(p_real, eps, 1.0 - eps)
    p_fake_safe = np.clip(p_fake, eps, 1.0 - eps)
    discriminator_loss = -float(
        np.mean(np.log(p_real_safe) + np.log(1.0 - p_fake_safe))
    )

    # 3. Analytical gradient with respect to D_W using unclipped probabilities
    grad_W = (X_real.T @ (p_real - 1.0) + X_fake.T @ p_fake) / batch_size

    # 4. Gradient descent update
    new_weights = (W - learning_rate * grad_W).astype(np.float64)

    return {
        "new_discriminator_weights": new_weights,
        "discriminator_loss": discriminator_loss,
    }
