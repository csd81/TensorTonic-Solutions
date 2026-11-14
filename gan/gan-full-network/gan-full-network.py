import numpy as np


def gan_forward(
    z: np.ndarray,
    real_data: np.ndarray,
    G_W: np.ndarray,
    G_b: np.ndarray,
    D_W: np.ndarray,
) -> dict:
    """
    Returns generated samples, probabilities, and both GAN losses.
    """
    z = np.asarray(z, dtype=np.float64)
    real_data = np.asarray(real_data, dtype=np.float64)
    G_W = np.asarray(G_W, dtype=np.float64)
    G_b = np.asarray(G_b, dtype=np.float64)
    D_W = np.asarray(D_W, dtype=np.float64)

    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    # 1. Generate fake samples: X_hat = tanh(Z * W_G + b_G)
    generated_samples = np.tanh(z @ G_W + G_b).astype(np.float64)

    # 2. Discriminator probabilities for real and fake batches
    real_probabilities = sigmoid(real_data @ D_W).astype(np.float64)
    fake_probabilities = sigmoid(generated_samples @ D_W).astype(np.float64)

    # 3. Clip probabilities strictly before computing logarithms
    eps = 1e-12
    p_real_clipped = np.clip(real_probabilities, eps, 1.0 - eps)
    p_fake_clipped = np.clip(fake_probabilities, eps, 1.0 - eps)

    # 4. Compute adversarial losses
    discriminator_loss = float(
        -np.mean(np.log(p_real_clipped) + np.log(1.0 - p_fake_clipped))
    )
    generator_loss = float(-np.mean(np.log(p_fake_clipped)))

    return {
        "generated_samples": generated_samples,
        "real_probabilities": real_probabilities,
        "fake_probabilities": fake_probabilities,
        "discriminator_loss": discriminator_loss,
        "generator_loss": generator_loss,
    }
