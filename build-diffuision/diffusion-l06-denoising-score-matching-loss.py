import torch

def denoising_score_matching_loss(x0: torch.Tensor, noise: torch.Tensor, sigmas: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> dict:
    """
    Returns a dict with the batch loss "loss" (Python float) and "per_sample_loss" (tensor of shape (B,)).
    """
    # Reshape sigmas to (B, 1) for clean broadcasting across feature dimension D
    sigma_col = sigmas.unsqueeze(-1)

    # 1. Corrupt clean samples with scaled Gaussian noise
    x_noisy = x0 + sigma_col * noise

    # 2. Compute predicted score s = (x_noisy @ W + b) / sigma
    pred_score = (x_noisy @ weight + bias) / sigma_col

    # 3. Compute target score (-noise / sigma) and feature-wise squared residual
    target_score = -noise / sigma_col
    sq_error = (pred_score - target_score) ** 2

    # 4. Sum across features D and apply 0.5 * sigma^2 weighting per sample
    per_sample_loss = 0.5 * (sigmas ** 2) * sq_error.sum(dim=-1)

    # 5. Average across the batch and convert scalar loss to Python float
    loss = per_sample_loss.mean().item()

    return {
        "loss": loss,
        "per_sample_loss": per_sample_loss,
    }
