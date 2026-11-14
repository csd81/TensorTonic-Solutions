import torch


def noise_distribution(
    counts: torch.Tensor, alpha: float = 0.75
) -> torch.Tensor:
    """
    Returns the float64 negative-sampling distribution over the vocabulary.
    """
    # Ensure float64 precision
    counts = counts.to(dtype=torch.float64)

    # Raise each count to the power of alpha
    powered_counts = torch.pow(counts, alpha)

    # Normalize so the probabilities sum to 1
    probs = powered_counts / powered_counts.sum()

    return probs
