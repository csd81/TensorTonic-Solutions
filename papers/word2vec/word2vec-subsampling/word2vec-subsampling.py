import torch


def subsample_keep_probs(counts: torch.Tensor, t: float = 1e-5) -> torch.Tensor:
    """
    Returns the float64 keep probability for every vocabulary word.
    """
    # Ensure counts is float64 for high precision
    counts = counts.to(dtype=torch.float64)

    # Calculate relative frequency f(w) for each word
    freqs = counts / counts.sum()

    # Apply the square-root subsampling rule and clamp to a maximum of 1.0
    keep_probs = torch.clamp(torch.sqrt(t / freqs), max=1.0)

    return keep_probs
