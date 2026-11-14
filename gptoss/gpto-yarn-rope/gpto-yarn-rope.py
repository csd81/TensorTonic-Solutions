import math
import torch


def compute_yarn_rope_freqs(
    head_dim: int,
    base: float,
    initial_context_length: int,
    scaling_factor: float,
    ntk_alpha: float = 1.0,
    ntk_beta: float = 32.0,
) -> dict:
    """
    Returns concentration and inv_freq in a dictionary.
    """
    freq = base ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)

    if scaling_factor <= 1.0:
        return {
            "concentration": 1.0,
            "inv_freq": 1.0 / freq,
        }

    concentration = 0.1 * math.log(scaling_factor) + 1.0

    # NTK-by-parts wavelength bounds mapped to dimension pair indices
    d_half = head_dim / 2.0
    low = (
        d_half
        * math.log(initial_context_length / (ntk_beta * 2.0 * math.pi))
        / math.log(base)
    )
    high = (
        d_half
        * math.log(initial_context_length / (ntk_alpha * 2.0 * math.pi))
        / math.log(base)
    )

    # Linear ramp mask: 1 for high-frequency channels, 0 for low-frequency channels
    idx = torch.arange(0, d_half, dtype=torch.float64)
    ramp = (idx - low) / (high - low)
    mask = 1.0 - ramp.clamp(0.0, 1.0)

    inv_freq_extrapolation = 1.0 / freq
    inv_freq_interpolation = 1.0 / (scaling_factor * freq)
    inv_freq = inv_freq_interpolation * (1.0 - mask) + inv_freq_extrapolation * mask

    return {
        "concentration": float(concentration),
        "inv_freq": inv_freq,
    }
