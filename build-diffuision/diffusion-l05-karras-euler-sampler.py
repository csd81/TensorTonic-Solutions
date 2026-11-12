import torch

def karras_euler_sample(initial_noise: torch.Tensor, num_steps: int, sigma_min: float, sigma_max: float, rho: float, data_mean: float, data_std: float, method: str) -> dict:
    """
    Returns a dict with the noise levels "sigmas" of shape (num_steps + 1,) and the final "samples" with the shape of initial_noise.
    """
    device = initial_noise.device
    dtype = initial_noise.dtype

    # 1. Build the Karras noise schedule with an appended 0
    step_indices = torch.arange(num_steps, device=device, dtype=dtype)
    inv_rho = 1.0 / rho
    sigmas = (sigma_max ** inv_rho + (step_indices / (num_steps - 1)) * (sigma_min ** inv_rho - sigma_max ** inv_rho)) ** rho
    sigmas = torch.cat([sigmas, torch.zeros(1, device=device, dtype=dtype)])

    # Exact denoiser for Gaussian data N(data_mean, data_std^2)
    s_sq = data_std ** 2
    def denoiser(x_in: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma_sq = sigma ** 2
        return (s_sq * x_in + sigma_sq * data_mean) / (s_sq + sigma_sq)

    # 2. Initialize sample at sigma_max
    x = initial_noise * sigmas[0]

    # 3. Integrate the probability flow ODE
    for i in range(num_steps):
        sigma_i = sigmas[i]
        sigma_next = sigmas[i + 1]
        dt = sigma_next - sigma_i

        # Euler slope and proposal
        d = (x - denoiser(x, sigma_i)) / sigma_i
        x_next = x + dt * d

        # Apply 2nd-order Heun correction when sigma_next > 0
        if method == "heun" and sigma_next > 0:
            d_next = (x_next - denoiser(x_next, sigma_next)) / sigma_next
            x = x + dt * 0.5 * (d + d_next)
        else:
            x = x_next

    return {
        "sigmas": sigmas,
        "samples": x,
    }
