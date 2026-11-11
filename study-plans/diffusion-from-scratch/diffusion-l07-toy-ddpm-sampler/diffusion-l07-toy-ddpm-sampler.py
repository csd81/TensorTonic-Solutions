import math
from typing import Optional

import torch
import torch.nn.functional as F

def toy_ddpm_sample(x_T: torch.Tensor, betas: torch.Tensor, noise: torch.Tensor, params: dict, embed_dim: int, clip_value: Optional[float]) -> dict:
    """
    Returns a dict with "samples" of shape (N, D) and "pred_x0" of shape (T, N, D).
    """
    device = x_T.device
    dtype = x_T.dtype
    N, D = x_T.shape
    T = betas.shape[0]

    # Precompute noise schedule terms
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    # Precompute sinusoidal frequencies: f_i = exp(-ln(10000) * i / half)
    half = embed_dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device, dtype=dtype) / half
    )

    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]

    x = x_T.clone()
    pred_x0_history = []

    for k in range(T):
        t = T - 1 - k

        # 1. Build sinusoidal timestep embedding e(t) of shape (N, embed_dim)
        angles = t * freqs
        emb = torch.cat([torch.cos(angles), torch.sin(angles)], dim=0)
        if embed_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(1, device=device, dtype=dtype)], dim=0)
        emb_batch = emb.unsqueeze(0).expand(N, -1)

        # 2. Predict noise via 2-layer SiLU MLP
        mlp_in = torch.cat([x, emb_batch], dim=-1)
        hidden = F.silu(mlp_in @ W1 + b1)
        eps_hat = hidden @ W2 + b2

        # 3. Estimate clean sample x_hat_0 and optionally clamp
        alpha_t = alphas[t]
        beta_t = betas[t]
        alpha_bar_t = alpha_bars[t]
        alpha_bar_prev = alpha_bars[t - 1] if t > 0 else torch.ones_like(alpha_bar_t)

        x_hat_0 = (x - torch.sqrt(1.0 - alpha_bar_t) * eps_hat) / torch.sqrt(alpha_bar_t)
        if clip_value is not None:
            x_hat_0 = torch.clamp(x_hat_0, -clip_value, clip_value)

        pred_x0_history.append(x_hat_0)

        # 4. Compute posterior mean mu from (possibly clamped) x_hat_0 and x_t
        coeff_x0 = (beta_t * torch.sqrt(alpha_bar_prev)) / (1.0 - alpha_bar_t)
        coeff_xt = ((1.0 - alpha_bar_prev) * torch.sqrt(alpha_t)) / (1.0 - alpha_bar_t)
        mu = coeff_x0 * x_hat_0 + coeff_xt * x

        # 5. Add posterior noise for t > 0; return mu alone at t == 0
        if t > 0:
            beta_tilde = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            x = mu + torch.sqrt(beta_tilde) * noise[k]
        else:
            x = mu

    pred_x0 = torch.stack(pred_x0_history, dim=0)

    return {
        "samples": x,
        "pred_x0": pred_x0,
    }
