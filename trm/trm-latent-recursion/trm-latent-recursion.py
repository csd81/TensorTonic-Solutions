import torch

def tiny_net(h: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Returns the stand-in TRM network output with the same shape as h.
    """
    for layer in range(params["w_gate"].shape[0]):
        gate = torch.nn.functional.silu(h @ params["w_gate"][layer])
        up = h @ params["w_up"][layer]
        h = h + (gate * up) @ params["w_down"][layer]
        h = h * torch.rsqrt(h.square().mean(dim=-1, keepdim=True) + 1e-5)
    return h

def latent_recursion(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    n: int,
    params: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns the refined answer, the final latent, and the stacked latent history.
    """
    z_curr = z
    history = []

    for _ in range(n):
        z_curr = tiny_net(x + y + z_curr, params)
        history.append(z_curr)

    y_refined = tiny_net(y + z_curr, params)
    z_history = torch.stack(history, dim=0)

    return y_refined, z_curr, z_history
