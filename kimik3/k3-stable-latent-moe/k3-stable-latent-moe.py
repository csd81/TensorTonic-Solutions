import torch
import torch.nn.functional as F


def stable_latent_moe(
    tokens: torch.Tensor,
    latent_down_projection: torch.Tensor,
    latent_up_projection: torch.Tensor,
    router_projection: torch.Tensor,
    current_bias: torch.Tensor,
    routed_gate_weights: torch.Tensor,
    routed_up_weights: torch.Tensor,
    routed_down_weights: torch.Tensor,
    shared_gate_weights: torch.Tensor,
    shared_up_weights: torch.Tensor,
    shared_down_weights: torch.Tensor,
    selected_count: int,
    eps: float = 1e-6,
    gate_cap: float = 4.0,
    up_cap: float = 25.0,
) -> dict[str, torch.Tensor]:
    """
    Returns a dictionary containing the output, routes, mixture weights, and latent routed aggregate.
    """
    orig_dtype = tokens.dtype
    device = tokens.device

    # Compute in float64 for numerical stability, then cast back at the end
    x = tokens.to(dtype=torch.float64)
    w_down_lat = latent_down_projection.to(device=device, dtype=torch.float64)
    w_up_lat = latent_up_projection.to(device=device, dtype=torch.float64)
    w_r = router_projection.to(device=device, dtype=torch.float64)
    bias = current_bias.to(device=device, dtype=torch.float64)

    r_g = routed_gate_weights.to(device=device, dtype=torch.float64)
    r_u = routed_up_weights.to(device=device, dtype=torch.float64)
    r_d = routed_down_weights.to(device=device, dtype=torch.float64)

    s_g = shared_gate_weights.to(device=device, dtype=torch.float64)
    s_u = shared_up_weights.to(device=device, dtype=torch.float64)
    s_d = shared_down_weights.to(device=device, dtype=torch.float64)

    def situ_glu_expert(
        inp: torch.Tensor, wg: torch.Tensor, wu: torch.Tensor, wd: torch.Tensor
    ) -> torch.Tensor:
        g = F.linear(inp, wg)
        u = F.linear(inp, wu)
        # SiTU-GLU: soft-capped tanh * sigmoid on gate branch, soft-capped tanh on up branch
        gate_act = (gate_cap * torch.tanh(g / gate_cap)) * torch.sigmoid(g)
        up_act = up_cap * torch.tanh(u / up_cap)
        return F.linear(gate_act * up_act, wd)

    # 1. Router scores & Top-K selection with bias
    raw_scores = torch.sigmoid(F.linear(x, w_r))  # (N, E)
    biased_scores = raw_scores + bias.unsqueeze(0)
    _, selected_experts = torch.topk(biased_scores, k=selected_count, dim=-1)  # (N, k)

    # 2. Normalize unbiased raw scores of selected experts to get mixture weights
    selected_raw = torch.gather(raw_scores, dim=-1, index=selected_experts)  # (N, k)
    mixture_weights = selected_raw / selected_raw.sum(dim=-1, keepdim=True)  # (N, k)

    # 3. Down-project tokens to latent space: z = W_down @ x
    z = F.linear(x, w_down_lat)  # (N, l)

    # 4. Aggregate weighted routed expert outputs in latent space
    num_tokens, latent_dim = z.shape
    latent_routed_aggregate = torch.zeros(
        (num_tokens, latent_dim), dtype=torch.float64, device=device
    )
    for slot in range(selected_count):
        exp_ids = selected_experts[:, slot]  # (N,)
        weights = mixture_weights[:, slot].unsqueeze(-1)  # (N, 1)
        for exp_idx in exp_ids.unique():
            mask = exp_ids == exp_idx
            exp_out = situ_glu_expert(z[mask], r_g[exp_idx], r_u[exp_idx], r_d[exp_idx])
            latent_routed_aggregate[mask] += weights[mask] * exp_out

    # 5. Apply parameter-free RMSNorm to u, then up-project to model width
    rms = torch.sqrt(latent_routed_aggregate.pow(2).mean(dim=-1, keepdim=True) + eps)
    routed_model_out = F.linear(latent_routed_aggregate / rms, w_up_lat)

    # 6. Compute the two shared experts directly at model width
    shared_out = situ_glu_expert(x, s_g[0], s_u[0], s_d[0]) + situ_glu_expert(
        x, s_g[1], s_u[1], s_d[1]
    )

    output = shared_out + routed_model_out

    return {
        "output": output.to(dtype=orig_dtype),
        "selected_experts": selected_experts.to(dtype=torch.long),
        "mixture_weights": mixture_weights.to(dtype=orig_dtype),
        "latent_routed_aggregate": latent_routed_aggregate.to(dtype=orig_dtype),
    }
