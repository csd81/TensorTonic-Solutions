import torch

_E2M1_VALUES = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def mxfp4_expert_linear(
    latent_tokens: torch.Tensor,
    packed_weights: torch.Tensor,
    scale_bytes: torch.Tensor,
    selected_experts: torch.Tensor,
    mixture_weights: torch.Tensor,
    shared_output: torch.Tensor,
) -> torch.Tensor:
    """
    Returns the combined routed and shared expert output tensor.
    """
    orig_dtype = latent_tokens.dtype
    device = latent_tokens.device

    x = latent_tokens.to(dtype=torch.float64)
    out = shared_output.to(device=device, dtype=torch.float64).clone()
    weights = mixture_weights.to(device=device, dtype=torch.float64)

    lut = torch.tensor(_E2M1_VALUES, dtype=torch.float64, device=device)

    # Identify only the unique experts actually selected across all tokens
    unique_experts = selected_experts.unique()

    for exp_idx in unique_experts:
        # packed_weights[exp_idx]: (O, G, 16)
        p_bytes = packed_weights[exp_idx].to(device=device, dtype=torch.long)
        s_bytes = scale_bytes[exp_idx].to(device=device, dtype=torch.float64)  # (O, G)

        # Extract low nibble first, then high nibble, and interleave along the last axis
        low_nibbles = p_bytes & 0x0F
        high_nibbles = (p_bytes >> 4) & 0x0F
        nibbles = torch.stack((low_nibbles, high_nibbles), dim=-1).reshape(
            p_bytes.shape[0], p_bytes.shape[1], 32
        )  # (O, G, 32)

        # Map 4-bit codes to E2M1 values and apply E8M0 block scale 2^(s - 127)
        decoded_vals = lut[nibbles]  # (O, G, 32)
        scales = torch.exp2(s_bytes - 127.0).unsqueeze(-1)  # (O, G, 1)
        w_e = (decoded_vals * scales).reshape(p_bytes.shape[0], -1)  # (O, I)

        # Accumulate weighted expert projections for all token-slot pairs routing to exp_idx
        token_mask = selected_experts == exp_idx  # (T, K)
        token_indices, slot_indices = torch.where(token_mask)

        proj = x[token_indices] @ w_e.T  # (M, O)
        slot_weights = weights[token_indices, slot_indices].unsqueeze(-1)  # (M, 1)
        out.index_add_(0, token_indices, slot_weights * proj)

    return out.to(dtype=orig_dtype)
