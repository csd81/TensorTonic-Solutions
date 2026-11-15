import torch
import torch.nn.functional as F


def coarse_moe(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    expert_bias: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    shared_W_gate: torch.Tensor,
    shared_W_up: torch.Tensor,
    shared_W_down: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Returns the shared-plus-routed expert mixture with the same shape as x."""
    # Ensure float64
    x = x.to(torch.float64)

    # 1. Compute SwiGLU for the always-active shared expert
    # shared_W_gate, shared_W_up, shared_W_down omit leading expert dimension
    shared_gate_out = F.silu(torch.matmul(x, shared_W_gate.mT) if shared_W_gate.ndim == 2 else F.linear(x, shared_W_gate))
    # Standard SwiGLU: W_down (SiLU(W_gate x) * W_up x)
    shared_up_out = F.linear(x, shared_W_up)
    shared_gate_activated = F.silu(F.linear(x, shared_W_gate))
    shared_out = F.linear(shared_gate_activated * shared_up_out, shared_W_down)

    # Alternative linear application helper for 3D/2D tensors:
    # Let's write a clean helper or explicit matmul/einsum for SwiGLU:
    # For routed experts: W_gate has shape (E, I, D), W_up: (E, I, D), W_down: (E, D, I)
    # x has shape (B, S, D)

    B, S, D = x.shape
    E = router_weight.shape[0]

    # 2. Compute independent sigmoid router scores
    # router_weight: (E, D), x: (B, S, D) -> logits: (B, S, E)
    router_logits = torch.einsum("bsd,ed->bse", x, router_weight)
    sigmoid_scores = torch.sigmoid(router_logits)  # (B, S, E)

    # 3. Add expert_bias only when selecting top_k experts
    biased_scores = sigmoid_scores + expert_bias  # (B, S, E)

    # Select top_k indices
    topk_vals, topk_indices = torch.topk(biased_scores, k=top_k, dim=-1)  # (B, S, top_k)

    # 4. Normalize the selected original (unbiased) sigmoid scores
    selected_orig_scores = torch.gather(sigmoid_scores, dim=-1, index=topk_indices)  # (B, S, top_k)
    normalized_weights = selected_orig_scores / (selected_orig_scores.sum(dim=-1, keepdim=True) + 1e-12)  # (B, S, top_k)

    # 5. Run selected routed experts
    # W_gate: (E, I, D), W_up: (E, I, D), W_down: (E, D, I)
    # We can compute routed expert outputs for each token/batch or scatter/gather
    # For a general vectorized/batched approach:
    # Flatten B and S: (N, D) where N = B * S
    x_flat = x.reshape(-1, D)  # (N, D)
    N = x_flat.shape[0]

    routed_out_flat = torch.zeros_like(x_flat)  # (N, D)

    # Gather weights for the selected top_k experts per token
    # topk_indices has shape (B, S, top_k) -> reshape to (N, top_k)
    topk_idx_flat = topk_indices.reshape(N, top_k)  # (N, top_k)
    norm_w_flat = normalized_weights.reshape(N, top_k)  # (N, top_k)

    # Evaluate selected experts
    for k_idx in range(top_k):
        expert_idx = topk_idx_flat[:, k_idx]  # (N,)
        w_weight = norm_w_flat[:, k_idx:k_idx+1]  # (N, 1)

        # Index the weights for these expert indices
        # Since expert_idx can vary per token, we can group or index per unique expert
        unique_experts = torch.unique(expert_idx)
        for e in unique_experts:
            mask = expert_idx == e
            if not mask.any():
                continue
            xe = x_flat[mask]  # (M, D)
            gw = W_gate[e]     # (I, D)
            uw = W_up[e]       # (I, D)
            dw = W_down[e]     # (D, I)

            gate_proj = F.linear(xe, gw)  # (M, I)
            up_proj = F.linear(xe, uw)    # (M, I)
            swiglu = F.silu(gate_proj) * up_proj
            out_e = F.linear(swiglu, dw)  # (M, D)

            routed_out_flat[mask] += w_weight[mask] * out_e

    routed_out = routed_out_flat.reshape(B, S, D)

    # Shared expert helper lambda or function
    shared_gate_proj = F.linear(x, shared_W_gate)
    shared_up_proj = F.linear(x, shared_W_up)
    shared_swiglu = F.silu(shared_gate_proj) * shared_up_proj
    shared_out = F.linear(shared_swiglu, shared_W_down)

    # Add weighted routed outputs to shared expert output
    output = shared_out + routed_out
    return output.to(torch.float64)
