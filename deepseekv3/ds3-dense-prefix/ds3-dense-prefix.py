import torch
import torch.nn.functional as F

def _swiglu(x: torch.Tensor, W_gate: torch.Tensor, W_up: torch.Tensor, W_down: torch.Tensor) -> torch.Tensor:
    gate = F.silu(x @ W_gate.T)
    up = x @ W_up.T
    return (gate * up) @ W_down.T

def hybrid_ffn(
    x: torch.Tensor,
    weights: dict,
    layer_idx: int,
    num_dense_layers: int,
    top_k: int
) -> torch.Tensor:
    """
    Returns the selected dense or sparse feedforward output.
    """
    # 1. Dense prefix path
    if layer_idx < num_dense_layers:
        out = _swiglu(
            x,
            weights["dense_W_gate"],
            weights["dense_W_up"],
            weights["dense_W_down"],
        )
        return out.to(torch.float64)

    # 2. Sparse DeepSeekMoE path
    B, S, D = x.shape
    T = B * S
    x_flat = x.reshape(T, D)

    # Always-active shared expert
    shared_out = _swiglu(
        x_flat,
        weights["shared_W_gate"],
        weights["shared_W_up"],
        weights["shared_W_down"],
    )

    # Router: biased selection, unbiased normalized mixture weights
    router_weight = weights["router_weight"]
    expert_bias = weights["expert_bias"]
    E = router_weight.shape[0]

    affinities = torch.sigmoid(x_flat @ router_weight.T)       # (T, E)
    biased_scores = affinities + expert_bias                   # (T, E)

    _, topk_indices = torch.topk(biased_scores, k=top_k, dim=-1)
    selected_affinities = torch.gather(affinities, dim=-1, index=topk_indices)
    mixture_weights = selected_affinities / selected_affinities.sum(dim=-1, keepdim=True)

    # Selected routed experts
    W_gate = weights["W_gate"]
    W_up = weights["W_up"]
    W_down = weights["W_down"]

    routed_out = torch.zeros_like(x_flat)
    for e in range(E):
        token_idx, k_idx = torch.where(topk_indices == e)
        if token_idx.numel() == 0:
            continue

        out_e = _swiglu(x_flat[token_idx], W_gate[e], W_up[e], W_down[e])
        w_e = mixture_weights[token_idx, k_idx].unsqueeze(-1)
        routed_out.index_add_(0, token_idx, w_e * out_e)

    out = (shared_out + routed_out).reshape(B, S, D)
    return out.to(torch.float64)
