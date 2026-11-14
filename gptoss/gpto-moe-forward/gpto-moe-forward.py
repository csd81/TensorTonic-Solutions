import torch

def gpto_moe_forward(
    x: torch.Tensor,
    gate_W: torch.Tensor,
    gate_b: torch.Tensor,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
    W3: torch.Tensor,
    b3: torch.Tensor,
    top_k: int,
    swiglu_limit: float = 7.0,
) -> torch.Tensor:
    """
    Returns sparse MoE outputs with the same shape as x.
    """
    # Ensure float64 precision throughout
    x = x.to(torch.float64)
    gate_W = gate_W.to(torch.float64)
    gate_b = gate_b.to(torch.float64)
    W1, b1 = W1.to(torch.float64), b1.to(torch.float64)
    W2, b2 = W2.to(torch.float64), b2.to(torch.float64)
    W3, b3 = W3.to(torch.float64), b3.to(torch.float64)

    n_tokens, H = x.shape
    num_experts = gate_W.shape[1]

    # 1. Compute router logits and select top-k experts per token
    logits = x @ gate_W + gate_b  # (n_tokens, E)
    top_logits, top_indices = torch.topk(logits, k=top_k, dim=-1)  # (n_tokens, top_k)

    # 2. Apply softmax exclusively across the selected top-k logits
    routing_weights = torch.softmax(top_logits, dim=-1)  # (n_tokens, top_k)

    # Accumulator for final weighted outputs
    out = torch.zeros_like(x, dtype=torch.float64)

    # 3. Dispatch tokens to their selected experts
    for e in range(num_experts):
        # Find which (token_idx, k_idx) pairs chose expert `e`
        token_idx, k_idx = torch.where(top_indices == e)
        
        if token_idx.numel() == 0:
            continue

        x_e = x[token_idx]  # (n_selected, H)

        # Gate and linear projections
        a = x_e @ W1[e] + b1[e]  # (n_selected, I)
        b = x_e @ W2[e] + b2[e]  # (n_selected, I)

        # Asymmetric GPT-OSS clamping
        a_clamped = torch.clamp(a, max=swiglu_limit)
        b_clamped = torch.clamp(b, min=-swiglu_limit, max=swiglu_limit)

        # Clamped SwiGLU activation with +1 offset on the linear branch
        z = (a_clamped * torch.sigmoid(1.702 * a_clamped)) * (b_clamped + 1.0)

        # Project back to hidden size
        expert_out = z @ W3[e] + b3[e]  # (n_selected, H)

        # Weight by routing probabilities and scatter-add into output
        weights = routing_weights[token_idx, k_idx].unsqueeze(-1)  # (n_selected, 1)
        out.index_add_(0, token_idx, expert_out * weights)

    return out
