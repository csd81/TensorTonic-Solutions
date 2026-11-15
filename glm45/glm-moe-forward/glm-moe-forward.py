import torch
import torch.nn.functional as F

def glm_moe_forward(
    x: torch.Tensor,
    gate_W: torch.Tensor,
    rW_gate: torch.Tensor,
    rW_up: torch.Tensor,
    rW_down: torch.Tensor,
    sW_gate: torch.Tensor,
    sW_up: torch.Tensor,
    sW_down: torch.Tensor,
    n_routed_experts: int,
    num_experts_per_tok: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float = 1.0,
    norm_topk_prob: bool = True,
) -> torch.Tensor:
    n, H = x.shape
    # 1) Router logits and sigmoid scores per expert
    logits = x @ gate_W  # (n, E)
    scores = torch.sigmoid(logits)  # (n, E)

    # 2) Group-restricted routing
    experts_per_group = n_routed_experts // n_group
    group_scores = scores.view(n, n_group, experts_per_group)
    # Sum of top-2 expert scores defines each group's score
    top2 = group_scores.topk(min(2, experts_per_group), dim=-1).values.sum(dim=-1)  # (n, G)
    chosen_groups = top2.topk(topk_group, dim=-1).indices  # (n, KG)
    
    group_mask = torch.zeros(n, n_group, dtype=scores.dtype, device=scores.device)
    group_mask.scatter_(1, chosen_groups, 1.0)
    
    mask = group_mask[:, :, None].expand(n, n_group, experts_per_group).reshape(n, n_routed_experts)
    masked = scores * mask  # (n, E) with unselected groups zeroed / masked

    # 3) Pick top-K experts (only inside chosen groups), keep their sigmoid scores as weights
    top_w, top_idx = masked.topk(num_experts_per_tok, dim=-1)  # (n, K)
    if norm_topk_prob:
        top_w = top_w / (top_w.sum(dim=-1, keepdim=True) + 1e-20)
    top_w = top_w * routed_scaling_factor

    # 4) Routed mixture via SwiGLU experts
    out = torch.zeros_like(x)
    for t in range(n):
        for j in range(num_experts_per_tok):
            e = int(top_idx[t, j].item())
            w = top_w[t, j]
            y = (F.silu(x[t:t+1] @ rW_gate[e]) * (x[t:t+1] @ rW_up[e])) @ rW_down[e]
            out[t] = out[t] + w * y[0]

    # 5) Shared expert (always-on dense SwiGLU)
    shared = (F.silu(x @ sW_gate) * (x @ sW_up)) @ sW_down
    return shared + out
