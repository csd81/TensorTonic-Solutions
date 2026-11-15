import torch


def glm_route(
    x: torch.Tensor,
    gate_W: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    n_routed_experts: int,
    num_experts_per_tok: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float = 1.0,
    norm_topk_prob: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Returns a dictionary containing top_indices and top_weights tensors.
    """
    n_tokens, _ = x.shape
    experts_per_group = n_routed_experts // n_group

    # 1. Compute original sigmoid routing affinities: (n_tokens, n_routed_experts)
    logits = x @ gate_W
    scores = torch.sigmoid(logits)

    # 2. Add correction bias for routing decisions only
    scores_for_choice = scores + e_score_correction_bias.unsqueeze(0)

    # 3. Group scores by summing the top-2 corrected affinities per group
    grouped_scores = scores_for_choice.view(n_tokens, n_group, experts_per_group)
    top2_per_group = torch.topk(grouped_scores, k=2, dim=-1).values
    group_scores = top2_per_group.sum(dim=-1)  # (n_tokens, n_group)

    # 4. Select topk_group groups and build a boolean mask over all experts
    top_group_indices = torch.topk(group_scores, k=topk_group, dim=-1).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(dim=-1, index=top_group_indices, value=True)

    # Expand group mask to (n_tokens, n_routed_experts)
    expert_mask = (
        group_mask.unsqueeze(-1)
        .expand(n_tokens, n_group, experts_per_group)
        .reshape(n_tokens, n_routed_experts)
    )

    # 5. Mask out unselected groups and pick top num_experts_per_tok experts
    masked_choice_scores = scores_for_choice.masked_fill(~expert_mask, float("-inf"))
    top_indices = torch.topk(
        masked_choice_scores, k=num_experts_per_tok, dim=-1
    ).indices

    # 6. Gather mixture weights from the original uncorrected sigmoid scores
    top_weights = scores.gather(dim=-1, index=top_indices)

    # 7. Optionally normalize and apply scaling factor
    if norm_topk_prob:
        denominator = top_weights.sum(dim=-1, keepdim=True) + 1e-20
        top_weights = top_weights / denominator

    top_weights = top_weights * routed_scaling_factor

    return {
        "top_indices": top_indices.to(torch.int64),
        "top_weights": top_weights.to(torch.float64),
    }
