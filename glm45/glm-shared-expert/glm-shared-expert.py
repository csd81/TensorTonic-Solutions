import torch
import torch.nn.functional as F


def swiglu(x: torch.Tensor, W_gate: torch.Tensor, W_up: torch.Tensor, W_down: torch.Tensor) -> torch.Tensor:
    """Computes bias-free SwiGLU output."""
    gate = F.silu(x @ W_gate)
    up = x @ W_up
    return (gate * up) @ W_down


def glm_shared_plus_routed(
    x: torch.Tensor,
    routed_W_gate: torch.Tensor,
    routed_W_up: torch.Tensor,
    routed_W_down: torch.Tensor,
    shared_W_gate: torch.Tensor,
    shared_W_up: torch.Tensor,
    shared_W_down: torch.Tensor,
    top_indices: torch.Tensor,
    top_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Returns the float64 shared-plus-routed output with shape (n_tokens, hidden_size).
    """
    n_tokens, hidden_size = x.shape
    k = top_indices.shape[1]

    # 1. Compute shared expert output for all tokens: shape (n_tokens, hidden_size)
    shared_out = swiglu(x, shared_W_gate, shared_W_up, shared_W_down)

    # 2. Compute routed experts output
    # routed_W_gate shape: (n_experts, hidden_size, moe_intermediate_size)
    # We can compute all routed experts at once or per token/expert.
    n_experts = routed_W_gate.shape[0]
    
    # Compute all experts for all tokens:
    # x: (n_tokens, hidden_size) @ W_gate: (n_experts, hidden_size, intermediate) -> (n_tokens, n_experts, intermediate)
    gate_all = F.silu(torch.einsum('th, ehi -> tei', x, routed_W_gate))
    up_all = torch.einsum('th, ehi -> tei', x, routed_W_up)
    inter = gate_all * up_all
    routed_outputs = torch.einsum('tei, eih -> teh', inter, routed_W_down) # shape: (n_tokens, n_experts, hidden_size)

    # 3. Gather top-k routed expert outputs
    # top_indices shape: (n_tokens, k)
    batch_indices = torch.arange(n_tokens, device=x.device)[:, None].expand(-1, k)
    selected_routed = routed_outputs[batch_indices, top_indices] # shape: (n_tokens, k, hidden_size)

    # 4. Weight and sum routed outputs
    # top_weights shape: (n_tokens, k) -> expand to (n_tokens, k, 1)
    weighted_routed = (selected_routed * top_weights.unsqueeze(-1)).sum(dim=1)

    # 5. Combine shared output and weighted routed outputs
    out = shared_out + weighted_routed
    return out.to(torch.float64)
