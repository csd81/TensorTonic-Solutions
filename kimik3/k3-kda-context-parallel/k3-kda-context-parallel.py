import torch


def kda_context_parallel(
    transition_matrices: torch.Tensor,
    local_contributions: torch.Tensor,
    initial_state: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Returns a dictionary containing incoming states, outgoing states, and the final global state.
    """
    # Ensure floating-point dtype and preserve device without mutating inputs
    dtype = (
        initial_state.dtype
        if initial_state.is_floating_point()
        else torch.float32
    )
    device = initial_state.device

    M = transition_matrices.to(device=device, dtype=dtype)
    U = local_contributions.to(device=device, dtype=dtype)
    S = initial_state.to(device=device, dtype=dtype)

    num_segments = M.shape[0]
    incoming_list = []
    outgoing_list = []

    curr_state = S
    for i in range(num_segments):
        incoming_list.append(curr_state)
        # Apply F_i(S) = M_i @ S + U_i
        next_state = torch.matmul(M[i], curr_state) + U[i]
        outgoing_list.append(next_state)
        curr_state = next_state

    incoming_states = torch.stack(incoming_list, dim=0)
    outgoing_states = torch.stack(outgoing_list, dim=0)
    final_state = curr_state

    return {
        "incoming_states": incoming_states,
        "outgoing_states": outgoing_states,
        "final_state": final_state,
    }
