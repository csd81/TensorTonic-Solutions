import torch

def tensor_parallel_allreduce_cost(
    message_element_count: int,
    bytes_per_element: int,
    tensor_parallel_world_size: int,
    link_bandwidth: float,
    per_hop_latency: float,
    collective_count: int,
) -> torch.Tensor:
    """
    Returns a float64 (2,) tensor: per-rank bytes and seconds across all collectives.
    """
    # If there is only 1 rank, no network communication occurs.
    if tensor_parallel_world_size <= 1:
        return torch.tensor([0.0, 0.0], dtype=torch.float64)
        
    p = float(tensor_parallel_world_size)
    N = float(message_element_count)
    b = float(bytes_per_element)
    W = float(link_bandwidth)
    alpha = float(per_hop_latency)
    C = float(collective_count)
    
    # Calculate steps per collective
    s = 2.0 * (p - 1.0)
    
    # Calculate total communicated bytes per rank
    V = (C * s * N * b) / p
    
    # Calculate total communication time per rank
    t = C * (s * alpha + (s * N * b) / (p * W))
    
    return torch.tensor([V, t], dtype=torch.float64)
