import torch

def latency_percentiles(latencies: torch.Tensor) -> torch.Tensor:
    """
    Returns a floating tensor of shape (3,), ordered P50, P95, P99.
    """
    # 1. Sort the input latencies in ascending order
    v = torch.sort(latencies).values.to(torch.float32)
    n = v.size(0)
    
    # 2. Define the target percentiles as a tensor
    p = torch.tensor([0.50, 0.95, 0.99], dtype=torch.float32, device=latencies.device)
    
    # 3. Calculate fractional ranks
    r = p * (n - 1)
    
    # 4. Find the floor and ceiling indices
    r_floor = torch.floor(r).long()
    r_ceil = torch.ceil(r).long()
    
    # 5. Extract values at neighboring indices
    v_floor = v[r_floor]
    v_ceil = v[r_ceil]
    
    # 6. Apply linear interpolation formula
    percentiles = v_floor + (r - r_floor.float()) * (v_ceil - v_floor)
    
    return percentiles
