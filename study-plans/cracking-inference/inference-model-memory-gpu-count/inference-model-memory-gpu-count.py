import math
import torch

def estimate_inference_gpu_count(
    parameter_count: int,
    bytes_per_parameter: float,
    kv_cache_bytes: int,
    activation_bytes: int,
    runtime_overhead_fraction: float,
    usable_bytes_per_gpu: int,
) -> torch.Tensor:
    """
    Returns an int64 (2,) tensor: required bytes, minimum GPU count.
    """
    # Follow Python float (binary64) arithmetic exactly as requested
    raw_bytes = (parameter_count * bytes_per_parameter + kv_cache_bytes + activation_bytes) * (1.0 + runtime_overhead_fraction)
    
    # Apply ceiling to find the total required bytes
    required_bytes = math.ceil(raw_bytes)
    
    # Apply ceiling to find the minimum number of GPUs
    gpu_count = math.ceil(required_bytes / usable_bytes_per_gpu)
    
    return torch.tensor([required_bytes, gpu_count], dtype=torch.int64)
