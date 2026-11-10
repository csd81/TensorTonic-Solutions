import torch

def compute_serving_metrics(
    arrival_times: torch.Tensor,
    first_token_times: torch.Tensor,
    token_timestamps: torch.Tensor,
    output_token_counts: torch.Tensor,
) -> tuple:
    """
    Returns (ttft, tpot, mean_itl, throughput): three floating (N,) tensors and one floating scalar tensor.
    """
    # 1. Compute Time to First Token (TTFT)
    ttft = first_token_times - arrival_times
    
    # 2. Gather the last valid token timestamp for each request
    # output_token_counts is 1-indexed count, so index is count - 1
    last_token_indices = output_token_counts - 1
    last_token_times = token_timestamps[torch.arange(len(output_token_counts)), last_token_indices]
    
    # 3. Compute TPOT and Mean ITL
    # Mathematically, the sum of inter-token latencies telescopes to (last_token_time - first_token_time)
    # Therefore, mean_itl is identical to tpot for each individual request.
    num_gaps = (output_token_counts - 1).float()
    
    # Avoid division by zero for single-token requests where num_gaps == 0
    tpot = torch.where(
        num_gaps > 0, 
        (last_token_times - first_token_times) / num_gaps, 
        torch.zeros_like(num_gaps)
    )
    mean_itl = tpot.clone()
    
    # 4. Compute Aggregate Throughput
    total_tokens = output_token_counts.sum().float()
    min_arrival = arrival_times.min()
    max_completion = last_token_times.max()
    throughput = total_tokens / (max_completion - min_arrival)
    
    return ttft, tpot, mean_itl, throughput
