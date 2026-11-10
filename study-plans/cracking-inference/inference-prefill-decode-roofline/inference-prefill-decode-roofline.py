import torch

def analyze_prefill_decode_roofline(
    d_model: int,
    num_layers: int,
    batch_size: int,
    prompt_length: int,
    generated_length: int,
    element_size: int,
    peak_flops: float,
    memory_bandwidth: float,
) -> torch.Tensor:
    """
    Returns a float64 (2, 5) tensor: prefill/decode rows; FLOPs, bytes, FLOP/byte, FLOP/s, seconds.
    """
    # Estimate parameter count
    N = 12 * num_layers * (d_model ** 2)
    
    # Prefill calculations
    T_prefill = batch_size * prompt_length
    F_prefill = 2 * N * T_prefill
    M_prefill = N * element_size
    I_prefill = F_prefill / M_prefill
    A_prefill = min(peak_flops, memory_bandwidth * I_prefill)
    t_prefill = F_prefill / A_prefill
    
    # Decode calculations
    T_decode = batch_size * generated_length
    F_decode = 2 * N * T_decode
    M_decode = N * element_size * generated_length
    I_decode = F_decode / M_decode
    A_decode = min(peak_flops, memory_bandwidth * I_decode)
    t_decode = F_decode / A_decode
    
    # Construct the result tensor
    res = [
        [float(F_prefill), float(M_prefill), float(I_prefill), float(A_prefill), float(t_prefill)],
        [float(F_decode), float(M_decode), float(I_decode), float(A_decode), float(t_decode)]
    ]
    
    return torch.tensor(res, dtype=torch.float64)
