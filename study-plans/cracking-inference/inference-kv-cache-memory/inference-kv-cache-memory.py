import torch

def kv_cache_memory_bytes(
    batch_size: int,
    seq_len: int,
    num_layers: int,
    num_query_heads: int,
    gqa_kv_heads: int,
    head_dim: int,
    mla_latent_dim: int,
    mla_rotary_key_dim: int,
    bytes_per_element: int,
) -> torch.Tensor:
    """
    Returns an int64 tensor of byte counts ordered [MHA, MQA, GQA, MLA].
    """
    # Base multiplier for Standard Attention Mechanisms (MHA, MQA, GQA)
    # Formula: 2 * B * S * L * H * D * b
    base_factor = 2 * batch_size * seq_len * num_layers * head_dim * bytes_per_element
    
    mha_mem = base_factor * num_query_heads
    mqa_mem = base_factor * 1
    gqa_mem = base_factor * gqa_kv_heads
    
    # Formula for Multi-Head Latent Attention (MLA)
    # Formula: B * S * L * (Dc + Dr) * b
    mla_mem = batch_size * seq_len * num_layers * (mla_latent_dim + mla_rotary_key_dim) * bytes_per_element
    
    return torch.tensor([mha_mem, mqa_mem, gqa_mem, mla_mem], dtype=torch.int64)
