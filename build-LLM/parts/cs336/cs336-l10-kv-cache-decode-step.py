import torch

def kv_cache_decode_step(
    query: torch.Tensor, new_key: torch.Tensor, new_value: torch.Tensor,
    key_cache: torch.Tensor, value_cache: torch.Tensor,
    num_query_heads: int, num_kv_heads: int,
) -> dict:
    """
    Returns a dict of tensors: output, new_key_cache, new_value_cache.
    """
    orig_dtype = query.dtype
    D = query.shape[-1]
    G = num_query_heads // num_kv_heads
    
    # 1. Extend the KV Caches along the sequence dimension (S + 1)
    new_key_cache = torch.cat([key_cache, new_key.unsqueeze(2)], dim=2)
    new_value_cache = torch.cat([value_cache, new_value.unsqueeze(2)], dim=2)
    
    # 2. Reshape query to separate KV head and Group dimensions: (B, Hkv, G, D)
    B = query.shape[0]
    query_reshaped = query.view(B, num_kv_heads, G, D)
    
    # 3. Perform attention arithmetic in float32 for numerical stability
    q_f32 = query_reshaped.to(torch.float32)
    k_f32 = new_key_cache.to(torch.float32)
    v_f32 = new_value_cache.to(torch.float32)
    
    # Compute attention scores: (B, Hkv, G, S + 1)
    scores = torch.einsum("bhgd,bhsd->bhgs", q_f32, k_f32) / (D ** 0.5)
    
    # Softmax over the sequence dimension
    attn_weights = torch.softmax(scores, dim=-1)
    
    # Compute weighted sum of values: (B, Hkv, G, D)
    output_f32 = torch.einsum("bhgs,bhsd->bhgd", attn_weights, v_f32)
    
    # 4. Reshape back to the original format and restore the target dtype
    output = output_f32.view(B, num_query_heads, D).to(orig_dtype)
    
    return {
        "output": output,
        "new_key_cache": new_key_cache,
        "new_value_cache": new_value_cache
    }
