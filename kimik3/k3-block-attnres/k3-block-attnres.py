import torch

def _read_depth_sources(sources, pseudo_query, eps):
    normalized = sources / torch.sqrt(sources.square().mean(dim=-1, keepdim=True) + eps)
    logits = (normalized * pseudo_query).sum(dim=-1)
    weights = torch.softmax(logits, dim=0)
    retrieved = (weights.unsqueeze(-1) * sources).sum(dim=0)
    return retrieved, weights

def block_attention_residual(
    embedding: torch.Tensor,
    previous_outputs: torch.Tensor,
    pseudo_query: torch.Tensor,
    block_size: int,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """
    Returns a dictionary containing the retrieved values, depth weights, and block sources.
    """
    # Ensure floating-point dtype and preserve device
    dtype = embedding.dtype if torch.is_floating_point(embedding) else torch.float32
    device = embedding.device
    
    emb = embedding.to(device=device, dtype=dtype)
    query = pseudo_query.to(device=device, dtype=dtype)
    
    # Source 0 is always the embedding: shape (1, B, S, D)
    sources = [emb.unsqueeze(0)]
    
    # Handle case where previous_outputs might be passed as an empty list or empty tensor
    if not isinstance(previous_outputs, torch.Tensor):
        prev = torch.as_tensor(previous_outputs, device=device, dtype=dtype)
    else:
        prev = previous_outputs.to(device=device, dtype=dtype)
        
    num_layers = prev.shape[0] if prev.numel() > 0 else 0
    
    if num_layers > 0:
        num_full_blocks = num_layers // block_size
        remainder = num_layers % block_size
        
        # Sum each completed block of size `block_size`
        if num_full_blocks > 0:
            full_len = num_full_blocks * block_size
            # Reshape (num_full_blocks * block_size, B, S, D) -> (num_full_blocks, block_size, B, S, D)
            full_blocks = prev[:full_len].view(num_full_blocks, block_size, *emb.shape).sum(dim=1)
            sources.append(full_blocks)
            
        # Add one partial block only if leftover layers exist
        if remainder > 0:
            partial_block = prev[num_full_blocks * block_size :].sum(dim=0, keepdim=True)
            sources.append(partial_block)
            
    # Concatenate along the depth dimension: (num_sources, B, S, D)
    block_sources = torch.cat(sources, dim=0)
    
    # Perform the shared Full Attention Residuals read over constructed block sources
    retrieved_representation, attention_weights = _read_depth_sources(block_sources, query, eps)
    
    return {
        "retrieved_representation": retrieved_representation,
        "attention_weights": attention_weights,
        "block_sources": block_sources,
    }
