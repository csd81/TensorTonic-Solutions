import torch

def embedding_head(token_states: torch.Tensor, attention_mask: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    """
    Returns the unit-norm embeddings with shape (B, d).
    """
    # Expand attention_mask to match the last dimension of token_states (B, L, 1)
    mask_expanded = attention_mask.unsqueeze(-1).to(token_states.dtype)
    
    # Sum the real token states: (B, L, d_M)
    sum_states = (token_states * mask_expanded).sum(dim=1)
    
    # Count real tokens per batch item: (B, 1)
    real_token_counts = mask_expanded.sum(dim=1).clamp(min=1.0)
    
    # Compute mean pooled vector: (B, d_M)
    pooled_vector = sum_states / real_token_counts
    
    # Apply linear projection: (B, d)
    # projection has shape (d, d_M), so pooled_vector @ projection.T
    projected = pooled_vector @ projection.transpose(-1, -2)
    
    # Normalize to unit length along the last dimension
    output = torch.nn.functional.normalize(projected, p=2, dim=-1)
    
    return output
