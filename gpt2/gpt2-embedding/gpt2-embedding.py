import torch

def gpt2_embedding(token_ids: torch.Tensor, token_embedding: torch.Tensor, position_embedding: torch.Tensor) -> torch.Tensor:
    """
    Returns token-plus-position embeddings with shape (sequence, model_width).
    """
    seq_len = token_ids.size(0)
    
    # 1. Look up token embeddings for each token ID
    tok_emb = token_embedding[token_ids]  # shape: (seq_len, model_width)
    
    # 2. Get positional indices from 0 to seq_len - 1
    positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.int64)
    
    # 3. Look up position embeddings and add elementwise
    pos_emb = position_embedding[positions]  # shape: (seq_len, model_width)
    
    out = tok_emb + pos_emb
    return out.to(torch.float64)
