import numpy as np

def bert_embeddings(
    token_ids: np.ndarray,
    segment_ids: np.ndarray,
    token_embeddings: np.ndarray,
    position_embeddings: np.ndarray,
    segment_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Returns the float64 BERT input embeddings with shape (B, S, H).
    """
    _, S = token_ids.shape

    # Look up token and segment embeddings: both yield shape (B, S, H)
    tok_emb = token_embeddings[token_ids]
    seg_emb = segment_embeddings[segment_ids]

    # Slice the first S position vectors: yields shape (S, H)
    pos_emb = position_embeddings[:S]

    # Sum all three; NumPy automatically broadcasts (S, H) across batch dimension B
    result = tok_emb + pos_emb + seg_emb

    return result.astype(np.float64)
