import numpy as np

def cosine_topk(queries: list, corpus: list, k: int) -> dict:
    """
    Ranks corpus embeddings for each query using a stabilized cosine score and returns the top-k indices and scores.
    """
    Q = np.array(queries, dtype=np.float64)
    C = np.array(corpus, dtype=np.float64)
    eps = 1e-12
    
    # Compute L2 norms along the feature dimension
    q_norms = np.linalg.norm(Q, axis=1, keepdims=True) + eps
    c_norms = np.linalg.norm(C, axis=1, keepdims=True) + eps
    
    # Normalize the vectors
    Q_normed = Q / q_norms
    C_normed = C / c_norms
    
    # Compute similarity matrix via broadcasting to avoid matrix multiplication precision drift
    similarity_matrix = np.sum(Q_normed[:, None, :] * C_normed[None, :, :], axis=-1)
    
    # Use stable argsort on the negative similarities to rank in descending order
    order = np.argsort(-similarity_matrix, axis=1, kind="stable")[:, :k]
    
    M = Q.shape[0]
    indices = [[int(order[i, j]) for j in range(k)] for i in range(M)]
    scores = [[round(float(similarity_matrix[i, order[i, j]]), 4) for j in range(k)] for i in range(M)]
    
    return {
        "indices": indices,
        "scores": scores
    }
