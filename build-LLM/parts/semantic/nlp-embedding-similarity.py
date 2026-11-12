import numpy as np

def embedding_similarity(query: list, candidates: list, metric: str = 'cosine', k: int = 1) -> list:
    """
    Returns a list of k candidate indices in ranked order.
    """
    q = np.array(query, dtype=np.float64)
    cands = [np.array(c, dtype=np.float64) for c in candidates]

    if metric == "cosine":
        q = q / np.max(np.abs(q))
        cands = [c / np.max(np.abs(c)) for c in cands]
        scores = [np.dot(q, c) / np.linalg.norm(c) for c in cands]
    elif metric == "dot":
        scores = [np.dot(q, c) for c in cands]
    elif metric == "euclidean":
        scores = [-np.linalg.norm(q - c) for c in cands]

    ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return ranked[:k]