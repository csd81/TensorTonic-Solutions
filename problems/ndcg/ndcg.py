import math

def ndcg(relevance_scores: list, k: int) -> float:
    """
    Returns NDCG as a float.
    """
    def compute_dcg(scores: list, cutoff: int) -> float:
        dcg = 0.0
        for idx, rel in enumerate(scores[:cutoff]):
            rank = idx + 1
            dcg += (2 ** rel - 1) / math.log2(rank + 1)
        return dcg

    dcg_val = compute_dcg(relevance_scores, k)
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg_val = compute_dcg(ideal_scores, k)

    if idcg_val == 0.0:
        return 0.0

    return float(dcg_val / idcg_val)
