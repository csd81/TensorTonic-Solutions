import math
from collections import Counter
import numpy as np

def bm25_score(query_tokens: list[str], docs: list[list[str]], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    """
    Returns a NumPy array with one score per document.
    """
    N = len(docs)
    if N == 0:
        return np.array([], dtype=float)

    unique_query = set(query_tokens)
    if not unique_query:
        return np.zeros(N, dtype=float)

    doc_counters = [Counter(doc) for doc in docs]
    doc_lengths = np.array([len(doc) for doc in docs], dtype=float)
    avgdl = float(np.mean(doc_lengths))

    # Count document frequency df(t) only for terms present in the query
    df = {
        term: sum(1 for counter in doc_counters if term in counter)
        for term in unique_query
    }

    # Precompute IDF(t) for each unique query term
    idf = {
        term: math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
        for term in unique_query
    }

    scores = np.zeros(N, dtype=float)

    for i, counter in enumerate(doc_counters):
        dl = doc_lengths[i]
        if dl == 0:
            continue

        norm = k1 * (1.0 - b + b * (dl / avgdl))
        doc_score = 0.0

        for term in unique_query:
            tf = counter.get(term, 0)
            if tf > 0:
                doc_score += idf[term] * (tf * (k1 + 1.0)) / (tf + norm)

        scores[i] = doc_score

    return scores
