from collections import Counter
import math

def document_ranking(documents: list, query: list, k1: float = 1.5, b: float = 0.75) -> dict:
    """
    Ranks tokenized documents for a query using the BM25 relevance scoring algorithm.
    Repeated query tokens count once; document token frequencies retain repetitions.
    """
    N = len(documents)
    if N == 0:
        return {
            "scores": [],
            "ranking": [],
            "idf_values": {},
            "avg_doc_length": 0.0,
            "term_contributions": {}
        }
    
    # Calculate document lengths and average length
    doc_lengths = [len(doc) for doc in documents]
    avg_doc_length = sum(doc_lengths) / N
    
    # Extract distinct query terms sorted alphabetically to keep consistent ordering
    unique_query_terms = sorted(list(set(query)))
    
    # Compute Document Frequency (n(t)) for each term
    df = Counter()
    for doc in documents:
        for term in set(doc):
            df[term] += 1
            
    # Calculate IDF for each unique query term
    idf_values = {}
    for t in unique_query_terms:
        nt = df[t]
        idf = math.log(1 + (N - nt + 0.5) / (nt + 0.5))
        idf_values[t] = idf
        
    # Initialize trackers for scores and term contributions
    term_contributions = {t: [] for t in unique_query_terms}
    unrounded_scores = [0.0] * N
    
    # Compute BM25 scores and contributions per document
    for i, doc in enumerate(documents):
        doc_len = len(doc)
        tf = Counter(doc)
        for t in unique_query_terms:
            freq = tf[t]
            if freq > 0:
                denom = freq + k1 * (1 - b + b * doc_len / avg_doc_length)
                contrib = idf_values[t] * (freq * (k1 + 1)) / denom
            else:
                contrib = 0.0
            
            term_contributions[t].append(contrib)
            unrounded_scores[i] += contrib
            
    # Rank indices based on unrounded scores descending, breaking ties by smaller document index
    ranking = sorted(range(N), key=lambda i: (-unrounded_scores[i], i))
    
    # Round all requested float metrics to 4 decimal places
    rounded_scores = [round(s, 4) for s in unrounded_scores]
    rounded_idf = {t: round(v, 4) for t, v in idf_values.items()}
    rounded_term_contrib = {t: [round(v, 4) for v in l] for t, l in term_contributions.items()}
    
    return {
        "scores": rounded_scores,
        "ranking": ranking,
        "idf_values": rounded_rounded_idf if 'rounded_rounded_idf' in locals() else rounded_idf,
        "avg_doc_length": round(avg_doc_length, 4),
        "term_contributions": rounded_term_contrib
    }
