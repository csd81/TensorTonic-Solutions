from collections import Counter
import numpy as np


def tfidf_vectorizer(documents: list[str]) -> dict:
    """
    Returns a dictionary with tfidf_matrix and vocabulary.
    """
    # 1. Tokenize documents (lowercase + whitespace split)
    tokenized_docs = [doc.lower().split() for doc in documents]
    n_docs = len(tokenized_docs)

    # 2. Build alphabetically sorted unique vocabulary
    vocabulary = sorted({token for tokens in tokenized_docs for token in tokens})
    vocab_size = len(vocabulary)
    term_to_idx = {term: idx for idx, term in enumerate(vocabulary)}

    # 3. Compute Term Frequency (TF) matrix of shape (N, V)
    tf = np.zeros((n_docs, vocab_size), dtype=np.float64)
    for doc_idx, tokens in enumerate(tokenized_docs):
        doc_len = len(tokens)
        if doc_len == 0:
            continue
        counts = Counter(tokens)
        for term, count in counts.items():
            tf[doc_idx, term_to_idx[term]] = count / doc_len

    # 4. Compute Document Frequency (DF) and unsmoothed natural-log IDF
    df = np.count_nonzero(tf > 0, axis=0)
    idf = np.log(n_docs / df)

    # 5. Multiply TF by IDF via broadcasting
    tfidf_matrix = tf * idf

    return {
        "tfidf_matrix": tfidf_matrix,
        "vocabulary": vocabulary,
    }
