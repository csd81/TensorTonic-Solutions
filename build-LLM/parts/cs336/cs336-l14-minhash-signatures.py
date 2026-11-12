import hashlib
import json
import numpy as np


def minhash_signatures(documents: list, shingle_size: int, num_hashes: int, seed: int) -> dict:
    """
    Returns a dict: signatures (uint32 array), similarities (float64 array).
    """
    num_docs = len(documents)
    max_uint32 = (1 << 32) - 1

    # Initialize signatures with 2^32 - 1 (default value for documents without shingles)
    signatures = np.full((num_docs, num_hashes), max_uint32, dtype=np.uint32)
    is_empty = np.zeros(num_docs, dtype=bool)

    # Cache shingle hashes across documents to avoid redundant SHA-256 computations
    shingle_cache = {}

    for doc_idx, tokens in enumerate(documents):
        if len(tokens) < shingle_size:
            is_empty[doc_idx] = True
            continue

        # Extract unique consecutive token windows encoded as compact JSON strings
        unique_shingles = set()
        for start in range(len(tokens) - shingle_size + 1):
            window = tokens[start : start + shingle_size]
            encoded = json.dumps(window, ensure_ascii=False, separators=(",", ":"))
            unique_shingles.add(encoded)

        # Compute or retrieve the num_hashes column values for each unique shingle
        doc_hashes = []
        for shingle in unique_shingles:
            if shingle not in shingle_cache:
                col_hashes = np.empty(num_hashes, dtype=np.uint32)
                for h in range(num_hashes):
                    payload = f"{seed}:{h}:{shingle}".encode("utf-8")
                    digest = hashlib.sha256(payload).digest()
                    col_hashes[h] = int.from_bytes(digest[:4], byteorder="big", signed=False)
                shingle_cache[shingle] = col_hashes
            doc_hashes.append(shingle_cache[shingle])

        # Take the column-wise minimum across all shingles in the document
        signatures[doc_idx] = np.min(np.vstack(doc_hashes), axis=0)

    # Compute pairwise similarity matrix
    similarities = np.zeros((num_docs, num_docs), dtype=np.float64)

    for i in range(num_docs):
        similarities[i, i] = 1.0
        for j in range(i + 1, num_docs):
            if is_empty[i] and is_empty[j]:
                sim = 1.0
            elif is_empty[i] or is_empty[j]:
                sim = 0.0
            else:
                sim = float(np.mean(signatures[i] == signatures[j]))
            similarities[i, j] = sim
            similarities[j, i] = sim

    return {
        "signatures": signatures,
        "similarities": similarities,
    }
