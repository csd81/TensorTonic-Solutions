import numpy as np

def minhash_dedup(documents: list, num_hashes: int, shingle_size: int, threshold: float, seed: int = 42) -> dict:
    """
    Identifies near-duplicate documents using MinHash over distinct character shingles.
    
    Returns a dict containing:
    - shingles: unique character shingles sorted alphabetically for each document
    - signatures: MinHash signature integer list for each document
    - estimated_similarities: [idx1, idx2, similarity] for every pair of documents
    - duplicate_pairs: [kept_idx, duplicate_idx] pairs
    - unique_indices: list of kept document indices
    """
    P = 2147483647
    rng = np.random.RandomState(seed)
    
    # Draw all a coefficients from [1, P-1], then all b coefficients from [0, P-1]
    a = rng.randint(1, P, size=num_hashes)
    b = rng.randint(0, P, size=num_hashes)
    
    shingles_all = []
    signatures_all = []
    
    for doc in documents:
        # Extract unique character shingles of size shingle_size
        shingles = set()
        for i in range(len(doc) - shingle_size + 1):
            shingles.add(doc[i:i+shingle_size])
        
        shingles_sorted = sorted(list(shingles))
        shingles_all.append(shingles_sorted)
        
        # Calculate the integer value v(s) for each unique shingle
        v_list = []
        for s in shingles_sorted:
            v = 0
            for r, char in enumerate(s):
                v += ord(char) * (256 ** (shingle_size - 1 - r))
            v %= P
            v_list.append(v)
            
        # Compute signature entries Hk(S)
        sig = []
        for ak, bk in zip(a, b):
            m = min((int(ak) * int(v) + int(bk)) % P for v in v_list)
            sig.append(int(m))
        signatures_all.append(sig)
        
    # Calculate estimated similarities for all unique pairs
    estimated_similarities = []
    n = len(documents)
    sim_matrix = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            equal_count = sum(1 for x, y in zip(signatures_all[i], signatures_all[j]) if x == y)
            sim = round(equal_count / num_hashes, 4)
            estimated_similarities.append([i, j, sim])
            sim_matrix[(i, j)] = sim
            
    # Determine unique indices and duplicate pairs sequential processing
    unique_indices = []
    duplicate_pairs = []
    
    for i in range(n):
        is_duplicate = False
        for kept_idx in unique_indices:
            if sim_matrix[(kept_idx, i)] >= threshold:
                duplicate_pairs.append([kept_idx, i])
                is_duplicate = True
                break
        if not is_duplicate:
            unique_indices.append(i)
            
    return {
        "shingles": shingles_all,
        "signatures": signatures_all,
        "estimated_similarities": estimated_similarities,
        "duplicate_pairs": duplicate_pairs,
        "unique_indices": unique_indices
    }
