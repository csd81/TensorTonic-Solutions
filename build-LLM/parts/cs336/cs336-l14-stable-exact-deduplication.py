import hashlib
import re
import unicodedata


def stable_exact_deduplication(
    documents: list,
    lowercase: bool = True,
    collapse_whitespace: bool = True,
    hash_bits: int = 64,
) -> dict:
    """
    Returns a dict: retained_ids (list of strings), removed_to_retained (dict of string IDs).
    """
    retained_ids = []
    removed_to_retained = {}
    
    # Maps bucket_key -> dict of {normalized_text: earliest_retained_id}
    buckets = {}
    mask = (1 << hash_bits) - 1

    for doc in documents:
        doc_id = doc["id"]
        text = doc["text"]

        # 1. Always apply Unicode NFKC normalization first
        norm_text = unicodedata.normalize("NFKC", text)

        # 2. Apply Unicode case folding if enabled
        if lowercase:
            norm_text = norm_text.casefold()

        # 3. Trim and collapse runs of Unicode whitespace into a single ASCII space
        if collapse_whitespace:
            norm_text = re.sub(r"\s+", " ", norm_text).strip()

        # 4. Compute SHA-256 digest and extract the low-order hash_bits
        digest_bytes = hashlib.sha256(norm_text.encode("utf-8")).digest()
        digest_int = int.from_bytes(digest_bytes, byteorder="big", signed=False)
        bucket_key = digest_int & mask

        # 5. Check bucket for an exact normalized text match
        bucket = buckets.setdefault(bucket_key, {})
        if norm_text in bucket:
            removed_to_retained[doc_id] = bucket[norm_text]
        else:
            bucket[norm_text] = doc_id
            retained_ids.append(doc_id)

    return {
        "retained_ids": retained_ids,
        "removed_to_retained": removed_to_retained,
    }
