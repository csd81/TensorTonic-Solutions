import math
from collections import Counter


def bleu_score(candidate: list, reference: list, max_n: int) -> float:
    """
    Returns the unsmoothed BLEU score.
    """
    c = len(candidate)
    r = len(reference)

    if c == 0:
        return 0.0

    log_precisions = []

    for n in range(1, max_n + 1):
        # If candidate has fewer than n tokens, there are no n-grams (precision = 0)
        if c < n:
            return 0.0

        cand_ngrams = Counter(
            tuple(candidate[i : i + n]) for i in range(c - n + 1)
        )
        ref_ngrams = Counter(
            tuple(reference[i : i + n]) for i in range(r - n + 1)
        )

        clipped_matches = sum(
            min(count, ref_ngrams[ng]) for ng, count in cand_ngrams.items()
        )
        total_cand_ngrams = c - n + 1

        if clipped_matches == 0:
            return 0.0

        precision = clipped_matches / total_cand_ngrams
        log_precisions.append(math.log(precision))

    # Brevity penalty
    if c >= r:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (r / c))

    # Geometric mean of precisions weighted uniformly by 1 / max_n
    geo_mean = math.exp(sum(log_precisions) / max_n)

    return float(bp * geo_mean)
