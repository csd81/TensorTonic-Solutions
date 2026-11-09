from collections import Counter
import math

def eval_metrics(candidate: list, references: list, max_n: int = 4) -> dict:
    """
    Returns a dict of BLEU, precisions, lengths and ROUGE precision/recall/F1 metrics.
    """
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    cand_len = len(candidate)
    ref_lens = [len(ref) for ref in references]

    closest_ref_len = min(ref_lens, key=lambda r: (abs(r - cand_len), r))

    if cand_len >= closest_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - closest_ref_len / cand_len)

    precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = get_ngrams(candidate, n)
        if not cand_ngrams:
            precisions.append(0.0)
            continue
        cand_counts = Counter(cand_ngrams)

        max_ref_counts = Counter()
        for ref in references:
            ref_counts = Counter(get_ngrams(ref, n))
            for ng in ref_counts:
                max_ref_counts[ng] = max(max_ref_counts[ng], ref_counts[ng])

        clipped = sum(min(cand_counts[ng], max_ref_counts.get(ng, 0))
                      for ng in cand_counts)
        total = sum(cand_counts.values())
        precisions.append(clipped / total if total > 0 else 0.0)

    if any(p == 0 for p in precisions):
        bleu = 0.0
    else:
        log_avg = sum(math.log(p) for p in precisions) / max_n
        bleu = bp * math.exp(log_avg)

    def rouge_n_scores(cand_tokens, ref_tokens, n):
        cand_ng = Counter(get_ngrams(cand_tokens, n))
        ref_ng = Counter(get_ngrams(ref_tokens, n))
        matched = sum(min(cand_ng[ng], ref_ng[ng]) for ng in cand_ng)
        cand_total = sum(cand_ng.values())
        ref_total = sum(ref_ng.values())
        p = matched / cand_total if cand_total > 0 else 0.0
        r = matched / ref_total if ref_total > 0 else 0.0
        total = cand_total + ref_total
        f1 = 2 * matched / total if total > 0 else 0.0
        return {'precision': p, 'recall': r,
                'f1': f1}

    def lcs_length(a, b):
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    def rouge_l_scores(cand_tokens, ref_tokens):
        lcs = lcs_length(cand_tokens, ref_tokens)
        p = lcs / len(cand_tokens) if len(cand_tokens) > 0 else 0.0
        r = lcs / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
        total = len(cand_tokens) + len(ref_tokens)
        f1 = 2 * lcs / total if total > 0 else 0.0
        return {'precision': p, 'recall': r,
                'f1': f1}

    best_r1 = max((rouge_n_scores(candidate, ref, 1)
                    for ref in references), key=lambda x: x['f1'])
    best_r2 = max((rouge_n_scores(candidate, ref, 2)
                    for ref in references), key=lambda x: x['f1'])
    best_rl = max((rouge_l_scores(candidate, ref)
                    for ref in references), key=lambda x: x['f1'])

    return {
        'bleu': round(bleu, 4),
        'brevity_penalty': round(bp, 4),
        'precisions': [round(p, 4) for p in precisions],
        'candidate_length': cand_len,
        'closest_ref_length': closest_ref_len,
        'rouge_1': {key: round(value, 4) for key, value in best_r1.items()},
        'rouge_2': {key: round(value, 4) for key, value in best_r2.items()},
        'rouge_l': {key: round(value, 4) for key, value in best_rl.items()},
    }