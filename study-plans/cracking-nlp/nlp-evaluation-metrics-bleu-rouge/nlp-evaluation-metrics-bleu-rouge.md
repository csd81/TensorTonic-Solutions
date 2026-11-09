# <span style="font-size: 20px;">Evaluation Metrics (BLEU, ROUGE)</span>

## BLEU (Bilingual Evaluation Understudy)

BLEU measures how many n-grams in the candidate text appear in the reference(s). It was originally designed for machine translation but is widely used for any text generation task.

### Modified N-gram Precision

For each n-gram order (1 through max_n), BLEU computes a **modified precision**: the count of candidate n-grams that appear in the reference, clipped to avoid over-counting repeated n-grams.

For each candidate n-gram, the clipped count is:

$$\text{count}_{\text{clip}} = \min(\text{count}_{\text{candidate}}, \max_{\text{ref}} \text{count}_{\text{ref}})$$

Modified precision for order n:

$$p_n = \frac{\sum_{\text{n-gram}} \text{count}_{\text{clip}}}{\sum_{\text{n-gram}} \text{count}_{\text{candidate}}}$$

### Brevity Penalty

Short candidates can achieve high precision by containing only "safe" n-grams. The brevity penalty penalizes candidates shorter than the closest reference:

$$BP = \begin{cases} 1 & \text{if } c \geq r \\ \exp(1 - r/c) & \text{if } c < r \end{cases}$$

where c is candidate length and r is the closest reference length (tie-break: shorter reference).

### Final BLEU Score

$$\text{BLEU} = BP \cdot \exp\left(\frac{1}{N}\sum_{n=1}^{N} \log p_n\right)$$

If any $p_n = 0$, BLEU is 0 (the geometric mean of precisions is 0).

## ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

ROUGE measures overlap between candidate and reference, emphasizing recall. It is the standard metric for summarization.

### ROUGE-N

ROUGE-N computes n-gram overlap using multiset intersection:

$$\text{precision} = \frac{\text{matched n-grams}}{|\text{candidate n-grams}|}, \quad \text{recall} = \frac{\text{matched n-grams}}{|\text{reference n-grams}|}$$

$$F_1 = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$$

ROUGE-1 (unigrams) captures content overlap; ROUGE-2 (bigrams) captures fluency and word order.

### ROUGE-L

ROUGE-L uses the **longest common subsequence** (LCS) between candidate and reference. Unlike n-gram methods, LCS captures sentence-level structure without requiring consecutive matches:

$$\text{precision} = \frac{\text{LCS length}}{|\text{candidate}|}, \quad \text{recall} = \frac{\text{LCS length}}{|\text{reference}|}$$

The LCS is computed via dynamic programming in O(m * n) time.

### Multiple References

With multiple references, ROUGE computes scores against each reference independently and takes the maximum F1. BLEU handles multiple references differently: the n-gram clipping uses the maximum count across all references jointly.

---

## Common Interview Follow-ups

### Why does BLEU use precision while ROUGE uses recall?

BLEU was designed for machine translation where outputs should be precise (every generated word should be justified by the reference). ROUGE was designed for summarization where outputs should cover the reference content (recall). In practice, both report F1 scores, but BLEU's core mechanism is precision-based with a brevity penalty serving as a recall proxy.

### What are the limitations of BLEU?

BLEU correlates poorly with human judgment at the sentence level (it works better corpus-wide). It ignores synonyms, paraphrases, and semantic meaning. A translation can be perfectly valid but score 0 BLEU if it uses different words. It also weights all n-grams equally regardless of importance.

### Why use the geometric mean in BLEU instead of arithmetic?

The geometric mean penalizes low precision at any n-gram order more heavily than the arithmetic mean. If 4-gram precision is 0 (no 4-gram matches), BLEU is 0 regardless of unigram precision. This reflects the intuition that good translations need both word-level (unigram) and phrase-level (higher n-gram) accuracy.

### How does ROUGE-L differ from ROUGE-2?

ROUGE-2 requires exact bigram matches; ROUGE-L allows gaps in the matching. For "the big cat sat" vs "the cat sat down", ROUGE-2 misses "the cat" (not a bigram in the candidate), but ROUGE-L finds the subsequence "the cat sat" (length 3). ROUGE-L is better at capturing reordering but less sensitive to local word order.

### What is the modified precision clipping for?

Without clipping, a candidate like "the the the the" would get perfect unigram precision against any reference containing "the". Clipping limits each n-gram's count to the maximum times it appears in any single reference. This prevents degenerate candidates from gaming the metric.
