# <span style="font-size: 20px;">Spell Correction</span>

<span style="font-size: 14px;">This exercise selects the dictionary word with the smallest Levenshtein distance, breaking ties by higher frequency and then Python string order. It returns one correction per input word. The noisy channel model below is optional background with a different ranking objective, not the rule assessed here.</span>

---

## <span style="font-size: 16px;">Optional Background: Noisy Channel Model</span>

<span style="font-size: 14px;">By Bayes' rule:</span>

$$\hat{w} = \arg\max_{w \in V} P(w \mid x) = \arg\max_{w \in V} P(x \mid w) \cdot P(w)$$

<span style="font-size: 14px;">where $P(w)$ is the **language model** (prior probability of word $w$) and $P(x \mid w)$ is the **channel model** (probability that $w$ is corrupted into $x$).</span>

---

## <span style="font-size: 16px;">Language Model (Prior)</span>

<span style="font-size: 14px;">Use unigram probabilities from a word frequency dictionary:</span>

$$P(w) = \frac{\text{count}(w)}{\sum_{w' \in V} \text{count}(w')}$$

<span style="font-size: 14px;">Words not in the dictionary have $P(w) = 0$ and are never candidates.</span>

---

## <span style="font-size: 16px;">Channel Model (Error Probability)</span>

<span style="font-size: 14px;">The channel probability is based on minimum edit distance (Levenshtein distance) between the misspelling and the candidate word. Each edit operation (insertion, deletion, substitution) has cost 1.</span>

$$P(x \mid w) = \alpha^{d(x, w)}$$

<span style="font-size: 14px;">where $d(x, w)$ is the edit distance between $x$ and $w$, and $\alpha \in (0, 1)$ is a per-edit penalty (typically 0.05). Words with more edits are exponentially less likely to be the source. A noisy-channel extension may cap the candidate edit distance; this exercise considers every dictionary word.</span>

---

## <span style="font-size: 16px;">Candidate Generation</span>

<span style="font-size: 14px;">An optional noisy-channel extension can restrict candidates by a chosen edit-distance limit and compute the following score:</span>

$$\text{score}(w) = P(x \mid w) \cdot P(w) = \alpha^{d(x, w)} \cdot P(w)$$

<span style="font-size: 14px;">In this optional extension, candidates are ranked by score, with alphabetical tie-breaking, and multiple suggestions may be returned. This exercise instead returns one word using distance, frequency, and word order.</span>

---

## <span style="font-size: 16px;">Edit Distance (Levenshtein)</span>

<span style="font-size: 14px;">The minimum number of single-character insertions, deletions, or substitutions to transform string $a$ into string $b$. Computed via dynamic programming:</span>

$$D[i][j] = \begin{cases} j & \text{if } i = 0 \\ i & \text{if } j = 0 \\ D[i-1][j-1] & \text{if } a[i] = b[j] \\ 1 + \min(D[i-1][j], D[i][j-1], D[i-1][j-1]) & \text{otherwise} \end{cases}$$

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

* <span style="font-size: 14px;">**Why noisy channel over edit distance alone?** Pure edit distance picks the closest word but ignores word frequency. "the" is more likely than "thee" even at the same distance. The language model prior captures this</span>
* <span style="font-size: 14px;">**Why exponential decay for the channel model?** Each additional edit makes the corruption less probable. The exponential form $\alpha^d$ is a simple parameterization where $\alpha$ controls how aggressively we penalize distant words. Real systems learn edit probabilities from data</span>
* <span style="font-size: 14px;">**Why might an extension cap edit distance?** Edit distance computation is expensive. Most real spelling errors involve 1-2 edits. Capping at 2-3 edits keeps candidate generation tractable while covering the vast majority of errors</span>
* <span style="font-size: 14px;">**How would you extend this to context?** Replace the unigram prior with a bigram or n-gram model: $P(w_i \mid w_{i-1})$. This lets the model prefer corrections that make sense in context, e.g., correcting "teh" to "the" before a noun</span>

---