# <span style="font-size: 20px;">CRF Layer</span>

<span style="font-size: 14px;">A linear-chain Conditional Random Field (CRF) models dependencies between adjacent labels in a sequence. Unlike independent classifiers that predict each position separately, a CRF jointly optimizes the entire label sequence by incorporating transition scores between tags. This is critical for sequence labeling tasks (NER, POS tagging) where label constraints exist (e.g., I-PER cannot follow B-LOC).</span>

---

## <span style="font-size: 16px;">CRF Scoring</span>

<span style="font-size: 14px;">Given a sequence of $N$ tokens with $K$ possible tags, the CRF uses two score matrices:</span>

* <span style="font-size: 14px;">**Emission scores** $E$ of shape $(N, K)$: $E[i][k]$ is the score for assigning tag $k$ to position $i$</span>
* <span style="font-size: 14px;">**Transition scores** $T$ of shape $(K, K)$: $T[j][k]$ is the score for transitioning from tag $j$ to tag $k$</span>

<span style="font-size: 14px;">The score of a tag sequence $y = (y_1, \ldots, y_N)$ is:</span>

$$S(y) = \sum_{i=1}^{N} E[i][y_i] + \sum_{i=2}^{N} T[y_{i-1}][y_i]$$

---

## <span style="font-size: 16px;">Viterbi Decoding</span>

<span style="font-size: 14px;">Find the highest-scoring tag sequence using dynamic programming:</span>

$$V[1][k] = E[1][k] \quad \text{for all } k$$

$$V[i][k] = E[i][k] + \max_{j} (V[i-1][j] + T[j][k]) \quad \text{for } i = 2, \ldots, N$$

<span style="font-size: 14px;">Track backpointers to recover the best path. The best final tag is $\arg\max_k V[N][k]$, then trace back through the backpointers.</span>

---

## <span style="font-size: 16px;">Forward Algorithm (Partition Function)</span>

<span style="font-size: 14px;">The partition function $Z$ sums the scores of all possible tag sequences using log-sum-exp for numerical stability:</span>

$$\alpha[1][k] = E[1][k]$$

$$\alpha[i][k] = E[i][k] + \text{logsumexp}_j(\alpha[i-1][j] + T[j][k])$$

$$\log Z = \text{logsumexp}_k(\alpha[N][k])$$

<span style="font-size: 14px;">where $\text{logsumexp}(x_1, \ldots, x_n) = \log(\sum_i e^{x_i})$, computed as $m + \log(\sum_i e^{x_i - m})$ with $m = \max(x_i)$.</span>

---

## <span style="font-size: 16px;">Log-Likelihood</span>

<span style="font-size: 14px;">Given the gold tag sequence $y^*$, the log-likelihood is:</span>

$$\log P(y^* \mid x) = S(y^*) - \log Z$$

<span style="font-size: 14px;">Training maximizes this quantity. A higher score for the gold sequence relative to the partition function means the model assigns more probability to the correct labeling.</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

* <span style="font-size: 14px;">**Why CRF over softmax per position?** Softmax makes independent predictions and can produce invalid sequences (e.g., I-PER following B-LOC). The CRF transition matrix penalizes invalid transitions and rewards valid ones, producing globally consistent label sequences</span>
* <span style="font-size: 14px;">**Why log-sum-exp?** Directly summing $e^{S(y)}$ over all sequences causes overflow. Working in log-space and using the max-subtraction trick keeps values numerically stable. The forward algorithm computes this efficiently in $O(NK^2)$ instead of enumerating all $K^N$ sequences</span>
* <span style="font-size: 14px;">**Viterbi vs forward algorithm?** Viterbi uses max (finds the single best sequence). The forward algorithm uses log-sum-exp (sums over all sequences). They have identical structure, differing only in the aggregation operation</span>
* <span style="font-size: 14px;">**How does BiLSTM-CRF work?** The BiLSTM produces emission scores (contextual representations projected to tag space). The CRF layer adds transition scores and performs Viterbi decoding at inference time. During training, the CRF loss uses the forward algorithm to compute the partition function</span>

---