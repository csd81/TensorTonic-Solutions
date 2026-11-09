<span style="font-size: 14px;">Span extraction is a core task in extractive question answering. Given a passage of tokens, a QA model produces two vectors of logits:</span>

- <span style="font-size: 14px;">**Start logits**: a score for each token position indicating how likely that position is the beginning of the answer</span>
- <span style="font-size: 14px;">**End logits**: a score for each token position indicating how likely that position is the end of the answer</span>

<span style="font-size: 14px;">The candidate answer span from position $i$ to position $j$ receives a combined score:</span>

$$
\text{score}(i, j) = s_i + e_j
$$

<span style="font-size: 14px;">where $s_i$ is the start logit at position $i$ and $e_j$ is the end logit at position $j$.</span>

<span style="font-size: 14px;">**Validity constraints** ensure the span is meaningful:</span>

- <span style="font-size: 14px;">The start must come before or at the end: $i \le j$</span>
- <span style="font-size: 14px;">The span length must not exceed a maximum: $j - i + 1 \le L$</span>

<span style="font-size: 14px;">**Why additive scores?** In transformer-based QA models like BERT, the start and end positions are predicted independently via separate linear heads on top of the final hidden states. The joint probability of a span factorizes as:</span>

$$
P(i, j) = P_{\text{start}}(i) \times P_{\text{end}}(j)
$$

<span style="font-size: 14px;">Taking the log converts this to an additive score, which is equivalent to summing the raw logits before softmax.</span>

<span style="font-size: 14px;">**Tie-breaking** is important when multiple spans share the maximum score. Common conventions prefer:</span>

- <span style="font-size: 14px;">Shorter spans (more precise answers)</span>
- <span style="font-size: 14px;">Earlier start positions (first occurrence)</span>

<span style="font-size: 14px;">**Complexity considerations**: A naive search over all $(i, j)$ pairs is $O(n^2)$. With a max span length constraint $L$, this reduces to $O(n \times L)$, which is practical for typical passage lengths.</span>