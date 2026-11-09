# <span style="font-size: 20px;">Beam Search Decoding</span>

---

## <span style="font-size: 16px;">Core Concept</span>

<span style="font-size: 14px;">Beam search is a heuristic search algorithm that explores multiple candidate sequences in parallel during text generation. Instead of greedily picking the single most likely token at each step, beam search maintains the top-k highest-scoring partial sequences (called "beams") and expands all of them, keeping only the best k at each step. This helps avoid locally optimal but globally suboptimal sequences that greedy decoding often produces.</span>

---

## <span style="font-size: 16px;">How Beam Search Works</span>

Start with the start token and score zero. Each round extends active beams using the outgoing entries for their final token in the supplied probability table. No transformer forward pass is required.

Rank all extensions together and keep at most the beam width. Equal scores preserve generation order. Save retained end-token candidates as completed sequences; only the retained unfinished candidates expand next round.

The cumulative score is

$$
s(y_0,\ldots,y_t)=\sum_{i=1}^{t}\ln P(y_i\mid y_{i-1}).
$$

Here, y with subscript zero is the start token, subsequent y values are generated tokens, i indexes transitions, t is the number of generated tokens, P is the supplied transition probability, and s is the natural-log score. The end-token probability contributes to the score. This task applies no length normalization.

The length limit includes the start token and any end token. Prefer the highest-scoring saved completion, even when an unfinished beam scores higher. If no completion exists, return the best remaining unfinished beam, or the start token alone if none remains. Return one token list, retaining the start token and omitting a terminating end token.

---

## <span style="font-size: 16px;">Underlying Transformer</span>

In a language model, a transformer can supply next-token probabilities for each prefix. This exercise supplies a first-order transition table instead, so predictions depend only on the last token. Model inference is background context, not part of the implementation.

---

## <span style="font-size: 16px;">Beam Search vs Greedy Decoding</span>

<span style="font-size: 14px;">Greedy decoding picks the single highest-probability token at each step (beam_width=1). This is fast but can miss better sequences where an individually less probable token leads to a higher overall sequence probability. Beam search with width k explores k paths simultaneously, giving a better approximation to the true most-likely sequence at the cost of k times more computation per step.</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

### <span style="font-size: 14px;">Why not just use exhaustive search?</span>
<span style="font-size: 14px;">Exhaustive search over all $V^T$ possible sequences is computationally impossible for real vocabularies. Beam search is a tractable approximation that keeps only the most promising candidates.</span>

### <span style="font-size: 14px;">What is length normalization?</span>
<span style="font-size: 14px;">Longer sequences accumulate more negative log-probability terms, biasing beam search toward shorter outputs. Length normalization divides the score by $T^\alpha$ (where $\alpha \in [0.5, 1.0]$) to correct this bias.</span>

### <span style="font-size: 14px;">How does beam width affect quality?</span>
<span style="font-size: 14px;">Larger beam width explores more candidates and generally finds higher-scoring sequences, but with diminishing returns. Very large beams can sometimes produce less diverse or less fluent text due to the "beam search curse" - the highest-probability sequence is not always the most natural one.</span>

### <span style="font-size: 14px;">When is beam search preferred over sampling?</span>
<span style="font-size: 14px;">Beam search is preferred for tasks requiring precision (translation, summarization) where finding the most likely output matters. Sampling-based methods are preferred for creative tasks (story generation) where diversity matters more than exact likelihood.</span>

---
