# <span style="font-size: 20px;">Tokenizer Training (Byte Pair Encoding)</span>

## Core Idea

Byte Pair Encoding (BPE) builds a subword vocabulary by iteratively merging the most frequent adjacent token pairs in a corpus. Starting from individual characters, it greedily combines pairs until reaching the desired vocabulary size $|V|$. This produces a vocabulary that balances between character-level (tiny vocab, long sequences) and word-level (huge vocab, short sequences) tokenization.

## Algorithm

### Initialization

Split each word into characters and append an end-of-word marker `</w>`:
- "low" -> ["l", "o", "w", "</w>"]
- "newest" -> ["n", "e", "w", "e", "s", "t", "</w>"]

The initial vocabulary $V_0$ is the set of all unique characters plus `</w>`, with $|V_0| = C + 1$ where $C$ is the number of distinct characters in the corpus.

### Iterative Merging

Let $W = \{(w_k, f_k)\}$ be the corpus of words $w_k$ with their frequencies $f_k$. Each word is represented as a sequence of tokens. At merge step $t$:

1. **Count pairs**: For every adjacent token pair $(a, b)$, compute the frequency-weighted count:

$$
\text{count}(a, b) = \sum_{k:\,(a,b) \in w_k} f_k \cdot n_k(a, b)
$$

where $n_k(a, b)$ is the number of times pair $(a, b)$ appears as adjacent tokens in word $w_k$.

2. **Select best**: Choose $(a^*, b^*) = \arg\max_{(a,b)} \text{count}(a, b)$. Break ties lexicographically on the pair tuple.

3. **Merge**: Replace every occurrence of the adjacent pair $(a^*, b^*)$ with the concatenated token $a^*b^*$ throughout the corpus.

4. **Update vocab**: Add $a^*b^*$ to the vocabulary: $V_t = V_{t-1} \cup \{a^*b^*\}$.

### Termination

Stop after $M$ merges (the requested number), or earlier if no adjacent pairs remain. The final vocabulary size is:

$$
|V| = C + 1 + M
$$

## Why End-of-Word Markers Matter

The `</w>` marker distinguishes word-final tokens from word-internal tokens. Without it, "un" at the end of "fun" and "un" at the start of "under" would be the same token. The marker lets the tokenizer learn that certain subwords only appear at word boundaries, improving the quality of the segmentation.

## Frequency Weighting

Pair counts must be weighted by word frequency $f_k$. If "the" appears $f_{\text{the}} = 1000$ times and "theater" appears $f_{\text{theater}} = 5$ times, the pair ("t","h") gets a count of $1000 + 5 = 1005$, not 2. This ensures common words drive the vocabulary construction, producing tokens that compress the corpus efficiently.

## Vocabulary Growth

Each merge adds exactly one new token to the vocabulary (the concatenation of the merged pair). After $M$ merges starting from $C$ initial characters plus `</w>`, the vocabulary size is $|V| = C + 1 + M$. No tokens are removed during merging.

---

## Common Interview Follow-ups

### How does BPE handle unseen words at inference time?

BPE applies the learned merge rules in priority order to any new word. Starting from characters, it greedily applies the highest-priority applicable merge, then the next, and so on. Any character present in the training data can be represented, so the only truly out-of-vocabulary characters are those never seen during training.

### What is the difference between BPE and WordPiece?

WordPiece (used in BERT) selects merges based on likelihood improvement rather than raw frequency. It chooses the pair that maximizes the likelihood ratio:

$$
\text{score}(a, b) = \frac{\text{count}(ab)}{\text{count}(a) \cdot \text{count}(b)}
$$

This is equivalent to maximizing $P(ab) / (P(a) \cdot P(b))$ under a unigram model. It tends to merge rare-but-informative pairs earlier than BPE would. In practice, both produce similar quality vocabularies.

### How do you choose the vocabulary size?

Vocabulary size $|V|$ is a hyperparameter balancing sequence length against embedding table size. Larger vocabularies produce shorter sequences (faster inference) but require more parameters (the embedding matrix has $|V| \times d$ entries, where $d$ is the embedding dimension). Typical sizes: $30{,}000$-$50{,}000$ for English models, $100{,}000$+ for multilingual models. The optimal size depends on the downstream task and compute budget.

### What is the role of pre-tokenization?

Before BPE, text is split into words using rules (whitespace, punctuation). This prevents merges across word boundaries, which would create tokens like "the_cat" that waste vocabulary entries. GPT-2 uses regex-based pre-tokenization; SentencePiece operates on raw text without pre-tokenization.

### Can BPE handle multiple languages?

Yes, but the vocabulary must cover all relevant character sets. Multilingual BPE (used in mBERT, XLM-R) trains on a balanced multilingual corpus. High-resource languages may dominate the vocabulary, so techniques like temperature-scaled sampling across languages help ensure fair coverage. Given language $l$ with corpus size $n_l$, the sampling probability is $p_l \propto n_l^{1/T}$ where $T > 1$ upweights low-resource languages.
