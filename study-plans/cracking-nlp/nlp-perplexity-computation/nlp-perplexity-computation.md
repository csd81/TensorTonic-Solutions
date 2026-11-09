# <span style="font-size: 20px;">Perplexity Computation</span>

---

## <span style="font-size: 16px;">Core Concept</span>

<span style="font-size: 14px;">Perplexity measures how well a language model predicts a sequence of tokens. A lower perplexity means the model assigns higher probability to the observed sequence - it is less "surprised" by the data. Perplexity is the standard evaluation metric for language models and is defined as the exponential of the average negative log-likelihood per token.</span>

---

## <span style="font-size: 16px;">Mathematical Definition</span>

<span style="font-size: 14px;">Given a sequence $t_0, t_1, \ldots, t_{N-1}$ and a language model that produces $P(t_{i+1} \mid t_0, \ldots, t_i)$ at each position:</span>

$$
\text{PPL} = \exp\left(-\frac{1}{N-1} \sum_{i=0}^{N-2} \log P(t_{i+1} \mid t_0, \ldots, t_i)\right)
$$

<span style="font-size: 14px;">The sum runs over $N-1$ predictions because the first token has no preceding context to predict it from. Each term $-\log P$ is the negative log-likelihood (NLL), also called the cross-entropy loss at that position.</span>

---

## <span style="font-size: 16px;">Computing Perplexity with a Causal Transformer</span>

### <span style="font-size: 14px;">Single Forward Pass</span>

<span style="font-size: 14px;">A causal transformer with a causal mask processes all $N$ tokens in parallel. The mask ensures position $i$ only attends to positions $0, 1, \ldots, i$, so the output at position $i$ depends only on the current and past tokens. This means a single forward pass produces logits at all $N$ positions simultaneously.</span>

### <span style="font-size: 14px;">Extracting Next-Token Probabilities</span>

<span style="font-size: 14px;">The logits at position $i$ represent the model's prediction for token $i+1$. Apply softmax to get a probability distribution over the vocabulary, then extract the probability assigned to the actual next token $t_{i+1}$. Positions $0$ through $N-2$ yield $N-1$ next-token probabilities.</span>

### <span style="font-size: 14px;">From Probabilities to Perplexity</span>

<span style="font-size: 14px;">Take the negative log of each probability to get the NLL. Average over all $N-1$ positions. Exponentiate the average NLL to get perplexity. A perplexity of $k$ means the model is as uncertain as if it were choosing uniformly among $k$ options at each step.</span>

---

## <span style="font-size: 16px;">Interpreting Perplexity</span>

<span style="font-size: 14px;">For a vocabulary of size $V$, uniform predictions have perplexity $V$. A perfect model that always predicts the correct next token with probability 1 has perplexity 1. Uniform predictions provide a baseline, not an upper bound: assigning lower probability to the observed tokens can produce perplexity greater than the vocabulary size. GPT-2 achieves perplexity around 20-30 on standard benchmarks, meaning it is roughly as uncertain as choosing from 20-30 equally likely tokens at each step.</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

<span style="font-size: 14px;">1. **Why exponentiate the average NLL?** Working in log-space (NLL) avoids numerical underflow from multiplying many small probabilities. Exponentiating converts back to a human-interpretable scale.</span>

<span style="font-size: 14px;">2. **Perplexity vs cross-entropy**: Perplexity is simply $\exp(\text{cross-entropy})$. They contain the same information, but perplexity has a more intuitive interpretation as an "effective vocabulary size."</span>

<span style="font-size: 14px;">3. **Why causal masking matters**: Without the causal mask, position $i$ can attend to future tokens, leaking information about $t_{i+1}$ into the prediction of $t_{i+1}$. This makes perplexity artificially low and meaningless.</span>

<span style="font-size: 14px;">4. **Bits per character (BPC)**: An alternative metric that uses $\log_2$ instead of $\ln$. BPC = $\text{avg\_nll} / \ln(2)$.</span>

<span style="font-size: 14px;">5. **Perplexity across datasets**: Perplexity depends heavily on the data domain and tokenization. A model's perplexity on Wikipedia is not comparable to its perplexity on code or dialogue.</span>

---