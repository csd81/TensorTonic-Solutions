# <span style="font-size: 20px;">Language Modeling</span>

<span style="font-size: 14px;">A language model estimates the probability of a token given the preceding context: $P(w_t \mid w_1, \ldots, w_{t-1})$. Neural language models use a neural network to learn these conditional distributions from data. The simplest neural architecture for sequence modeling is the Recurrent Neural Network (RNN), which maintains a hidden state that captures information from all previously seen tokens.</span>

---

## <span style="font-size: 16px;">RNN Language Model</span>

<span style="font-size: 14px;">An RNN language model processes tokens one at a time, updating a hidden state vector at each step. The hidden state acts as a compressed summary of the sequence so far.</span>

### <span style="font-size: 14px;">Forward Pass</span>

<span style="font-size: 14px;">Given a sequence of tokens $w_1, w_2, \ldots, w_T$:</span>

$$x_t = E[w_t]$$

<span style="font-size: 14px;">where $E$ is a $(|V| \times D)$ embedding matrix that maps each token to a dense vector.</span>

$$h_t = \tanh(x_t W_x + h_{t-1} W_h + b_h)$$

<span style="font-size: 14px;">where $W_x$ is $(D \times H)$, $W_h$ is $(H \times H)$, $b_h$ is $(H,)$, and $h_0 = \mathbf{0}$.</span>

$$P(w_{t+1} \mid w_1 \ldots w_t) = \text{softmax}(h_t W_y + b_y)$$

<span style="font-size: 14px;">where $W_y$ is $(H \times |V|)$ and $b_y$ is $(|V|,)$.</span>

---

## <span style="font-size: 16px;">Teacher Forcing</span>

<span style="font-size: 14px;">During evaluation (and during training), the model receives the actual ground-truth token as input at each time step, not its own prediction from the previous step. This is called teacher forcing. At time $t$, the input is always the embedding of the actual token $w_t$, regardless of what the model predicted at the previous step.</span>

---

## <span style="font-size: 16px;">Numerically Stable Softmax</span>

<span style="font-size: 14px;">The softmax function converts logits to probabilities. Direct computation risks overflow when logits are large:</span>

$$\text{softmax}(z)_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}$$

<span style="font-size: 14px;">The numerically stable version subtracts the maximum logit before exponentiation:</span>

$$\text{softmax}(z)_i = \frac{\exp(z_i - \max(z))}{\sum_j \exp(z_j - \max(z))}$$

<span style="font-size: 14px;">This is mathematically equivalent because the $\exp(\max(z))$ factor cancels in the numerator and denominator.</span>

---

## <span style="font-size: 16px;">Perplexity</span>

<span style="font-size: 14px;">Perplexity measures how well the model predicts the observed sequence:</span>

$$\text{PPL}(w_1 \ldots w_T) = \exp\left(-\frac{1}{N} \sum_{t=1}^{N} \log P(w_{t+1} \mid w_1 \ldots w_t)\right)$$

<span style="font-size: 14px;">where $N = T - 1$ is the number of predictions (we predict tokens 2 through $T$ given their preceding context). Lower perplexity indicates a better model. A perplexity of $k$ means the model is on average as uncertain as choosing uniformly among $k$ options.</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

* <span style="font-size: 14px;">**Why tanh activation?** Tanh bounds hidden state values to $[-1, 1]$, preventing the hidden state from growing without bound during sequential processing. ReLU can cause exploding activations in recurrent networks</span>
* <span style="font-size: 14px;">**Teacher forcing vs autoregressive generation?** Teacher forcing feeds actual tokens during evaluation and training. Autoregressive generation feeds the model's own predictions back as input, used when generating new text. The discrepancy between training and generation is called exposure bias</span>
* <span style="font-size: 14px;">**Why do vanilla RNNs struggle with long sequences?** Gradients are multiplied through the tanh Jacobian at each time step during backpropagation. For long sequences, gradients either vanish (most elements less than 1) or explode. LSTM and GRU architectures use gating mechanisms to maintain more stable gradient flow</span>
* <span style="font-size: 14px;">**Perplexity vs cross-entropy loss?** Perplexity equals $\exp(\text{cross-entropy loss})$. Cross-entropy is the standard training objective; perplexity provides a more interpretable evaluation metric</span>

---