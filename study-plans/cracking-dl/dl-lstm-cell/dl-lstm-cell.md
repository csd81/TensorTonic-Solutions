# <span style="font-size: 20px;">LSTM Cell</span>

<span style="font-size: 14px;">The Long Short-Term Memory network (Hochreiter & Schmidhuber, 1997) was designed to solve the vanishing gradient problem that cripples vanilla RNNs. By introducing a cell state with additive updates and learned gates that control information flow, the LSTM can learn dependencies spanning hundreds of time steps. It remained the dominant sequence model for nearly two decades until the Transformer.</span>

---

## <span style="font-size: 16px;">The Core Idea: Cell State Highway</span>

<span style="font-size: 14px;">The key innovation is the cell state</span> $c_t$<span style="font-size: 14px;">, a separate memory pathway that runs through the entire sequence with only element-wise operations (multiply and add). Unlike the vanilla RNN where information must pass through a tanh and matrix multiplication at every step, the cell state provides a "highway" where gradients can flow unchanged.</span>

<span style="font-size: 14px;">The cell state update is additive:</span>

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

<span style="font-size: 14px;">When the forget gate</span> $f_t \approx 1$ <span style="font-size: 14px;">and the input gate</span> $i_t \approx 0$<span style="font-size: 14px;">, cell state passes through unchanged. This is in contrast to the vanilla RNN's multiplicative update</span> $h_t = \tanh(W \cdot h_{t-1} + ...)$ <span style="font-size: 14px;">where gradients inevitably shrink.</span>

---

## <span style="font-size: 16px;">The Four Gates</span>

<span style="font-size: 14px;">Let</span> $z = [h_{t-1}, x_t]$ <span style="font-size: 14px;">be the concatenation of previous hidden state and current input:</span>

<span style="font-size: 14px;">**Forget gate** - decides what to discard from cell state:</span>

$$
f_t = \sigma(W_f \cdot z + b_f)
$$

<span style="font-size: 14px;">**Input gate** - decides which new information to store:</span>

$$
i_t = \sigma(W_i \cdot z + b_i)
$$

<span style="font-size: 14px;">**Candidate cell state** - proposes new information:</span>

$$
\tilde{c}_t = \tanh(W_c \cdot z + b_c)
$$

<span style="font-size: 14px;">**Output gate** - decides what to expose as hidden state:</span>

$$
o_t = \sigma(W_o \cdot z + b_o)
$$

<span style="font-size: 14px;">The sigmoid gates output values in $(0, 1)$, acting as soft switches. The candidate uses tanh to propose values in $(-1, 1)$. The output gate filters the cell state to produce the hidden state:</span> $h_t = o_t \odot \tanh(c_t)$.

---

## <span style="font-size: 16px;">BPTT Through the LSTM</span>

<span style="font-size: 14px;">The backward pass is more complex than vanilla RNN because gradients flow through two pathways: hidden state and cell state.</span>

<span style="font-size: 14px;">At each time step $t$ (traversing backward), let $\delta_h = \frac{\partial L}{\partial h_t} + \delta_{h,\text{future}}$:</span>

<span style="font-size: 14px;">**Through the output gate:**</span>

$$
\begin{aligned}
\delta_o = \delta_h \odot \tanh(c_t), \\
\delta_c = \delta_h \odot o_t \odot (1 - \tanh^2(c_t)) + \delta_{c,\text{future}}
\end{aligned}
$$

<span style="font-size: 14px;">**Through the cell state update:**</span>

$$
\begin{aligned}
\delta_f = \delta_c \odot c_{t-1}, \\
\delta_i = \delta_c \odot \tilde{c}_t, \\
\delta_{\tilde{c}} = \delta_c \odot i_t
\end{aligned}
$$

<span style="font-size: 14px;">**Through the activations (sigmoid and tanh derivatives):**</span>

$$
\bar{\delta}_f = \delta_f \odot f_t \odot (1 - f_t), \quad \bar{\delta}_i = \delta_i \odot i_t \odot (1 - i_t)
$$

$$
\begin{aligned}
\bar{\delta}_{\tilde{c}} = \delta_{\tilde{c}} \odot (1 - \tilde{c}_t^2), \\
\bar{\delta}_o = \delta_o \odot o_t \odot (1 - o_t)
\end{aligned}
$$

<span style="font-size: 14px;">The cell state gradient flowing to the previous step is</span> $\delta_{c,\text{future}} = \delta_c \odot f_t$<span style="font-size: 14px;">. Notice this is a multiplication by the forget gate value - when</span> $f_t \approx 1$<span style="font-size: 14px;">, the gradient passes through almost unchanged. This is why LSTMs preserve long-range gradients.</span>

---

## <span style="font-size: 16px;">Why LSTM Solves Vanishing Gradients</span>

<span style="font-size: 14px;">In a vanilla RNN, the gradient from step $t$ to step $k$ is:</span>

$$
\frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^{t} \text{diag}(1 - h_i^2) \cdot W_{hh}
$$

<span style="font-size: 14px;">This product shrinks exponentially because $|\tanh'| \leq 1$ and $||W_{hh}|| < 1$ typically.</span>

<span style="font-size: 14px;">In an LSTM, the cell state gradient is:</span>

$$
\frac{\partial c_t}{\partial c_k} = \prod_{i=k+1}^{t} f_i
$$

<span style="font-size: 14px;">When the network learns to set forget gates close to 1, this product stays close to 1 regardless of $t - k$. The forget gate bias is often initialized to a positive value (1 or 2) to encourage this behavior from the start of training (Jozefowicz et al., 2015).</span>

---

## <span style="font-size: 16px;">Practical Considerations</span>

- <span style="font-size: 14px;">**Forget gate bias initialization**: setting $b_f$ to 1 or 2 at initialization ensures the forget gate starts near 1, allowing gradients to flow freely before the network has learned what to forget</span>
- <span style="font-size: 14px;">**Peephole connections**: a variant where gates also look at the cell state directly: $f_t = \sigma(W_f \cdot z + w_{cf} \odot c_{t-1} + b_f)$. Rarely used in practice</span>
- <span style="font-size: 14px;">**Coupled forget-input gate**: use $i_t = 1 - f_t$ to reduce parameters. The intuition is that you should only write new information when you forget old information</span>
- <span style="font-size: 14px;">**Weight concatenation**: in practice, all four weight matrices are concatenated into a single $W \in \mathbb{R}^{4H \times (H+D)}$ for a single matrix multiply, then the result is split into four gate pre-activations. This is more efficient and is how PyTorch implements it</span>
- <span style="font-size: 14px;">**Gradient clipping**: still recommended for LSTM training. While the cell state highway prevents vanishing, exploding gradients are still possible through the gates</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: How does the forget gate solve vanishing gradients?**</span>
  <span style="font-size: 14px;">A: The cell state gradient from step $t$ to step $k$ is $\prod f_i$. When forget gates are close to 1, this product stays near 1 regardless of distance. This is an additive gradient path, unlike the vanilla RNN's multiplicative path through $W_{hh}$ and tanh derivatives.</span>

- <span style="font-size: 14px;">**Q: What is the difference between LSTM and GRU?**</span>
  <span style="font-size: 14px;">A: GRU has 2 gates (reset, update) vs LSTM's 3 independent gates (forget, input, output) plus cell state. GRU merges the cell and hidden state into one, and couples the forget/input decisions via $z_t$ and $1-z_t$. GRU has fewer parameters and trains faster, but LSTM can be more expressive for complex tasks.</span>

- <span style="font-size: 14px;">**Q: Why initialize the forget gate bias to a positive value?**</span>
  <span style="font-size: 14px;">A: With zero bias, $\sigma(0) = 0.5$, so the network starts by forgetting half the cell state at every step. Initializing $b_f = 1$ or $2$ makes $\sigma(b_f) \approx 0.73$ or $0.88$, allowing long-range gradients from the start. Without this, the LSTM may fail to learn long dependencies early in training.</span>

- <span style="font-size: 14px;">**Q: How does the LSTM decide what to remember vs forget?**</span>
  <span style="font-size: 14px;">A: The forget gate $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$ examines the current input and previous hidden state to decide which cell state dimensions to keep. The input gate $i_t$ similarly decides which dimensions to update with new information. These are learned independently, so the network can simultaneously forget some information and remember other information in different dimensions.</span>

- <span style="font-size: 14px;">**Q: Why did Transformers replace LSTMs?**</span>
  <span style="font-size: 14px;">A: LSTMs process sequences sequentially (each step depends on the previous), preventing parallelization. Transformers compute attention over all positions simultaneously, enabling massive GPU parallelism. Transformers also access any position directly via attention (O(1) path length) rather than through a chain of hidden states. The trade-off is quadratic memory in sequence length, which sparse attention and other techniques address.</span>

---