# <span style="font-size: 20px;">GRU Cell</span>

<span style="font-size: 14px;">The Gated Recurrent Unit (Cho et al., 2014) was introduced as a simpler alternative to the LSTM. It achieves comparable performance on many tasks with fewer parameters and faster computation. The GRU merges the LSTM's cell state and hidden state into a single hidden state, and uses two gates instead of three.</span>

---

## <span style="font-size: 16px;">Architecture: Two Gates</span>

<span style="font-size: 14px;">The GRU has two gates that control information flow:</span>

<span style="font-size: 14px;">**Update gate** $z_t$ - decides how much of the new candidate to use vs keeping the old state:</span>

$$
z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)
$$

<span style="font-size: 14px;">**Reset gate** $r_t$ - decides how much of the previous hidden state to expose when computing the candidate:</span>

$$
r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)
$$

<span style="font-size: 14px;">**Candidate hidden state** - computed using the reset-gated previous state:</span>

$$
\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1},\; x_t] + b_h)
$$

<span style="font-size: 14px;">**Final update** - interpolation between old and new:</span>

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

<span style="font-size: 14px;">When $z_t \approx 0$, the hidden state is copied forward unchanged (memory). When $z_t \approx 1$, the state is replaced by the candidate (update). This interpolation is the GRU's mechanism for learning long-range dependencies.</span>

---

## <span style="font-size: 16px;">GRU vs LSTM: Structural Comparison</span>

<span style="font-size: 14px;">The GRU can be viewed as a simplified LSTM where:</span>

- <span style="font-size: 14px;">The LSTM's separate cell state and hidden state are merged into one hidden state</span>
- <span style="font-size: 14px;">The LSTM's forget gate and input gate are coupled: the GRU uses $z_t$ and $1 - z_t$ (if you forget more, you must accept more new information, and vice versa). The LSTM can independently control forgetting and input</span>
- <span style="font-size: 14px;">The LSTM's output gate is removed. The full hidden state is always exposed</span>
- <span style="font-size: 14px;">The reset gate roughly corresponds to the LSTM's forget gate, but it only affects the candidate computation, not the state update directly</span>

<span style="font-size: 14px;">Parameter count comparison for hidden size $H$ and input size $D$:</span>
- <span style="font-size: 14px;">LSTM: $4 \times H \times (H + D) + 4H$ parameters (4 gate matrices + 4 biases)</span>
- <span style="font-size: 14px;">GRU: $3 \times H \times (H + D) + 3H$ parameters (3 gate matrices + 3 biases)</span>
- <span style="font-size: 14px;">GRU uses 75% of LSTM's parameters</span>

---

## <span style="font-size: 16px;">BPTT Through the GRU</span>

<span style="font-size: 14px;">The backward pass must account for gradients flowing through three pathways at each time step. Given total gradient $\delta_h$ at step $t$:</span>

<span style="font-size: 14px;">**Through the interpolation** $h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$:</span>

$$
\begin{aligned}
\delta_z &= \delta_h \odot (\tilde{h}_t - h_{t-1}) \\[6pt]
\delta_{\tilde{h}} &= \delta_h \odot z_t \\[6pt]
\delta_{h,\text{interp}} &= \delta_h \odot (1 - z_t)
\end{aligned}
$$

<span style="font-size: 14px;">**Through the candidate** $\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)$:</span>

$$
\bar{\delta}_{\tilde{h}} = \delta_{\tilde{h}} \odot (1 - \tilde{h}_t^2)
$$

<span style="font-size: 14px;">This produces gradients for $W_h$, $b_h$, and for $r_t \odot h_{t-1}$. The gradient on the reset gate comes from the element-wise product with $h_{t-1}$.</span>

<span style="font-size: 14px;">**Three gradient paths to $h_{t-1}$**: (1) directly through the interpolation, (2) through the reset gate in the candidate computation, and (3) through the update and reset gate pre-activations. All three must be summed.</span>

---

## <span style="font-size: 16px;">When to Use GRU vs LSTM</span>

- <span style="font-size: 14px;">**GRU advantages**: fewer parameters (faster training, less overfitting on small datasets), simpler to implement, often comparable performance on many NLP tasks</span>
- <span style="font-size: 14px;">**LSTM advantages**: more expressive due to independent forget/input gates and separate cell state, better on tasks requiring fine-grained memory control (e.g., counting, precise copying)</span>
- <span style="font-size: 14px;">**Empirical findings** (Chung et al., 2014): neither consistently outperforms the other. GRU tends to do better on smaller datasets; LSTM on larger ones. The best choice depends on the task</span>
- <span style="font-size: 14px;">**In practice**: both have been largely replaced by Transformers for most tasks. GRUs remain popular in real-time applications (speech, on-device inference) where parameter efficiency matters</span>

---


## <span style="font-size: 16px;">Implementation Pitfalls</span>

<span style="font-size: 14px;">**Gate ordering matters.** The reset gate must be applied before computing the candidate hidden state. A common bug is applying it after, which means the candidate sees the full previous hidden state regardless of the reset gate value. The correct sequence is: compute reset gate, apply it to h_prev, then compute candidate.</span>

<span style="font-size: 14px;">**Bias initialization.** Initialize the update gate bias slightly positive (e.g., 1.0) so that the GRU initially passes information through, similar to the LSTM forget gate initialization trick. This helps with gradient flow early in training.</span>

<span style="font-size: 14px;">**Hidden state dimensions.** All three computations (reset gate, update gate, candidate) must produce the same hidden_size output. The weight matrices have shape (input_size + hidden_size, hidden_size) when input and hidden weights are concatenated, or separate matrices of shapes (input_size, hidden_size) and (hidden_size, hidden_size).</span>


## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: How does the GRU handle long-range dependencies without a separate cell state?**</span>
  <span style="font-size: 14px;">A: The interpolation $h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$ creates a direct additive path. When $z_t \approx 0$, the state passes through unchanged with gradient 1. This is analogous to the LSTM's cell state highway, but using the hidden state directly.</span>

- <span style="font-size: 14px;">**Q: What is the role of the reset gate?**</span>
  <span style="font-size: 14px;">A: It controls how much of the previous hidden state is visible when computing the candidate $\tilde{h}_t$. When $r_t \approx 0$, the candidate ignores the past and acts like a standard feedforward layer on $x_t$ alone. When $r_t \approx 1$, the candidate sees the full history. This allows the GRU to "start fresh" when needed.</span>

- <span style="font-size: 14px;">**Q: Why is the update gate coupled as $z_t$ and $1-z_t$?**</span>
  <span style="font-size: 14px;">A: This enforces a conservation constraint: the total "attention" to old and new information sums to 1 at each dimension. In contrast, LSTM's forget and input gates are independent, so the cell state magnitude can grow or shrink. The coupling reduces parameters and provides an implicit regularization.</span>

- <span style="font-size: 14px;">**Q: Can the GRU learn to be an identity function over long sequences?**</span>
  <span style="font-size: 14px;">A: Yes. If $z_t = 0$ everywhere, then $h_t = h_{t-1}$ and the state is copied forward unchanged. The network can learn this by setting $W_z$ and $b_z$ such that the update gate is always near 0. This is the GRU's equivalent of the LSTM setting forget gate to 1.</span>

- <span style="font-size: 14px;">**Q: How does gradient flow differ between GRU and vanilla RNN?**</span>
  <span style="font-size: 14px;">A: In a vanilla RNN, the gradient from step $t$ to step $k$ passes through $t-k$ tanh and $W_{hh}$ multiplications, causing exponential decay. In the GRU, the interpolation provides a shortcut: the gradient through $(1-z_t)$ is just a scalar multiplication per step, which can stay close to 1. This is why GRUs can learn much longer dependencies.</span>

---