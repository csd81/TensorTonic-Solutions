# <span style="font-size: 20px;">Vanilla RNN Cell</span>

<span style="font-size: 14px;">The Recurrent Neural Network (Elman, 1990) introduced the idea of maintaining a hidden state that evolves over time, allowing neural networks to process sequential data. Despite being largely superseded by LSTMs and Transformers, the vanilla RNN remains the foundation for understanding all sequence models and is a staple of DL interviews.</span>

---

## <span style="font-size: 16px;">The Recurrence</span>

<span style="font-size: 14px;">At each time step</span> $t$<span style="font-size: 14px;">, the RNN combines two sources of information:</span>

$$
h_t = \tanh(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b_h)
$$

- <span style="font-size: 14px;">$W_{xh} \in \mathbb{R}^{H \times D}$ maps the input $x_t \in \mathbb{R}^D$ into hidden space</span>
- <span style="font-size: 14px;">$W_{hh} \in \mathbb{R}^{H \times H}$ transforms the previous hidden state $h_{t-1} \in \mathbb{R}^H$</span>
- <span style="font-size: 14px;">The tanh squashes the result to $(-1, 1)$, preventing unbounded growth of activations</span>

<span style="font-size: 14px;">The same weights are shared across all time steps (weight tying). This gives RNNs the ability to process sequences of arbitrary length with a fixed number of parameters.</span>

<span style="font-size: 14px;">The hidden state $h_t$ serves as the network's "memory" - it is a compressed representation of everything the network has seen from $x_1$ through $x_t$. In practice, this compression is lossy, and vanilla RNNs struggle to remember information from many time steps ago.</span>

---

## <span style="font-size: 16px;">Backpropagation Through Time (BPTT)</span>

<span style="font-size: 14px;">Training an RNN requires computing gradients with respect to the shared weights. Since the same weights are used at every time step, we unroll the computation graph across time and apply the chain rule - this is called Backpropagation Through Time.</span>

<span style="font-size: 14px;">Given upstream gradients $\partial L / \partial h_t$ at each time step, we traverse backward from $t = T$ to $t = 1$:</span>

<span style="font-size: 14px;">**Step 1**: Total gradient at $h_t$ combines the external signal and the gradient flowing back from future:</span>

$$
\delta_t = \frac{\partial L}{\partial h_t} + \frac{\partial h_{t+1}}{\partial h_t}^T \cdot \delta_{t+1}
$$

<span style="font-size: 14px;">**Step 2**: Gradient through tanh:</span>

$$
\bar{\delta}_t = (1 - h_t^2) \odot \delta_t
$$

<span style="font-size: 14px;">**Step 3**: Accumulate parameter gradients:</span>

$$
\begin{aligned}
\frac{\partial L}{\partial W_{hh}} &\mathrel{+}= \bar{\delta}_t \cdot h_{t-1}^T \\[6pt]
\frac{\partial L}{\partial W_{xh}} &\mathrel{+}= \bar{\delta}_t \cdot x_t^T \\[6pt]
\frac{\partial L}{\partial b_h} &\mathrel{+}= \bar{\delta}_t
\end{aligned}
$$

<span style="font-size: 14px;">**Step 4**: Propagate gradient to previous hidden state and input:</span>

$$
\begin{aligned}
\delta_{t-1}^{\text{future}} = W_{hh}^T \cdot \bar{\delta}_t, \\
\frac{\partial L}{\partial x_t} = W_{xh}^T \cdot \bar{\delta}_t
\end{aligned}
$$

<span style="font-size: 14px;">The key insight is that gradients at each time step accumulate into the same weight matrices (since weights are shared). This is why we use += for parameter gradients.</span>

---

## <span style="font-size: 16px;">The Vanishing Gradient Problem</span>

<span style="font-size: 14px;">The gradient flowing from time step $t$ back to time step $k$ passes through $t - k$ tanh and matrix multiplication operations:</span>

$$
\frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^{t} \text{diag}(1 - h_i^2) \cdot W_{hh}
$$

<span style="font-size: 14px;">Since $|\tanh'(x)| \leq 1$ and the largest singular value of $W_{hh}$ is typically less than 1, this product shrinks exponentially with $t - k$. For sequences longer than about 10-20 steps, gradients from distant time steps effectively vanish to zero.</span>

<span style="font-size: 14px;">Consequences:</span>
- <span style="font-size: 14px;">The network cannot learn long-range dependencies (e.g., relating a pronoun to its antecedent 50 words earlier)</span>
- <span style="font-size: 14px;">Training focuses on short-term patterns because only recent gradients have meaningful magnitude</span>
- <span style="font-size: 14px;">This motivated the development of LSTM (Hochreiter & Schmidhuber, 1997) with its cell state highway, and later the Transformer's direct attention connections</span>

<span style="font-size: 14px;">If the spectral radius of $W_{hh}$ is greater than 1, the opposite problem occurs: gradients explode. Gradient clipping (capping the norm of the gradient vector) is the standard remedy.</span>

---

## <span style="font-size: 16px;">Practical Considerations</span>

- <span style="font-size: 14px;">**Truncated BPTT**: Instead of backpropagating through the entire sequence, truncate the gradient computation to the last $k$ steps. This trades off long-range learning for computational efficiency and is standard in practice</span>
- <span style="font-size: 14px;">**Gradient clipping**: Clip the gradient norm to a threshold (typically 1.0 or 5.0) to prevent exploding gradients. This does not fix vanishing gradients</span>
- <span style="font-size: 14px;">**Bidirectional RNNs**: Run two RNNs (forward and backward) and concatenate their hidden states. This gives each position access to both past and future context</span>
- <span style="font-size: 14px;">**Deep RNNs**: Stack multiple RNN layers where the hidden states of one layer become the inputs to the next. Typically 2-4 layers deep</span>
- <span style="font-size: 14px;">**Orthogonal initialization**: Initializing $W_{hh}$ as an orthogonal matrix (all singular values = 1) helps maintain gradient magnitude through time</span>

---


## <span style="font-size: 16px;">Modern Alternatives and Context</span>

<span style="font-size: 14px;">The vanilla RNN is rarely used in production today, but understanding it is essential because every recurrent architecture builds on the same core idea: maintaining hidden state across timesteps.</span>

<span style="font-size: 14px;">**LSTM and GRU** address the vanishing gradient problem through gating mechanisms that control information flow. LSTMs use three gates (forget, input, output) plus a cell state; GRUs simplify this to two gates (reset, update) with comparable performance on many tasks.</span>

<span style="font-size: 14px;">**Transformers** have largely replaced RNNs for sequence modeling because self-attention processes all positions in parallel, avoiding the sequential bottleneck of recurrence. However, RNN-inspired architectures are making a comeback: state-space models like Mamba combine the linear-time inference of RNNs with the training parallelism of transformers. Understanding the vanilla RNN makes these developments easier to reason about.</span>


## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: Why does the vanilla RNN struggle with long sequences?**</span>
  <span style="font-size: 14px;">A: The gradient from time step $t$ to step $k$ passes through $t-k$ matrix multiplications by $W_{hh}$ and tanh derivatives. Since $|\tanh'| \leq 1$, this product shrinks exponentially, making it impossible to learn dependencies beyond roughly 10-20 steps.</span>

- <span style="font-size: 14px;">**Q: How does LSTM solve the vanishing gradient problem?**</span>
  <span style="font-size: 14px;">A: LSTM introduces a cell state $c_t$ that is updated via additive interactions (not multiplicative). The forget gate controls a direct linear path $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$. Gradients flow through the cell state with multiplication only by the forget gate values, which can stay close to 1.</span>

- <span style="font-size: 14px;">**Q: What is the difference between vanishing and exploding gradients?**</span>
  <span style="font-size: 14px;">A: Both stem from repeated multiplication by $W_{hh}$. If the spectral radius $< 1$, gradients vanish exponentially. If $> 1$, they explode. Gradient clipping fixes explosions but not vanishing. Architectural solutions (LSTM, GRU, Transformers) are needed for vanishing gradients.</span>

- <span style="font-size: 14px;">**Q: Why use tanh instead of ReLU in vanilla RNNs?**</span>
  <span style="font-size: 14px;">A: ReLU has unbounded output, so repeated application through time steps would cause hidden states to grow without bound. Tanh saturates at $\pm 1$, keeping activations bounded. However, ReLU RNNs can work with careful initialization (identity $W_{hh}$), as shown by Le et al. (2015).</span>

- <span style="font-size: 14px;">**Q: What is teacher forcing and why does it help RNN training?**</span>
  <span style="font-size: 14px;">A: In sequence generation, teacher forcing feeds the ground-truth previous token (not the model's prediction) as input at each step during training. This prevents error accumulation where one wrong prediction derails the entire sequence, but creates an exposure bias: at inference time the model must use its own predictions.</span>

---