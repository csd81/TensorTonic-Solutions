# <span style="font-size: 20px;">Complete GRU Cell</span>

<span style="font-size: 14px;">The Gated Recurrent Unit (GRU) is a recurrent neural network introduced by Cho et al. (2014) in "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation." It combines a reset gate, an update gate, and a candidate hidden state into a single cell that produces a new hidden state at each time step. The GRU achieves comparable performance to the LSTM while using fewer parameters and no separate cell state.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">The complete GRU cell is a recurrent unit that takes two inputs at each time step: the current input vector $x_t \in \mathbb{R}^D$ and the previous hidden state $h_{t-1} \in \mathbb{R}^H$, and produces a new hidden state $h_t \in \mathbb{R}^H$. Unlike the LSTM, which maintains both a hidden state and a separate cell state, the GRU uses only a single hidden state vector. All memory and gating is folded into this one vector.</span>

<span style="font-size: 14px;">The cell operates in three steps. First, the **reset gate** decides how much of the previous hidden state to expose when forming a candidate. Second, the **candidate hidden state** is computed using the reset-gated version of $h_{t-1}$. Third, the **update gate** interpolates between the old hidden state and the new candidate to produce $h_t$. These three steps use three weight matrices and three bias vectors, making the GRU simpler than the LSTM's four-gate architecture.</span>

<span style="font-size: 14px;">The paper describes this as an architecture where "gating units modulate the flow of information inside the unit, without having separate memory cells." The GRU was developed as part of an RNN Encoder-Decoder for machine translation, where the encoder reads a source sentence and the decoder generates the target one token at a time.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">Let $x_t \in \mathbb{R}^D$ be the input at time step $t$ and $h_{t-1} \in \mathbb{R}^H$ be the previous hidden state. The concatenation $[h_{t-1}, x_t] \in \mathbb{R}^{H+D}$ is formed by stacking these two vectors. All four equations execute in sequence at each time step.</span>

<span style="font-size: 14px;">**Equation 1 -- Reset gate.** Determines how much of the previous hidden state to forget when computing the candidate:</span>

$$
r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)
$$

<span style="font-size: 14px;">Here $W_r \in \mathbb{R}^{H \times (H+D)}$ and $b_r \in \mathbb{R}^H$. The sigmoid squashes each element to $(0, 1)$. When $r_t \approx 0$, the previous hidden state is effectively erased before forming the candidate, allowing the unit to behave as if reading the first token of a new subsequence.</span>

<span style="font-size: 14px;">**Equation 2 -- Update gate.** Controls how much of the old hidden state to carry forward versus replace with new content:</span>

$$
z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)
$$

<span style="font-size: 14px;">Here $W_z \in \mathbb{R}^{H \times (H+D)}$ and $b_z \in \mathbb{R}^H$. The update gate serves a dual role corresponding to both the forget and input gates in an LSTM. When $z_t \approx 1$, the cell copies the old hidden state. When $z_t \approx 0$, the cell replaces the old state entirely with the candidate.</span>

<span style="font-size: 14px;">**Equation 3 -- Candidate hidden state.** Proposes new content to write into the hidden state:</span>

$$
\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)
$$

<span style="font-size: 14px;">Here $W_h \in \mathbb{R}^{H \times (H+D)}$ and $b_h \in \mathbb{R}^H$. The element-wise product $r_t \odot h_{t-1}$ applies the reset gate, scaling each hidden dimension independently. The tanh activation bounds the candidate to $(-1, 1)$. This is the only equation where the reset gate appears: it modifies the recurrent input before the linear transformation.</span>

<span style="font-size: 14px;">**Equation 4 -- Final hidden state.** Interpolates between old and new:</span>

$$
h_t = z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h}_t
$$

<span style="font-size: 14px;">This linear interpolation uses the update gate as a mixing coefficient. Each hidden dimension independently decides its own mix. The constraint that coefficients sum to $1$ means the GRU cannot simultaneously amplify both old and new information in the same dimension, acting as an implicit regularizer.</span>

---

## <span style="font-size: 16px;">The Three-Step Pipeline</span>

### <span style="font-size: 14px;">Step 1: Reset and Compute Candidate</span>

<span style="font-size: 14px;">The reset gate $r_t$ is computed from the full concatenation $[h_{t-1}, x_t]$, then applied element-wise to $h_{t-1}$ to produce a "reset" version of the previous state. This reset state is concatenated with $x_t$ and passed through a tanh-activated affine transform to produce $\tilde{h}_t$. The reset gate lets the network learn to selectively ignore parts of the history when forming new proposals. For language modeling, this allows the GRU to treat a sentence boundary as a partial reset without discarding everything.</span>

### <span style="font-size: 14px;">Step 2: Update Gate Controls Memory Persistence</span>

<span style="font-size: 14px;">The update gate $z_t$ is computed in parallel with the reset gate (same concatenation, different weights). A high update gate value means "keep the old value," while a low value means "accept the new candidate." This is the GRU's mechanism for long-range memory: dimensions with $z_t \approx 1$ persist across many time steps, since the interpolation reduces to $h_t \approx h_{t-1}$ for those dimensions.</span>

### <span style="font-size: 14px;">Step 3: Interpolation and Data Flow</span>

<span style="font-size: 14px;">The final state $h_t = z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h}_t$ has an important gradient property. During BPTT, the gradient with respect to $h_{t-1}$ includes a direct path through $z_t \odot h_{t-1}$. When $z_t$ is close to $1$, the gradient passes through nearly unchanged, mitigating the vanishing gradient problem. This additive structure is the same principle that makes LSTM cell states effective at preserving gradients across long sequences.</span>

<span style="font-size: 14px;">The complete data flow: concatenate $h_{t-1}$ and $x_t$, compute $r_t$ and $z_t$ (two independent affine + sigmoid), apply $r_t$ to $h_{t-1}$, concatenate with $x_t$, compute $\tilde{h}_t$ via affine + tanh, then interpolate. The output $h_t$ serves as both the cell's output and the recurrent state for the next time step.</span>

---

## <span style="font-size: 16px;">Why No Separate Cell State</span>

<span style="font-size: 14px;">The LSTM maintains two state vectors: the hidden state $h_t$ (exposed to the outside) and the cell state $c_t$ (internal memory). The output gate filters the cell state to produce $h_t = o_t \odot \tanh(c_t)$, so the LSTM can store information not yet reflected in its output.</span>

<span style="font-size: 14px;">The GRU merges these roles into a single vector $h_t$. Whatever the GRU remembers is always directly available as output. This reduces the state carried between time steps by half, simplifies gradient flow to a single path, and eliminates the output gate entirely. The trade-off is expressiveness: the LSTM can store "private" information in $c_t$ that only becomes visible when the output gate opens. The GRU has no such private memory. Empirically, this matters on tasks requiring complex state tracking but is negligible on many standard benchmarks.</span>

---

## <span style="font-size: 16px;">Parameter Count</span>

<span style="font-size: 14px;">The GRU has three weight matrices and three bias vectors. Each weight matrix has shape $(H, H + D)$ and each bias vector has shape $(H,)$.</span>

* <span style="font-size: 14px;">**Reset gate:** $W_r$ has $H \times (H + D)$ weights, $b_r$ has $H$ biases</span>
* <span style="font-size: 14px;">**Update gate:** $W_z$ has $H \times (H + D)$ weights, $b_z$ has $H$ biases</span>
* <span style="font-size: 14px;">**Candidate:** $W_h$ has $H \times (H + D)$ weights, $b_h$ has $H$ biases</span>

<span style="font-size: 14px;">**Total GRU parameters:**</span>

$$
3H(H + D) + 3H
$$

<span style="font-size: 14px;">The LSTM has four gates with the same shapes, totaling $4H(H + D) + 4H$. The GRU uses exactly $\frac{3}{4}$ of the LSTM's parameters. With $H = 256$ and $D = 128$: GRU has $3 \times 256 \times 384 + 768 = 295{,}680$ parameters, the equivalent LSTM has $4 \times 256 \times 384 + 1{,}024 = 394{,}240$. The GRU saves roughly 99,000 parameters per layer.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Cho et al. (2014) introduced the GRU in "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation." The paper proposed an RNN Encoder-Decoder where the encoder reads a source sentence token by token, compressing it into a fixed-length vector, and the decoder generates the target sentence from that vector. The GRU was the recurrent unit in both components.</span>

<span style="font-size: 14px;">The paper was motivated by the failure of vanilla RNNs to capture long-range dependencies. Machine translation requires remembering subject-verb agreement and syntactic structures across many tokens. The vanishing gradient problem makes this effectively impossible for longer sentences. The GRU's gating mechanism was designed as a simpler alternative to the LSTM that could still maintain information over long spans.</span>

<span style="font-size: 14px;">The broader contribution was demonstrating that learned phrase representations from the encoder-decoder could improve phrase-based statistical machine translation. The GRU encoder-decoder scored phrase pairs, and these scores combined with traditional features improved BLEU on WMT'14 English-to-French. Chung et al. (2014) later compared GRU and LSTM across music, speech, and language modeling, concluding that neither consistently dominates, but the GRU's fewer parameters provide a practical efficiency advantage.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a GRU with $H = 2$ and $D = 2$. At time step $t$:</span>

$$
x_t = \begin{pmatrix} 0.5 \\ -0.3 \end{pmatrix}, \quad h_{t-1} = \begin{pmatrix} 0.1 \\ -0.2 \end{pmatrix}
$$

<span style="font-size: 14px;">The concatenation is $[h_{t-1}, x_t] = [0.1, -0.2, 0.5, -0.3]$. The weight matrices and biases:</span>

$$
W_r = \begin{pmatrix} 0.3 & -0.1 & 0.2 & 0.4 \\ 0.1 & 0.5 & -0.3 & 0.2 \end{pmatrix}, \quad b_r = \begin{pmatrix} 0.1 \\ -0.1 \end{pmatrix}
$$

$$
W_z = \begin{pmatrix} 0.2 & 0.3 & -0.1 & 0.5 \\ -0.2 & 0.1 & 0.4 & -0.3 \end{pmatrix}, \quad b_z = \begin{pmatrix} 0.0 \\ 0.1 \end{pmatrix}
$$

$$
W_h = \begin{pmatrix} -0.1 & 0.4 & 0.3 & -0.2 \\ 0.2 & -0.3 & 0.1 & 0.5 \end{pmatrix}, \quad b_h = \begin{pmatrix} 0.0 \\ 0.1 \end{pmatrix}
$$

<span style="font-size: 14px;">**Step 1: Reset gate.** $W_r \cdot [h, x] + b_r$:</span>

<span style="font-size: 14px;">Row 1: $0.3(0.1) + (-0.1)(-0.2) + 0.2(0.5) + 0.4(-0.3) + 0.1 = 0.03 + 0.02 + 0.10 - 0.12 + 0.1 = 0.13$</span>

<span style="font-size: 14px;">Row 2: $0.1(0.1) + 0.5(-0.2) + (-0.3)(0.5) + 0.2(-0.3) - 0.1 = 0.01 - 0.10 - 0.15 - 0.06 - 0.1 = -0.40$</span>

<span style="font-size: 14px;">$r_t = [\sigma(0.13), \sigma(-0.40)] = [0.5324, 0.4013]$</span>

<span style="font-size: 14px;">**Step 2: Update gate.** $W_z \cdot [h, x] + b_z$:</span>

<span style="font-size: 14px;">Row 1: $0.2(0.1) + 0.3(-0.2) + (-0.1)(0.5) + 0.5(-0.3) + 0.0 = 0.02 - 0.06 - 0.05 - 0.15 = -0.24$</span>

<span style="font-size: 14px;">Row 2: $(-0.2)(0.1) + 0.1(-0.2) + 0.4(0.5) + (-0.3)(-0.3) + 0.1 = -0.02 - 0.02 + 0.20 + 0.09 + 0.1 = 0.35$</span>

<span style="font-size: 14px;">$z_t = [\sigma(-0.24), \sigma(0.35)] = [0.4403, 0.5866]$</span>

<span style="font-size: 14px;">**Step 3: Candidate.** $r_t \odot h_{t-1} = [0.0532, -0.0803]$. Concatenate with $x_t$: $[0.0532, -0.0803, 0.5, -0.3]$. Compute $W_h \cdot [\cdot] + b_h$:</span>

<span style="font-size: 14px;">Row 1: $(-0.1)(0.0532) + 0.4(-0.0803) + 0.3(0.5) + (-0.2)(-0.3) + 0.0 = -0.005 - 0.032 + 0.15 + 0.06 = 0.173$</span>

<span style="font-size: 14px;">Row 2: $0.2(0.0532) + (-0.3)(-0.0803) + 0.1(0.5) + 0.5(-0.3) + 0.1 = 0.011 + 0.024 + 0.05 - 0.15 + 0.1 = 0.035$</span>

<span style="font-size: 14px;">$\tilde{h}_t = [\tanh(0.173), \tanh(0.035)] = [0.1714, 0.0350]$</span>

<span style="font-size: 14px;">**Step 4: Interpolation.**</span>

<span style="font-size: 14px;">$h_t[0] = 0.4403 \times 0.1 + 0.5597 \times 0.1714 = 0.0440 + 0.0959 = 0.1400$</span>

<span style="font-size: 14px;">$h_t[1] = 0.5866 \times (-0.2) + 0.4134 \times 0.0350 = -0.1173 + 0.0145 = -0.1028$</span>

<span style="font-size: 14px;">Final: $h_t = [0.1400, -0.1028]$. Dimension 0 ($z_t = 0.44$) absorbed more from the candidate, while dimension 1 ($z_t = 0.59$) retained more of the old state.</span>

---

## <span style="font-size: 16px;">GRU vs LSTM Comparison</span>

<span style="font-size: 14px;">**Parameters.** GRU: 3 weight matrices of shape $(H, H+D)$, 3 biases. LSTM: 4 of each. Parameter ratio is exactly $3/4$. For $H = 1024$, $D = 512$, the GRU saves over 1.5 million parameters per layer.</span>

<span style="font-size: 14px;">**Gates.** The LSTM has four gates: forget, input, candidate cell, output. The GRU has three: reset, update, candidate hidden. The update gate performs the combined role of LSTM's forget and input gates: $z_t$ controls both forgetting ($z_t \odot h_{t-1}$) and acceptance ($(1 - z_t) \odot \tilde{h}_t$). The LSTM decouples these, allowing $f_t$ and $i_t$ to vary independently.</span>

<span style="font-size: 14px;">**Cell state.** The LSTM maintains a separate $c_t$ shielded by the output gate. The GRU has no cell state. The LSTM can store information internally that is not yet reflected in its output; the GRU's memory is always fully exposed.</span>

<span style="font-size: 14px;">**Performance.** On most benchmarks (language modeling, speech, translation), GRU and LSTM perform comparably. The GRU trains faster per epoch due to fewer parameters. On tasks requiring precise internal state management, the LSTM sometimes edges ahead.</span>

<span style="font-size: 14px;">**When to prefer which.** Use the GRU when computational budget is limited or datasets are small (fewer parameters reduce overfitting). Use the LSTM when the task involves long sequences with intricate dependencies or maximum model capacity is desired.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Applying the reset gate to the wrong thing.** The reset gate must be applied to $h_{t-1}$ before concatenation with $x_t$: the candidate uses $[r_t \odot h_{t-1}, x_t]$. Applying reset after concatenation ($r_t \odot [h_{t-1}, x_t]$) incorrectly gates the input as well. The reset gate should only control how much history to use, not how much current input to accept.</span>

* <span style="font-size: 14px;">**Swapping the update gate interpolation direction.** The correct formula is $h_t = z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h}_t$. Reversing this to $(1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$ means high $z_t$ replaces old state instead of preserving it, inverting the memory behavior. The network may still train, but long-range dependencies become much harder to capture.</span>

* <span style="font-size: 14px;">**Inconsistent concatenation order across gates.** All three affine transformations must use the same concatenation order. If the reset and update gates use $[h_{t-1}, x_t]$ but the candidate uses $[x_t, r_t \odot h_{t-1}]$, $W_h$ applies learned weights to the wrong inputs. This causes no dimension error but produces subtly incorrect results.</span>

* <span style="font-size: 14px;">**Dimension mismatch in weight matrices.** Each weight matrix must have shape $(H, H+D)$, not $(H+D, H)$ or $(H, H)$. If $D \neq H$, using square matrices causes a runtime error. Verify that $W \cdot [h, x]$ is $(H, H+D) \times (H+D, 1) = (H, 1)$.</span>

* <span style="font-size: 14px;">**Using raw $h_{t-1}$ in the candidate instead of $r_t \odot h_{t-1}$.** The candidate requires the reset-gated hidden state. Using raw $h_{t-1}$ makes the reset gate a no-op, reducing the GRU to a two-gate architecture that cannot selectively forget history.</span>

* <span style="font-size: 14px;">**Confusing element-wise product with matrix multiplication.** The operations $r_t \odot h_{t-1}$ and $z_t \odot h_{t-1}$ are element-wise (Hadamard) products, not matrix multiplications. Both operands have shape $(H,)$ and produce shape $(H,)$. Matrix multiplication would yield a scalar or fail entirely.</span>

---