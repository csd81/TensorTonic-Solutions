# <span style="font-size: 20px;">Update Gate</span>

<span style="font-size: 14px;">The update gate is one of two gating mechanisms inside the Gated Recurrent Unit (GRU) introduced by Cho et al. (2014). It controls how much of the previous hidden state $h_{t-1}$ to carry forward versus how much of the newly computed candidate hidden state $\tilde{h}_t$ to accept. The paper describes it as performing "leaky integration," producing a smooth, learned interpolation between remembering and updating at every time step.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">The update gate is a vector $z_t \in \mathbb{R}^d$ (where $d$ is the hidden dimension) whose elements lie in $[0, 1]$ thanks to the sigmoid activation. At each time step $t$, the GRU uses $z_t$ to decide, dimension by dimension, whether to keep the old hidden state or replace it with a new candidate. When $z_t^{(j)} = 1$ for some dimension $j$, the GRU copies $h_{t-1}^{(j)}$ forward unchanged. When $z_t^{(j)} = 0$, it replaces that dimension entirely with the candidate $\tilde{h}_t^{(j)}$. Values between zero and one produce a weighted blend.</span>

<span style="font-size: 14px;">This is the mechanism that allows GRUs to capture long-range dependencies. Without the update gate, the hidden state would be overwritten at every step. With it, the network can learn to hold certain dimensions constant for hundreds of steps while freely updating others, adapting its memory retention on a per-dimension, per-time-step basis.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

### <span style="font-size: 14px;">Concatenation</span>

<span style="font-size: 14px;">The update gate operates on the concatenation of the previous hidden state $h_{t-1} \in \mathbb{R}^d$ and the current input $x_t \in \mathbb{R}^n$:</span>

$$
[h_{t-1}, x_t] \in \mathbb{R}^{d+n}
$$

<span style="font-size: 14px;">Both the current input and the previous hidden state jointly determine how much to update. The gate can learn patterns like "when the input signals a paragraph boundary, open the gate to accept new information" or "when the input is uninformative, keep the gate closed to preserve existing state."</span>

### <span style="font-size: 14px;">Linear Transformation and Sigmoid</span>

<span style="font-size: 14px;">The concatenated vector is multiplied by a learned weight matrix $W_z \in \mathbb{R}^{d \times (d+n)}$, a bias $b_z \in \mathbb{R}^d$ is added, and the result is passed through the element-wise sigmoid:</span>

$$
z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)
$$

<span style="font-size: 14px;">The weight matrix $W_z$ can be decomposed into two blocks: $W_{zh} \in \mathbb{R}^{d \times d}$ acting on $h_{t-1}$ and $W_{zx} \in \mathbb{R}^{d \times n}$ acting on $x_t$. Some implementations write this as $W_z x_t + U_z h_{t-1} + b_z$, where $U_z$ is the hidden-to-hidden weight block. Both formulations are mathematically identical. The sigmoid $\sigma(a) = 1/(1 + e^{-a})$ squashes each element into $(0, 1)$, producing valid interpolation coefficients.</span>

---

## <span style="font-size: 16px;">The Interpolation Mechanism</span>

<span style="font-size: 14px;">Once $z_t$ is computed, the GRU produces the final hidden state by linearly interpolating between the previous state $h_{t-1}$ and the candidate $\tilde{h}_t$:</span>

$$
h_t = z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h}_t
$$

<span style="font-size: 14px;">Here $\odot$ denotes element-wise (Hadamard) multiplication. This is a convex combination: for each dimension $j$, $h_t^{(j)} = z_t^{(j)} \cdot h_{t-1}^{(j)} + (1 - z_t^{(j)}) \cdot \tilde{h}_t^{(j)}$. The coefficients $z_t^{(j)}$ and $1 - z_t^{(j)}$ always sum to one.</span>

### <span style="font-size: 14px;">When $z_t = 1$: Perfect Memory</span>

<span style="font-size: 14px;">If $z_t^{(j)} = 1$, then $h_t^{(j)} = h_{t-1}^{(j)}$. The previous state is copied forward with no modification. The candidate is computed but completely ignored. A dimension with $z_t \approx 1$ at every step carries its value across potentially hundreds of time steps unchanged.</span>

### <span style="font-size: 14px;">When $z_t = 0$: Complete Replacement</span>

<span style="font-size: 14px;">If $z_t^{(j)} = 0$, then $h_t^{(j)} = \tilde{h}_t^{(j)}$. The old state is discarded entirely. This allows the GRU to rapidly adapt when the input requires a fresh representation, for instance at a sentence boundary or topic shift.</span>

### <span style="font-size: 14px;">Intermediate Values: Soft Blending</span>

<span style="font-size: 14px;">For $z_t^{(j)} = 0.7$, the GRU keeps $70\%$ of the old state and mixes in $30\%$ of the candidate. This gradual blending is differentiable and trainable by backpropagation through time. The network does not need to make hard binary decisions; it can learn smooth transitions that balance stability with plasticity.</span>

---

## <span style="font-size: 16px;">Why Leaky Integration</span>

<span style="font-size: 14px;">Cho et al. (2014) specifically describe the update gate as performing "leaky integration." A leaky integrator is a system whose state decays exponentially unless refreshed by new input. The canonical form is $s_t = \alpha \cdot s_{t-1} + (1 - \alpha) \cdot x_t$, where $\alpha \in [0, 1]$ is a fixed decay constant. The GRU's update equation matches this structure exactly, with two crucial generalizations: the leak rate $z_t$ is a learned, input-dependent vector that changes at every time step, and the "input" is the candidate $\tilde{h}_t$ rather than the raw $x_t$.</span>

### <span style="font-size: 14px;">Connection to Exponential Moving Averages</span>

<span style="font-size: 14px;">An exponential moving average (EMA) with smoothing factor $\alpha$ follows $s_t = \alpha \cdot s_{t-1} + (1 - \alpha) \cdot x_t$. When $z_t$ is nearly constant (say $z_t \approx 0.9$), the hidden state behaves like an EMA with $\alpha = 0.9$: the effective memory window spans roughly $1/(1 - 0.9) = 10$ time steps. The difference is that the GRU's $z_t$ varies dynamically, allowing the effective memory window to expand or contract based on input content.</span>

### <span style="font-size: 14px;">Adaptive Time Constants Per Dimension</span>

<span style="font-size: 14px;">Because $z_t$ is a vector, each hidden dimension has its own time constant. Some dimensions might maintain $z_t^{(j)} \approx 0.99$, giving them a time constant of roughly 100 steps. Others might have $z_t^{(j)} \approx 0.1$, responding rapidly with a time constant of about 1.1 steps. This per-dimension adaptivity lets the GRU simultaneously track slow-moving context (document topic) and fast-changing features (local syntax) within the same hidden state vector.</span>

---

## <span style="font-size: 16px;">How Update Differs from Reset</span>

<span style="font-size: 14px;">The GRU has two gates: the update gate $z_t$ and the reset gate $r_t$. They serve fundamentally different roles, and confusing them is a common source of implementation errors.</span>

### <span style="font-size: 14px;">Reset Gate: Candidate Preparation</span>

<span style="font-size: 14px;">The reset gate $r_t$ operates earlier in the computation. It controls how much of $h_{t-1}$ is visible when computing the candidate:</span>

$$
\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t] + b)
$$

<span style="font-size: 14px;">When $r_t \approx 0$, the previous hidden state is masked out, forcing the candidate to depend primarily on $x_t$. When $r_t \approx 1$, the full history is available and the candidate can be a refined version of the existing state.</span>

### <span style="font-size: 14px;">Update Gate: Final Mixing</span>

<span style="font-size: 14px;">The update gate operates after the candidate has been computed. It determines how the candidate is blended into the actual hidden state. Even if the reset gate produces a radically different candidate, the update gate can reject it by setting $z_t \approx 1$. The two gates form a two-stage decision: the reset gate asks "what should the candidate look like?" and the update gate asks "how much of that candidate should I actually use?"</span>

---

## <span style="font-size: 16px;">Paper Context</span>

### <span style="font-size: 14px;">Cho et al. (2014)</span>

<span style="font-size: 14px;">The GRU was introduced in "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation." The paper's primary contribution was the encoder-decoder architecture for sequence-to-sequence modeling, and the GRU was proposed as the recurrent unit within it. The authors motivated the gating mechanism by the need to capture long-range dependencies in machine translation, where word meaning may depend on context many tokens away.</span>

<span style="font-size: 14px;">The paper explicitly states: "The update gate $z_j$ selects whether the hidden state is to be updated with a new hidden state. This acts like a leaky integration." This positions the update gate as the central mechanism for temporal memory control, with the leak rate determining the effective time scale at which the network operates.</span>

### <span style="font-size: 14px;">Simpler Than LSTM</span>

<span style="font-size: 14px;">The LSTM uses three gates (forget, input, output) plus a separate cell state. The GRU achieves similar gating with only two gates and no separate cell state. The update gate plays the combined role of the LSTM's forget and input gates: $z_t$ simultaneously controls how much to forget ($z_t$ multiplies $h_{t-1}$) and how much new information to admit ($1 - z_t$ multiplies $\tilde{h}_t$). This coupling means three weight matrices instead of four and faster training, at the cost of slightly reduced flexibility.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a GRU with hidden dimension $d = 3$ and input dimension $n = 2$.</span>

### <span style="font-size: 14px;">Given Values</span>

<span style="font-size: 14px;">Previous hidden state and current input:</span>

$$
h_{t-1} = [0.8, -0.3, 0.5], \quad x_t = [1.0, -0.5]
$$

<span style="font-size: 14px;">Candidate hidden state (already computed via the reset gate path):</span>

$$
\tilde{h}_t = [0.2, 0.9, -0.4]
$$

<span style="font-size: 14px;">Update gate weights $W_z \in \mathbb{R}^{3 \times 5}$ and bias $b_z \in \mathbb{R}^3$:</span>

$$
W_z = \begin{pmatrix} 0.3 & -0.1 & 0.4 & 0.2 & -0.3 \\ 0.1 & 0.5 & -0.2 & 0.3 & 0.1 \\ -0.2 & 0.3 & 0.6 & -0.1 & 0.4 \end{pmatrix}, \quad b_z = [0.1, -0.1, 0.0]
$$

### <span style="font-size: 14px;">Step 1: Concatenate</span>

$$
[h_{t-1}, x_t] = [0.8, -0.3, 0.5, 1.0, -0.5]
$$

### <span style="font-size: 14px;">Step 2: Linear Transformation</span>

<span style="font-size: 14px;">Dimension 1: $a_z^{(1)} = 0.3(0.8) + (-0.1)(-0.3) + 0.4(0.5) + 0.2(1.0) + (-0.3)(-0.5) + 0.1 = 0.24 + 0.03 + 0.20 + 0.20 + 0.15 + 0.1 = 0.92$</span>

<span style="font-size: 14px;">Dimension 2: $a_z^{(2)} = 0.1(0.8) + 0.5(-0.3) + (-0.2)(0.5) + 0.3(1.0) + 0.1(-0.5) + (-0.1) = 0.08 - 0.15 - 0.10 + 0.30 - 0.05 - 0.1 = -0.02$</span>

<span style="font-size: 14px;">Dimension 3: $a_z^{(3)} = (-0.2)(0.8) + 0.3(-0.3) + 0.6(0.5) + (-0.1)(1.0) + 0.4(-0.5) + 0.0 = -0.16 - 0.09 + 0.30 - 0.10 - 0.20 + 0.0 = -0.25$</span>

$$
a_z = [0.92, -0.02, -0.25]
$$

### <span style="font-size: 14px;">Step 3: Sigmoid</span>

<span style="font-size: 14px;">$z_t^{(1)} = \sigma(0.92) = 1/(1 + e^{-0.92}) = 1/1.3985 = 0.7153$</span>

<span style="font-size: 14px;">$z_t^{(2)} = \sigma(-0.02) = 1/(1 + e^{0.02}) = 1/2.0202 = 0.4950$</span>

<span style="font-size: 14px;">$z_t^{(3)} = \sigma(-0.25) = 1/(1 + e^{0.25}) = 1/2.2840 = 0.4378$</span>

$$
z_t = [0.7153, 0.4950, 0.4378]
$$

### <span style="font-size: 14px;">Step 4: Interpolation</span>

<span style="font-size: 14px;">$h_t^{(1)} = 0.7153 \cdot 0.8 + 0.2847 \cdot 0.2 = 0.5722 + 0.0569 = 0.6292$</span>

<span style="font-size: 14px;">$h_t^{(2)} = 0.4950 \cdot (-0.3) + 0.5050 \cdot 0.9 = -0.1485 + 0.4545 = 0.3060$</span>

<span style="font-size: 14px;">$h_t^{(3)} = 0.4378 \cdot 0.5 + 0.5622 \cdot (-0.4) = 0.2189 - 0.2249 = -0.0060$</span>

$$
h_t = [0.6292, 0.3060, -0.0060]
$$

### <span style="font-size: 14px;">Interpreting the Result</span>

<span style="font-size: 14px;">Dimension 1 has $z_t^{(1)} = 0.72$, so it keeps most of $h_{t-1}^{(1)} = 0.8$, resulting in $0.63$ -- close to the original. Dimension 2 has $z_t^{(2)} = 0.50$, producing an equal blend of $-0.3$ and $0.9$, yielding $0.31$. Dimension 3 has the lowest gate value ($0.44$), leaning toward the candidate $-0.4$ and pulling the result near zero. Each dimension makes its own independent keep-versus-replace decision.</span>

---

## <span style="font-size: 16px;">Connection to LSTM</span>

<span style="font-size: 14px;">The LSTM's cell state update uses two separate gates:</span>

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

<span style="font-size: 14px;">Here $f_t$ is the forget gate and $i_t$ is the input gate. Crucially, $f_t$ and $i_t$ are computed independently, so there is no constraint that $f_t + i_t = 1$. The LSTM can simultaneously forget everything ($f_t \approx 0$) and accept nothing ($i_t \approx 0$), or keep everything ($f_t \approx 1$) and also add new information ($i_t \approx 1$).</span>

<span style="font-size: 14px;">The GRU's update gate imposes $z_t + (1 - z_t) = 1$, a strict convex combination. If the GRU wants to keep more of the old state, it must proportionally accept less of the candidate. One gate ($z_t$) replaces both $f_t$ and $i_t$, reducing the parameter count by one full gate's worth of weights. The LSTM also has an output gate $o_t$ that the GRU lacks; the GRU exposes the full hidden state as output at every step. The trade-off is that the LSTM's independent gates provide strictly more representational flexibility, though in practice this rarely translates to a meaningful performance advantage.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Swapping $z_t$ and $(1 - z_t)$ in the interpolation.** The correct formula is $h_t = z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h}_t$. Writing $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$ reverses the semantics: $z_t = 1$ now means "take the candidate" instead of "keep the old state." The network can compensate by learning flipped biases, but the interpretation of gate values is inverted, making debugging confusing.</span>

* <span style="font-size: 14px;">**Confusing $z_t = 1$ with "update" instead of "keep."** The name "update gate" is misleading. When $z_t = 1$, the effect is to keep the old state, not to update it. The update (replacement with $\tilde{h}_t$) happens when $z_t = 0$. A helpful mnemonic: $z_t$ measures how much the old state "survives," not how much new information enters.</span>

* <span style="font-size: 14px;">**Using the wrong gate for the wrong equation.** The update gate $z_t$ controls the final interpolation. The reset gate $r_t$ controls candidate preparation. Using $z_t$ where $r_t$ should appear (inside $\tilde{h}_t$) or vice versa produces a model that trains but underperforms, because the two gates have different learned roles.</span>

* <span style="font-size: 14px;">**Treating $z_t$ as a scalar instead of a vector.** The update gate is a vector, not a scalar. Each hidden dimension has its own gate value. Broadcasting a single scalar across all dimensions forces every dimension to update by the same amount, collapsing the per-dimension adaptivity that gives GRUs their expressive power. The multiplication $z_t \odot h_{t-1}$ must be element-wise.</span>

* <span style="font-size: 14px;">**Neglecting bias initialization.** If $b_z$ is initialized to zero, initial gate values are $\sigma(0) = 0.5$, meaning the GRU starts by averaging old and new states equally. Initializing $b_z$ to a positive value (e.g., 1.0) pushes $z_t$ near 1, biasing the network toward preserving state early in training. This mirrors the common LSTM practice of initializing forget gate biases to 1.0 and can improve learning of long-range dependencies.</span>

* <span style="font-size: 14px;">**Forgetting that gradients flow through both paths.** During backpropagation through time, gradients reach $h_{t-1}$ via two paths: directly through $z_t \odot h_{t-1}$ and indirectly through $(1 - z_t) \odot \tilde{h}_t$ (since $\tilde{h}_t$ depends on $h_{t-1}$). The direct path provides an unobstructed gradient highway when $z_t \approx 1$, analogous to the LSTM's cell state gradient flow. This is how gated RNNs mitigate the vanishing gradient problem.</span>

---